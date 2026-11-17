"""
Unified 5-Upgrade Neural Architecture for C/C++ Vulnerability Detection:
  Upgrade #1: Hierarchical Chunk-Attention Aggregator (handles 1,280+ tokens)
  Upgrade #2: Siamese Counterfactual Invariance Head (JSD + Cosine consistency)
  Upgrade #3: Data-Flow & AST Edge-Biased Self-Attention (Graphormer-style bias)
  Upgrade #4: Layer-Weighted Multi-Head Attention Pooling across all 12 layers
  Upgrade #5: Joint Function-Level + Line/Token-Level Multi-Task Localization Head
"""

import contextvars
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src import config

# Thread-safe context variable for injecting per-head AST/Def-Use attention bias
# into every Transformer self-attention layer (compatible with DataParallel & BF16).
_ACTIVE_EDGE_BIAS: contextvars.ContextVar[torch.Tensor | None] = contextvars.ContextVar(
    "_ACTIVE_EDGE_BIAS", default=None
)


class _EdgeBiasedSelfAttentionWrapper(nn.Module):
    """
    Wraps a backbone RobertaSelfAttention module to add learned AST / Def-Use /
    Control-Flow relational bias B_{AST/DFG} in R^{(B*K) x H x L x L} directly to
    attention scores before softmax while preserving all pretrained weights.
    """

    def __init__(self, orig_self_attn: nn.Module) -> None:
        super().__init__()
        self.self_attn = orig_self_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple | None = None,
        output_attentions: bool = False,
    ):
        edge_bias = _ACTIVE_EDGE_BIAS.get()
        if edge_bias is not None and attention_mask is not None:
            # attention_mask has shape (B*K, 1, 1, L); edge_bias has shape (B*K, H, L, L)
            combined_mask = attention_mask.to(dtype=hidden_states.dtype) + edge_bias.to(
                dtype=hidden_states.dtype
            )
        else:
            combined_mask = attention_mask

        return self.self_attn(
            hidden_states=hidden_states,
            attention_mask=combined_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )


class MultiHeadAttentionPooling(nn.Module):
    """
    Upgrade #4b: Multi-Head Attention Pooling over all non-padding code tokens.
    Allows specialized heads to attend to allocation sites, pointer arithmetic,
    array indexing, and error-return checks rather than relying solely on [CLS].
    """

    def __init__(self, hidden_size: int, num_heads: int = config.POOLING_NUM_HEADS, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.attn_scores = nn.Linear(hidden_size, num_heads)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (N, L, D)
            attention_mask:   (N, L) with 1 for valid tokens, 0 for padding
        Returns:
            pooled:           (N, D)
        """
        N, L, D = token_embeddings.shape
        # (N, L, H) -> (N, H, L)
        scores = self.attn_scores(token_embeddings).transpose(1, 2)
        mask = attention_mask.unsqueeze(1).bool()  # (N, 1, L)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))  # (N, H, L)

        # (N, L, D) -> (N, L, H, d_h) -> (N, H, L, d_h)
        values = self.value_proj(token_embeddings).view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        # (N, H, 1, L) @ (N, H, L, d_h) -> (N, H, 1, d_h) -> (N, D)
        pooled_heads = torch.matmul(weights.unsqueeze(2), values).squeeze(2).reshape(N, D)
        return self.norm(self.out_proj(pooled_heads))


class GatedChunkAggregator(nn.Module):
    """
    Upgrade #1: Hierarchical Chunk-Attention Aggregator.
    Models cross-chunk interactions via a 2-layer Transformer Encoder over chunk
    embeddings followed by Gated Attention Pooling across active chunks.
    """

    def __init__(self, hidden_size: int, max_chunks: int = config.MAX_CHUNKS, dropout: float = 0.1) -> None:
        super().__init__()
        self.chunk_pos_emb = nn.Parameter(torch.zeros(1, max_chunks, hidden_size))
        nn.init.normal_(self.chunk_pos_emb, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.chunk_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Gated Attention Pooling (Ilse et al.) across chunks
        self.gate_v = nn.Linear(hidden_size, hidden_size // 2)
        self.gate_u = nn.Linear(hidden_size, hidden_size // 2)
        self.gate_w = nn.Linear(hidden_size // 2, 1)
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, chunk_embeddings: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_embeddings: (B, K, D)
            chunk_mask:       (B, K) with 1 for active chunks, 0 for padded chunks
        Returns:
            func_embedding:   (B, D)
        """
        B, K, D = chunk_embeddings.shape
        x = chunk_embeddings + self.chunk_pos_emb[:, :K, :]
        key_padding_mask = ~(chunk_mask.bool())  # True where padded
        x = self.chunk_transformer(x, src_key_padding_mask=key_padding_mask)

        # Gated attention weights across K chunks
        gate = torch.tanh(self.gate_v(x)) * torch.sigmoid(self.gate_u(x))
        attn_logits = self.gate_w(gate).squeeze(-1)  # (B, K)
        attn_logits = attn_logits.masked_fill(key_padding_mask, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (B, K, 1)

        func_emb = torch.sum(x * attn_weights, dim=1)  # (B, D)
        return self.out_norm(func_emb)


class HierarchicalGraphSiameseVulnDetector(nn.Module):
    """
    Complete 5-Upgrade Vulnerability Detection Model.
    """

    def __init__(
        self,
        backbone_name: str = config.MODEL_REGISTRY["unixcoder"],
        max_chunks: int = config.MAX_CHUNKS,
        num_edge_types: int = config.NUM_EDGE_TYPES,
        use_graph_bias: bool = True,
        use_layer_pooling: bool = True,
        use_hierarchical: bool = True,
        use_line_head: bool = True,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.use_graph_bias = use_graph_bias
        self.use_layer_pooling = use_layer_pooling
        self.use_hierarchical = use_hierarchical
        self.use_line_head = use_line_head

        hf_cfg = AutoConfig.from_pretrained(backbone_name)
        hf_cfg.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(backbone_name, config=hf_cfg)

        hidden_size = hf_cfg.hidden_size
        num_layers = hf_cfg.num_hidden_layers
        num_heads = hf_cfg.num_attention_heads

        # Upgrade #3: Learned per-head relational bias embedding for AST / Def-Use / Control-Flow edges
        if self.use_graph_bias:
            self.edge_bias_emb = nn.Embedding(num_edge_types, num_heads)
            nn.init.zeros_(self.edge_bias_emb.weight)
            # Wrap every self-attention layer in the RoBERTa/UniXcoder encoder
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                for layer in self.encoder.encoder.layer:
                    layer.attention.self = _EdgeBiasedSelfAttentionWrapper(layer.attention.self)

        # Upgrade #4: Learnable softmax layer-weighting + Multi-Head Token Attention Pooling
        if self.use_layer_pooling:
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
            self.token_pooler = MultiHeadAttentionPooling(
                hidden_size=hidden_size, num_heads=config.POOLING_NUM_HEADS, dropout=dropout
            )

        # Upgrade #1: Hierarchical Chunk-Attention Aggregator
        if self.use_hierarchical:
            self.chunk_aggregator = GatedChunkAggregator(
                hidden_size=hidden_size, max_chunks=max_chunks, dropout=dropout
            )

        # Function-level binary vulnerability classifier head
        self.func_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

        # Upgrade #5: Token / Line-level vulnerability localization head
        if self.use_line_head:
            self.line_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

    def encode_branch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        edge_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes a batch of multi-chunk functions (B, K, L).
        Returns:
            func_logits:   (B, 2)
            func_emb:      (B, D)
            token_logits:  (B, K, L)
        """
        B, K, L = input_ids.shape
        flat_ids = input_ids.view(B * K, L)
        flat_mask = attention_mask.view(B * K, L)

        # Compute relational edge bias (B*K, H, L, L) if enabled
        edge_bias = None
        if self.use_graph_bias and edge_types is not None:
            flat_edges = edge_types.view(B * K, L, L)
            # (B*K, L, L, H) -> (B*K, H, L, L)
            edge_bias = self.edge_bias_emb(flat_edges).permute(0, 3, 1, 2).contiguous()
            # Zero out edge bias on rows corresponding to completely padded chunks
            active_flat = chunk_mask.view(B * K, 1, 1, 1).to(edge_bias.dtype)
            edge_bias = edge_bias * active_flat

        token_ctx = _ACTIVE_EDGE_BIAS.set(edge_bias)
        try:
            outputs = self.encoder(
                input_ids=flat_ids,
                attention_mask=flat_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            _ACTIVE_EDGE_BIAS.reset(token_ctx)

        # Upgrade #4: Layer-Weighted Token Representations + Multi-Head Attention Pooling
        if self.use_layer_pooling:
            # Stack transformer layers 1..12 -> (num_layers, B*K, L, D)
            stacked_layers = torch.stack(outputs.hidden_states[1:], dim=0)
            norm_weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            token_repr = torch.sum(stacked_layers * norm_weights, dim=0)  # (B*K, L, D)
            chunk_repr = self.token_pooler(token_repr, flat_mask)         # (B*K, D)
        else:
            token_repr = outputs.last_hidden_state
            chunk_repr = token_repr[:, 0, :]

        # Upgrade #1: Hierarchical Cross-Chunk Aggregation
        chunk_repr_3d = chunk_repr.view(B, K, -1)
        if self.use_hierarchical and K > 1:
            func_emb = self.chunk_aggregator(chunk_repr_3d, chunk_mask)
        else:
            func_emb = chunk_repr_3d[:, 0, :]

        func_logits = self.func_classifier(func_emb)  # (B, 2)

        if self.use_line_head:
            token_logits = self.line_classifier(token_repr).view(B, K, L)
        else:
            token_logits = torch.zeros((B, K, L), device=func_logits.device, dtype=func_logits.dtype)

        return func_logits, func_emb, token_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        edge_types: torch.Tensor | None = None,
        anon_input_ids: torch.Tensor | None = None,
        anon_attention_mask: torch.Tensor | None = None,
        anon_chunk_mask: torch.Tensor | None = None,
        anon_edge_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        func_logits, func_emb, token_logits = self.encode_branch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_mask=chunk_mask,
            edge_types=edge_types,
        )
        out = {
            "func_logits": func_logits,
            "func_emb": func_emb,
            "token_logits": token_logits,
        }

        # Upgrade #2: Siamese Counterfactual Branch on AST-Anonymized Twin
        if anon_input_ids is not None:
            anon_logits, anon_emb, _ = self.encode_branch(
                input_ids=anon_input_ids,
                attention_mask=anon_attention_mask,
                chunk_mask=anon_chunk_mask,
                edge_types=anon_edge_types,
            )
            out["anon_func_logits"] = anon_logits
            out["anon_func_emb"] = anon_emb

        return out


def compute_multitask_siamese_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    line_labels: torch.Tensor | None = None,
    siamese_weight: float = config.LAMBDA_SIAMESE_JSD,
    cosine_weight: float = config.LAMBDA_SIAMESE_COS,
    line_weight: float = config.BETA_LINE_MULTITASK,
    pos_weight: float = config.POS_CLASS_WEIGHT,
    focal_gamma: float = config.FOCAL_GAMMA,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes the composite multi-task training objective:
      L_total = L_func (Class-Weighted Focal CE)
              + beta * L_line (Masked Token/Line BCE from PrimeVul_paired diffs)
              + lambda_jsd * D_JS(p(y|x) || p(y|x_anon))
              + lambda_cos * (1 - cos(z_func, z_anon))
    """
    func_logits = outputs["func_logits"]

    # 1. Class-weighted Focal Cross-Entropy Loss for imbalanced PrimeVul function labels
    class_weights = torch.tensor([1.0, pos_weight], device=func_logits.device, dtype=func_logits.dtype)
    ce_per_sample = F.cross_entropy(func_logits, labels, weight=class_weights, reduction="none")
    pt = torch.exp(-F.cross_entropy(func_logits, labels, reduction="none"))
    loss_func = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()

    # 2. Upgrade #5: Masked Line/Token-Level Bug Localization Loss
    loss_line = torch.tensor(0.0, device=func_logits.device, dtype=func_logits.dtype)
    if line_labels is not None and "token_logits" in outputs:
        token_logits = outputs["token_logits"]
        valid_mask = line_labels != -100
        if valid_mask.any():
            valid_logits = token_logits[valid_mask]
            valid_targets = line_labels[valid_mask].to(dtype=valid_logits.dtype)
            # Upweight positive buggy lines within vulnerable functions
            line_pos_w = torch.tensor(4.0, device=valid_logits.device, dtype=valid_logits.dtype)
            loss_line = F.binary_cross_entropy_with_logits(
                valid_logits, valid_targets, pos_weight=line_pos_w
            )

    # 3. Upgrade #2: Symmetric Jensen-Shannon Divergence + Cosine Invariance Loss
    loss_jsd = torch.tensor(0.0, device=func_logits.device, dtype=func_logits.dtype)
    loss_cos = torch.tensor(0.0, device=func_logits.device, dtype=func_logits.dtype)
    if "anon_func_logits" in outputs and siamese_weight > 0.0:
        p = torch.softmax(func_logits.float(), dim=-1).clamp(min=1e-6, max=1.0 - 1e-6)
        q = torch.softmax(outputs["anon_func_logits"].float(), dim=-1).clamp(min=1e-6, max=1.0 - 1e-6)
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * torch.log(p / m), dim=-1)
        kl_qm = torch.sum(q * torch.log(q / m), dim=-1)
        loss_jsd = (0.5 * (kl_pm + kl_qm)).mean().to(dtype=func_logits.dtype)

        z1 = F.normalize(outputs["func_emb"].float(), dim=-1)
        z2 = F.normalize(outputs["anon_func_emb"].float(), dim=-1)
        loss_cos = (1.0 - torch.sum(z1 * z2, dim=-1)).mean().to(dtype=func_logits.dtype)

    total_loss = (
        loss_func
        + line_weight * loss_line
        + siamese_weight * loss_jsd
        + cosine_weight * loss_cos
    )

    stats = {
        "loss_total": float(total_loss.detach().item()),
        "loss_func": float(loss_func.detach().item()),
        "loss_line": float(loss_line.detach().item()),
        "loss_jsd": float(loss_jsd.detach().item()),
    }
    return total_loss, stats
