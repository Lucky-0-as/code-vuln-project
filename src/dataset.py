"""
Hierarchical Multi-Chunk, AST/Def-Use Graph, Line-Mask, and Siamese Dataset
for PrimeVul C/C++ Vulnerability Detection.
"""

import bisect
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src import config
from src.ast_graph import anonymize_code_function, build_chunk_edge_matrix


def _compute_line_char_starts(code: str) -> list[int]:
    """Returns the starting character index of each line in `code`."""
    starts = [0]
    for idx, ch in enumerate(code):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def encode_hierarchical_function(
    code: str,
    line_labels: list[int],
    tokenizer: PreTrainedTokenizerBase,
    max_chunks: int = config.MAX_CHUNKS,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_stride: int = config.CHUNK_STRIDE,
    hierarchical: bool = True,
) -> dict[str, np.ndarray]:
    """
    Tokenizes a C/C++ function into up to `max_chunks` overlapping windows of length
    `chunk_size` (512), aligns per-line vulnerability labels to subword tokens, and
    constructs per-chunk (L, L) AST + Variable Def-Use + Control-Flow edge matrices.

    If `hierarchical=False`, uses single-window Head+Tail truncation (max_chunks=1)
    for baseline ablation comparisons.
    """
    if not hierarchical:
        max_chunks = 1

    max_content = chunk_size - 2  # Reserve [CLS] and [SEP]
    line_starts = _compute_line_char_starts(code)
    n_lines = max(1, len(line_labels))

    # Full tokenization without truncation to obtain exact character offsets
    enc = tokenizer(
        code,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    full_ids: list[int] = enc["input_ids"]
    full_offsets: list[tuple[int, int]] = enc["offset_mapping"]
    total_tokens = len(full_ids)

    # Map each subword token to its line-level vulnerability label
    full_tok_labels: list[int] = []
    for c_start, c_end in full_offsets:
        if c_end <= c_start:
            full_tok_labels.append(-100)
        else:
            ln_idx = bisect.bisect_right(line_starts, c_start) - 1
            ln_idx = max(0, min(ln_idx, n_lines - 1))
            full_tok_labels.append(int(line_labels[ln_idx]))

    # Determine chunk slices
    chunk_slices: list[tuple[int, int]] = []
    if not hierarchical:
        # Single-window Head+Tail truncation (254 head + 256 tail)
        if total_tokens <= max_content:
            chunk_slices.append((0, total_tokens))
        else:
            # Build concatenated head+tail indices
            head_ids = full_ids[: config.HEAD_TOKENS] + full_ids[-config.TAIL_TOKENS :]
            head_offs = full_offsets[: config.HEAD_TOKENS] + full_offsets[-config.TAIL_TOKENS :]
            head_lbls = full_tok_labels[: config.HEAD_TOKENS] + full_tok_labels[-config.TAIL_TOKENS :]
            full_ids, full_offsets, full_tok_labels = head_ids, head_offs, head_lbls
            total_tokens = len(full_ids)
            chunk_slices.append((0, total_tokens))
    else:
        if total_tokens == 0:
            chunk_slices.append((0, 0))
        else:
            start = 0
            while start < total_tokens and len(chunk_slices) < max_chunks:
                end = min(start + max_content, total_tokens)
                # On the final allowed chunk, if tokens remain past `end`, shift window to cover the tail
                if len(chunk_slices) == max_chunks - 1 and end < total_tokens:
                    start = max(0, total_tokens - max_content)
                    end = total_tokens
                chunk_slices.append((start, end))
                if end >= total_tokens:
                    break
                start += chunk_stride

    # Allocate output tensors for K chunks
    input_ids_arr = np.full((max_chunks, chunk_size), tokenizer.pad_token_id, dtype=np.int64)
    attn_mask_arr = np.zeros((max_chunks, chunk_size), dtype=np.int64)
    chunk_mask_arr = np.zeros((max_chunks,), dtype=np.int64)
    line_labels_arr = np.full((max_chunks, chunk_size), -100, dtype=np.int64)
    edge_types_arr = np.zeros((max_chunks, chunk_size, chunk_size), dtype=np.int64)

    for k, (tok_start, tok_end) in enumerate(chunk_slices):
        chunk_mask_arr[k] = 1
        sub_ids = full_ids[tok_start:tok_end]
        sub_offs = full_offsets[tok_start:tok_end]
        sub_lbls = full_tok_labels[tok_start:tok_end]

        seq_ids = [tokenizer.cls_token_id] + sub_ids + [tokenizer.sep_token_id]
        seq_len = len(seq_ids)

        input_ids_arr[k, :seq_len] = seq_ids
        attn_mask_arr[k, :seq_len] = 1
        if len(sub_lbls) > 0:
            line_labels_arr[k, 1 : 1 + len(sub_lbls)] = sub_lbls

        # Build AST + Variable Def-Use + Control-Flow edge matrix for this chunk
        valid_spans = [off for off in sub_offs if off[1] > off[0]]
        if valid_spans:
            char_min = valid_spans[0][0]
            char_max = valid_spans[-1][1]
            chunk_text = code[char_min:char_max]
            rel_offsets = [(0, 0)] + [
                (max(0, s - char_min), max(0, e - char_min)) if e > s else (0, 0)
                for (s, e) in sub_offs
            ] + [(0, 0)]
            edge_types_arr[k] = build_chunk_edge_matrix(
                chunk_text=chunk_text,
                offset_mapping=rel_offsets,
                seq_length=chunk_size,
            )

    return {
        "input_ids": input_ids_arr,
        "attention_mask": attn_mask_arr,
        "chunk_mask": chunk_mask_arr,
        "line_labels": line_labels_arr,
        "edge_types": edge_types_arr,
    }


class PrimeVulGraphSiameseDataset(Dataset):
    """
    On-the-fly PyTorch Dataset supporting:
      - Upgrade #1: Multi-chunk hierarchical encoding (K=4 overlapping windows)
      - Upgrade #2: Siamese AST-anonymized twin generation (var_1, func_1)
      - Upgrade #3: AST + Variable Def-Use + Control-Flow relational edge matrices
      - Upgrade #5: Diff-derived token/line-level vulnerability labels
    """

    def __init__(
        self,
        codes: list[str],
        targets: list[int],
        line_masks_json: list[str],
        tokenizer: PreTrainedTokenizerBase,
        hierarchical: bool = True,
        include_siamese: bool = True,
        precomputed_anon_codes: list[str] | None = None,
    ) -> None:
        self.codes = list(codes)
        self.targets = [int(t) for t in targets]
        self.line_masks = [json.loads(m) if isinstance(m, str) else list(m) for m in line_masks_json]
        self.tokenizer = tokenizer
        self.hierarchical = hierarchical
        self.include_siamese = include_siamese
        self.precomputed_anon_codes = list(precomputed_anon_codes) if precomputed_anon_codes is not None else None

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        code = self.codes[idx]
        target = self.targets[idx]
        l_mask = self.line_masks[idx]
        max_k = config.MAX_CHUNKS if self.hierarchical else 1

        orig_enc = encode_hierarchical_function(
            code=code,
            line_labels=l_mask,
            tokenizer=self.tokenizer,
            max_chunks=max_k,
            hierarchical=self.hierarchical,
        )

        item = {
            "input_ids": torch.tensor(orig_enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(orig_enc["attention_mask"], dtype=torch.long),
            "chunk_mask": torch.tensor(orig_enc["chunk_mask"], dtype=torch.long),
            "line_labels": torch.tensor(orig_enc["line_labels"], dtype=torch.long),
            "edge_types": torch.tensor(orig_enc["edge_types"], dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }

        if self.include_siamese:
            if self.precomputed_anon_codes is not None:
                anon_code = self.precomputed_anon_codes[idx]
            else:
                anon_code, _, _ = anonymize_code_function(code)

            dummy_line_labels = [-100] * max(1, len(anon_code.splitlines()))
            anon_enc = encode_hierarchical_function(
                code=anon_code,
                line_labels=dummy_line_labels,
                tokenizer=self.tokenizer,
                max_chunks=max_k,
                hierarchical=self.hierarchical,
            )
            item["anon_input_ids"] = torch.tensor(anon_enc["input_ids"], dtype=torch.long)
            item["anon_attention_mask"] = torch.tensor(anon_enc["attention_mask"], dtype=torch.long)
            item["anon_chunk_mask"] = torch.tensor(anon_enc["chunk_mask"], dtype=torch.long)
            item["anon_edge_types"] = torch.tensor(anon_enc["edge_types"], dtype=torch.long)

        return item
