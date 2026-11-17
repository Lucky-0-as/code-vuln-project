"""
AST & Data-Flow Graph Engine (tree-sitter-cpp / tree-sitter-c):
1. Counterfactual Identifier Anonymizer (var_1, func_1) for Shortcut Removal & Auditing
2. Multi-Relational Token Adjacency Matrix Builder (AST Parent-Child, Variable Def-Use
   Data-Flow Edges, and Control-Flow Guard Edges) for Graphormer-style Attention Bias.
"""

import bisect
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config

# Initialize tree-sitter-cpp (handles both C and C++ syntax), with safe fallbacks
PARSER = None
TREE_SITTER_AVAILABLE = False

try:
    from tree_sitter import Language, Parser
    try:
        import tree_sitter_cpp as tscpp
        try:
            _LANG = Language(tscpp.language())
            PARSER = Parser(_LANG)
        except TypeError:
            _LANG = Language(tscpp.language(), "cpp")
            PARSER = Parser()
            PARSER.set_language(_LANG)
        TREE_SITTER_AVAILABLE = True
    except ImportError:
        import tree_sitter_c as tsc
        try:
            _LANG = Language(tsc.language())
            PARSER = Parser(_LANG)
        except TypeError:
            _LANG = Language(tsc.language(), "c")
            PARSER = Parser()
            PARSER.set_language(_LANG)
        TREE_SITTER_AVAILABLE = True
except Exception:
    TREE_SITTER_AVAILABLE = False


# C/C++ keywords, primitive types, STL containers, and standard libc security APIs
# preserved during identifier anonymization so language & API semantics stay intact.
PRESERVED_IDENTIFIERS = {
    # C/C++ Keywords & Control Flow
    "auto", "break", "case", "char", "class", "const", "constexpr", "continue",
    "default", "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit",
    "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int",
    "long", "mutable", "namespace", "new", "noexcept", "nullptr", "operator",
    "private", "protected", "public", "register", "reinterpret_cast", "return",
    "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
    "switch", "template", "this", "throw", "true", "try", "typedef", "typeid",
    "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "while",
    "_Bool", "_Complex", "_Imaginary", "bool", "NULL", "errno",
    # Fixed-width & Standard Types
    "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "ENOMEM", "EINVAL", "EIO", "EAGAIN", "ENOSYS", "ENOENT", "EPERM",
    "INT_MAX", "INT_MIN", "UINT_MAX", "SIZE_MAX", "ULONG_MAX",
    # C++ STL Core
    "std", "string", "vector", "array", "map", "unordered_map", "set", "unique_ptr",
    "shared_ptr", "weak_ptr", "make_unique", "make_shared", "move", "forward", "span",
    # Standard C Memory, String & I/O APIs
    "malloc", "calloc", "realloc", "free", "alloca",
    "memcpy", "memmove", "memset", "memcmp", "memchr",
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp",
    "strlen", "strnlen", "strchr", "strrchr", "strstr", "strdup", "strndup",
    "sprintf", "snprintf", "vsprintf", "vsnprintf", "printf", "fprintf",
    "scanf", "sscanf", "fscanf", "fopen", "fclose", "fread", "fwrite",
    "fseek", "ftell", "rewind", "fflush", "fgets", "fputs", "getc", "putc",
    "open", "close", "read", "write", "lseek", "mmap", "munmap",
    "exit", "abort", "assert", "abs", "labs", "atoi", "atol", "strtol", "strtoul",
}

CONTROL_FLOW_NODE_TYPES = {
    "if_statement",
    "for_statement",
    "for_range_loop",
    "while_statement",
    "do_statement",
    "switch_statement",
    "goto_statement",
    "return_statement",
}


def _collect_identifiers_ts(node, out_list: list) -> None:
    """Recursively collects identifier spans and tags function declarations/calls."""
    if node.type in ("identifier", "field_identifier"):
        parent_type = node.parent.type if node.parent is not None else ""
        is_func = False
        if parent_type == "function_declarator" and node.parent.child_by_field_name("declarator") == node:
            is_func = True
        elif parent_type == "call_expression" and node.parent.child_by_field_name("function") == node:
            is_func = True
        out_list.append((node.start_byte, node.end_byte, is_func))

    for child in node.children:
        _collect_identifiers_ts(child, out_list)


def anonymize_code_function(code: str) -> tuple[str, int, int]:
    """
    Rewrites user-defined variables to (var_1, var_2, ...) and function/method names
    to (func_1, func_2, ...) while preserving C/C++ keywords and security APIs.
    Returns (anonymized_code, num_unique_vars, num_unique_funcs).
    """
    if TREE_SITTER_AVAILABLE and PARSER is not None:
        code_bytes = code.encode("utf-8", errors="replace")
        tree = PARSER.parse(code_bytes)
        id_spans = []
        _collect_identifiers_ts(tree.root_node, id_spans)

        var_map: dict[str, str] = {}
        func_map: dict[str, str] = {}
        var_counter = 0
        func_counter = 0

        for start_b, end_b, is_func in id_spans:
            name = code_bytes[start_b:end_b].decode("utf-8", errors="replace")
            if name in PRESERVED_IDENTIFIERS:
                continue
            if is_func:
                if name not in func_map and name not in var_map:
                    func_counter += 1
                    func_map[name] = f"func_{func_counter}"
            else:
                if name not in var_map and name not in func_map:
                    var_counter += 1
                    var_map[name] = f"var_{var_counter}"

        mutable_bytes = bytearray(code_bytes)
        for start_b, end_b, _ in sorted(id_spans, key=lambda x: x[0], reverse=True):
            name = code_bytes[start_b:end_b].decode("utf-8", errors="replace")
            if name in PRESERVED_IDENTIFIERS:
                continue
            replacement = func_map.get(name) or var_map.get(name)
            if replacement is not None:
                mutable_bytes[start_b:end_b] = replacement.encode("utf-8")

        return mutable_bytes.decode("utf-8", errors="replace"), var_counter, func_counter

    # Fallback regex anonymizer if tree-sitter binary is unavailable
    var_map = {}
    func_map = {}
    var_counter = 0
    func_counter = 0
    token_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(\s*\()?")

    def _sub(m: re.Match) -> str:
        nonlocal var_counter, func_counter
        name = m.group(1)
        paren = m.group(2) or ""
        if name in PRESERVED_IDENTIFIERS:
            return m.group(0)
        if paren:
            if name not in func_map and name not in var_map:
                func_counter += 1
                func_map[name] = f"func_{func_counter}"
            return f"{func_map.get(name, var_map.get(name, name))}{paren}"
        if name not in var_map and name not in func_map:
            var_counter += 1
            var_map[name] = f"var_{var_counter}"
        return var_map.get(name, func_map.get(name, name))

    return token_re.sub(_sub, code), var_counter, func_counter


def _build_byte_to_char_map(text: str) -> list[int]:
    """Maps UTF-8 byte offsets [0..len(bytes)] to Python str character offsets."""
    b2c = [0]
    for ch in text:
        b_len = len(ch.encode("utf-8"))
        curr_char_idx = b2c[-1] + 1
        b2c.extend([curr_char_idx] * b_len)
    return b2c


def _char_to_token_index(char_pos: int, token_starts: list[int], token_ends: list[int]) -> int:
    """Finds the subword token index covering `char_pos` via binary search, or -1."""
    if not token_starts:
        return -1
    idx = bisect.bisect_right(token_starts, char_pos) - 1
    if 0 <= idx < len(token_starts) and token_starts[idx] <= char_pos < token_ends[idx]:
        return idx
    return -1


def build_chunk_edge_matrix(
    chunk_text: str,
    offset_mapping: list[tuple[int, int]],
    seq_length: int = config.CHUNK_SIZE,
) -> np.ndarray:
    """
    Constructs an (L, L) integer edge-type adjacency matrix over subword tokens
    for Upgrade #3 (Data-Flow & AST Edge-Biased Self-Attention):
      0 = No structural edge
      1 = AST Parent-Child syntactic link
      2 = Variable Definition-Use (Def-Use) data-flow edge
      3 = Control-Flow statement guard-to-body edge
    """
    edge_mat = np.zeros((seq_length, seq_length), dtype=np.int64)
    if not chunk_text.strip():
        return edge_mat

    # Extract valid content token spans (skipping special tokens [CLS], [SEP], [PAD] with (0,0))
    valid_token_indices = []
    token_starts = []
    token_ends = []
    for tok_idx, (c_start, c_end) in enumerate(offset_mapping):
        if tok_idx >= seq_length:
            break
        if c_end > c_start:
            valid_token_indices.append(tok_idx)
            token_starts.append(c_start)
            token_ends.append(c_end)

    if len(valid_token_indices) < 2:
        return edge_mat

    def _tok_at_char(c_pos: int) -> int:
        pos = _char_to_token_index(c_pos, token_starts, token_ends)
        return valid_token_indices[pos] if pos != -1 else -1

    def _add_edge(t1: int, t2: int, etype: int) -> None:
        if 0 <= t1 < seq_length and 0 <= t2 < seq_length and t1 != t2:
            # Higher-priority semantic edges (Def-Use=2, Control-Flow=3) override AST=1
            if edge_mat[t1, t2] < etype:
                edge_mat[t1, t2] = etype
                edge_mat[t2, t1] = etype

    if TREE_SITTER_AVAILABLE and PARSER is not None:
        code_bytes = chunk_text.encode("utf-8", errors="replace")
        b2c = _build_byte_to_char_map(chunk_text)
        max_b = len(b2c) - 1
        tree = PARSER.parse(code_bytes)

        var_occurrences: dict[str, list[int]] = {}

        def _traverse(node) -> None:
            b_start = min(node.start_byte, max_b)
            b_end = min(node.end_byte, max_b)
            c_start = b2c[b_start]
            node_tok = _tok_at_char(c_start)

            # 1. AST Parent-Child syntactic edges (for local expressions/statements < 250 chars)
            if node_tok != -1 and (b_end - b_start) <= 250:
                for child in node.children:
                    if child.is_named:
                        c_b_start = min(child.start_byte, max_b)
                        child_tok = _tok_at_char(b2c[c_b_start])
                        if child_tok != -1:
                            _add_edge(node_tok, child_tok, 1)

            # 2. Control-Flow statement links (condition -> body)
            if node.type in CONTROL_FLOW_NODE_TYPES and node_tok != -1:
                cond_node = node.child_by_field_name("condition")
                body_node = node.child_by_field_name("consequence") or node.child_by_field_name("body")
                if cond_node is not None and body_node is not None:
                    t_cond = _tok_at_char(b2c[min(cond_node.start_byte, max_b)])
                    t_body = _tok_at_char(b2c[min(body_node.start_byte, max_b)])
                    _add_edge(t_cond, t_body, 3)

            # 3. Collect Variable Def-Use occurrences
            if node.type in ("identifier", "field_identifier") and node_tok != -1:
                ident_name = code_bytes[b_start:b_end].decode("utf-8", errors="replace")
                if ident_name not in PRESERVED_IDENTIFIERS:
                    var_occurrences.setdefault(ident_name, []).append(node_tok)

            for child in node.children:
                _traverse(child)

        _traverse(tree.root_node)

        # Link each variable's first definition/declaration to all uses + sequential use chain
        for tok_positions in var_occurrences.values():
            # Deduplicate consecutive identical token indices
            dedup_pos = []
            for tp in tok_positions:
                if not dedup_pos or dedup_pos[-1] != tp:
                    dedup_pos.append(tp)
            if len(dedup_pos) >= 2:
                def_tok = dedup_pos[0]
                for use_tok in dedup_pos[1:]:
                    _add_edge(def_tok, use_tok, 2)
                for k in range(len(dedup_pos) - 1):
                    _add_edge(dedup_pos[k], dedup_pos[k + 1], 2)
    else:
        # Regex fallback Def-Use graph if tree-sitter is unavailable
        var_occurrences = {}
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", chunk_text):
            name = m.group(1)
            if name not in PRESERVED_IDENTIFIERS:
                tok = _tok_at_char(m.start(1))
                if tok != -1:
                    var_occurrences.setdefault(name, []).append(tok)
        for tok_positions in var_occurrences.values():
            if len(tok_positions) >= 2:
                for use_tok in tok_positions[1:]:
                    _add_edge(tok_positions[0], use_tok, 2)

    return edge_mat


def generate_anonymized_split(split_name: str = "test") -> pd.DataFrame:
    """Applies AST identifier anonymization to a processed Parquet split and caches it."""
    in_path = config.PROCESSED_DATA_DIR / f"{split_name}.parquet"
    out_path = config.ANONYMIZED_DATA_DIR / f"{split_name}_anonymized.parquet"

    df = pd.read_parquet(in_path)
    anon_codes, vars_cnt, funcs_cnt = [], [], []
    for code in tqdm(df["func_clean"], desc=f"Anonymizing {split_name} ASTs"):
        a_code, nv, nf = anonymize_code_function(code)
        anon_codes.append(a_code)
        vars_cnt.append(nv)
        funcs_cnt.append(nf)

    anon_df = df.copy()
    anon_df["func_anon"] = anon_codes
    anon_df["vars_renamed"] = vars_cnt
    anon_df["funcs_renamed"] = funcs_cnt
    anon_df.to_parquet(out_path, index=False)
    return anon_df
