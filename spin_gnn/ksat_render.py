"""Human-readable rendering of spin_gnn.ksat clauses/traces/tokens for
notebook display -- turns the {-1,0,1}-valued vectors used everywhere else
in this module back into CNF-style text ("(x1 | ~x5 | x9)") and readable
assignment strings ("x1=T x2=? x3=F ..."), so a person can read a generated
or ground-truth trace instead of staring at raw arrays.
"""


def clause_to_str(clause_row, var_names=None):
    """clause_row: (n_vars,) array in {-1,0,1}. Returns e.g. "(x1 | ~x5 | x9)"."""
    n = clause_row.shape[0]
    names = var_names or [f"x{i + 1}" for i in range(n)]
    literals = [names[v] if clause_row[v] > 0 else f"~{names[v]}" for v in range(n) if clause_row[v] != 0]
    return "(" + " | ".join(literals) + ")" if literals else "(empty clause)"


def clauses_to_text(clauses, var_names=None):
    """clauses: (m, n_vars) array. One clause per line, joined by ' &'."""
    return " &\n".join(clause_to_str(clauses[c], var_names) for c in range(clauses.shape[0]))


def assignment_to_str(assignment_row, var_names=None):
    """assignment_row: (n_vars,) array in {-1,0,1}. Returns "x1=T x2=? x3=F"."""
    n = assignment_row.shape[0]
    names = var_names or [f"x{i + 1}" for i in range(n)]
    symbol = {1: "T", -1: "F", 0: "?"}
    return " ".join(f"{names[v]}={symbol[int(assignment_row[v])]}" for v in range(n))


def token_to_str(token, var_names=None):
    """token: (n_vars+1,) array. Dispatches on the last dim (-1=assignment,
    +1=clause), per spin_gnn.ksat's encoding."""
    n = token.shape[0] - 1
    body, kind = token[:n], token[n]
    if kind < 0:
        return "assignment: " + assignment_to_str(body, var_names)
    return "clause:      " + clause_to_str(body, var_names)


def trace_to_text(tokens, step_lengths=None, max_steps=None, var_names=None):
    """tokens: (T, n_vars+1) array. step_lengths (from solve_with_trace)
    groups tokens into per-step blocks with a step header; if omitted (e.g.
    for a freely-generated sequence with no natural step boundaries), every
    token gets its own line with no step header."""
    lines = []
    if step_lengths is None:
        for row in tokens:
            lines.append(token_to_str(row, var_names))
        return "\n".join(lines)

    offset = 0
    for step_idx, length in enumerate(step_lengths):
        if max_steps is not None and step_idx >= max_steps:
            lines.append(f"... ({len(step_lengths) - max_steps} more steps)")
            break
        lines.append(f"step {step_idx}:")
        for row in tokens[offset : offset + length]:
            lines.append("  " + token_to_str(row, var_names))
        offset += length
    return "\n".join(lines)
