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


def infer_step_lengths(tokens):
    """Group a raw (T, n_vars+1) token stream into steps using the encoding's
    own grammar rather than an externally-known grouping: a step starts at
    whatever token comes next (regardless of its kind) and absorbs every
    immediately-following clause-type token (kind > 0, last dim). For a
    well-formed trace (solve_with_trace's own output: always one
    assignment token then zero or more clause tokens) this reproduces
    exactly the same lengths solve_with_trace itself returns. For a freely
    model-generated sequence -- which has no externally-known step
    boundaries at all -- this is the only way to segment it into steps, and
    if the model doesn't emit assignment/clause tokens in the expected
    alternation, it just reports whatever grouping actually falls out of
    what it emitted (informative in itself, not an error)."""
    n_tokens = tokens.shape[0]
    lengths = []
    i = 0
    while i < n_tokens:
        start = i
        i += 1
        while i < n_tokens and tokens[i, -1] > 0:
            i += 1
        lengths.append(i - start)
    return lengths


def trace_to_lines(tokens, step_lengths=None, var_names=None):
    """tokens: (T, n_vars+1) array. step_lengths: from solve_with_trace, or
    omit to infer step boundaries from the tokens themselves via
    infer_step_lengths -- the only option for a freely model-generated
    sequence, which has no externally-known grouping at all.

    Unlike trace_to_text's indented multi-line-per-step blocks, this is
    exactly one line per step: the current variable assignment, followed by
    every currently non-trivial clause, all on one line -- built for
    tabular/side-by-side display (see side_by_side_trace) rather than
    reading top-to-bottom."""
    if step_lengths is None:
        step_lengths = infer_step_lengths(tokens)
    n = tokens.shape[1] - 1
    lines = []
    offset = 0
    for length in step_lengths:
        step_tokens = tokens[offset : offset + length]
        offset += length
        assignment = assignment_to_str(step_tokens[0][:n], var_names)
        clauses = " ".join(clause_to_str(row[:n], var_names) for row in step_tokens[1:])
        lines.append(assignment if not clauses else f"{assignment}  {clauses}")
    return lines


def side_by_side_trace(left_lines, right_lines, left_title="MODEL", right_title="TRUE", col_width=60):
    """Two lists of per-step lines (e.g. two trace_to_lines() outputs),
    formatted as two aligned columns for direct visual comparison -- e.g. a
    model-generated trace against the actual solver trace for the same
    instance, step by step. Lines longer than col_width are truncated (with
    a trailing "..."); the shorter list is padded with blanks so both
    columns run the full length of the longer one, which also makes it
    visually obvious where the two traces diverge or one ends early."""

    def fit(line):
        return line if len(line) <= col_width else line[: col_width - 3] + "..."

    n_steps = max(len(left_lines), len(right_lines))
    header = f"{'step':>4}  {left_title:<{col_width}}  {right_title}"
    rule = "-" * len(header)
    rows = [header, rule]
    for i in range(n_steps):
        left = fit(left_lines[i]) if i < len(left_lines) else ""
        right = fit(right_lines[i]) if i < len(right_lines) else ""
        rows.append(f"{i:>4}  {left:<{col_width}}  {right}")
    return "\n".join(rows)
