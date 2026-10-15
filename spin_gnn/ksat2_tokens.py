"""Vocabulary + encode/decode for the v2 "reasoning trace" event encoding
(spin_gnn.ksat2), as opposed to ksat.py/ksat_render.py's vector-per-token
encoding. Every token here is a single categorical id into a fixed
per-instance vocabulary (12 special tokens + one token per variable + one
token per clause) -- a standard language-model-style tokenization, meant to
be fed through nn.Embedding rather than ksat.py's real-valued vectors.

Sequence grammar (see ksat2.py's module docstring for the event semantics):
  BEGIN_HINTS   (VAR (TRUE|FALSE))*
  BEGIN_STATEMENT   (CLAUSE_c (NOT? VAR)*)*                for c in 0..m-1
  BEGIN_REASONING   ((NO_UNIT | IS_UNIT CLAUSE_c) SET VAR (TRUE|FALSE) SATISFIED* | BACKTRACK VAR REACTIVATED*)*
  SAT (VAR (TRUE|FALSE))*   |   UNSAT

SATISFIED/REACTIVATED (each followed by one CLAUSE_c token, zero or more
repeats) are opt-in via `Vocab(..., use_clause_status=True)` -- off by
default so every existing checkpoint's vocabulary (and therefore its
embedding/readout shapes) is reconstructed identically by
`Vocab(n_vars, m_clauses)`, the 2-arg call every prior caller uses. ksat2.py's
solver always computes `event["satisfied"]`/`["reactivated"]`; whether those
lists get turned into tokens is purely a tokenizer-side choice.
"""
import numpy as np

SPECIAL_NAMES = [
    "BEGIN_HINTS", "BEGIN_STATEMENT", "NOT", "BEGIN_REASONING",
    "NO_UNIT", "IS_UNIT", "SET", "TRUE", "FALSE", "SAT", "UNSAT", "BACKTRACK",
]
CLAUSE_STATUS_NAMES = ["SATISFIED", "REACTIVATED"]


class Vocab:
    def __init__(self, n_vars, m_clauses, use_clause_status=False):
        self.n_vars = n_vars
        self.m_clauses = m_clauses
        self.use_clause_status = use_clause_status
        self._special_names = SPECIAL_NAMES + (CLAUSE_STATUS_NAMES if use_clause_status else [])
        self.n_special = len(self._special_names)
        for i, name in enumerate(self._special_names):
            setattr(self, name, i)
        self.var_offset = self.n_special
        self.clause_offset = self.n_special + n_vars
        self.size = self.n_special + n_vars + m_clauses

    def var_token(self, v):
        return self.var_offset + v

    def clause_token(self, c):
        return self.clause_offset + c

    def is_var_token(self, t):
        return self.var_offset <= t < self.var_offset + self.n_vars

    def is_clause_token(self, t):
        return self.clause_offset <= t < self.clause_offset + self.m_clauses

    def var_index(self, t):
        return t - self.var_offset

    def clause_index(self, t):
        return t - self.clause_offset

    def name(self, t):
        if t < self.n_special:
            return self._special_names[t]
        if self.is_var_token(t):
            return f"X{self.var_index(t) + 1}"
        if self.is_clause_token(t):
            return f"CLAUSE{self.clause_index(t) + 1}"
        return f"<oob:{t}>"


def encode_instance(clauses, n_vars, hints, trace, vocab):
    """clauses: (m, n_vars) int8 array (ksat.sample_random_ksat's output).
    hints: dict var -> +-1. trace: ksat2.solve_with_reasoning_trace's
    return dict. Returns an (T,) int64 array of token ids."""
    tokens = [vocab.BEGIN_HINTS]
    for var in sorted(hints):
        tokens.append(vocab.var_token(var))
        tokens.append(vocab.TRUE if hints[var] == 1 else vocab.FALSE)

    tokens.append(vocab.BEGIN_STATEMENT)
    m = clauses.shape[0]
    for c in range(m):
        tokens.append(vocab.clause_token(c))
        for v in range(n_vars):
            sign = clauses[c, v]
            if sign == 0:
                continue
            if sign < 0:
                tokens.append(vocab.NOT)
            tokens.append(vocab.var_token(v))

    tokens.append(vocab.BEGIN_REASONING)
    emit_status = getattr(vocab, "use_clause_status", False)
    for event in trace["events"]:
        if event["kind"] == "unassign":
            tokens.append(vocab.BACKTRACK)
            tokens.append(vocab.var_token(event["var"]))
            if emit_status:
                for c in event.get("reactivated", []):
                    tokens.append(vocab.REACTIVATED)
                    tokens.append(vocab.clause_token(c))
            continue
        if event["kind"] == "unit_prop":
            tokens.append(vocab.IS_UNIT)
            tokens.append(vocab.clause_token(event["clause"]))
        else:  # decision or flip
            tokens.append(vocab.NO_UNIT)
        tokens.append(vocab.SET)
        tokens.append(vocab.var_token(event["var"]))
        tokens.append(vocab.TRUE if event["value"] == 1 else vocab.FALSE)
        if emit_status:
            for c in event.get("satisfied", []):
                tokens.append(vocab.SATISFIED)
                tokens.append(vocab.clause_token(c))

    if trace["satisfiable"]:
        tokens.append(vocab.SAT)
        assignment = trace["assignment"]
        for v in range(n_vars):
            if assignment[v] != 0:
                tokens.append(vocab.var_token(v))
                tokens.append(vocab.TRUE if assignment[v] == 1 else vocab.FALSE)
    else:
        tokens.append(vocab.UNSAT)

    return np.array(tokens, dtype=np.int64)


class ParseError(Exception):
    """Raised by the strict helpers below (_expect*) on well-formed training
    data, where any mismatch is a real bug. decode() itself never raises --
    it catches this and reports where parsing stopped, since it's also used
    on freely model-generated sequences that may be malformed by
    construction, not by bug."""


def decode(tokens, vocab):
    """Parback into a structured dict, tolerant of malformed/truncated
    input (e.g. a freely model-generated sequence) -- stops and reports
    where it broke rather than raising, mirroring ksat_render.py's
    "informative, not a crash" philosophy.

    Returns: {"hints": {var: +-1}, "clauses": {c: {var: +-1}}, "events":
    [...], "outcome": "SAT"|"UNSAT"|None, "assignment": {var: +-1},
    "well_formed": bool, "error": str or None}.
    """
    tokens = [int(t) for t in tokens]
    i = 0
    n = len(tokens)
    result = {
        "hints": {}, "clauses": {}, "events": [], "outcome": None,
        "assignment": {}, "well_formed": False, "error": None,
    }

    def peek():
        return tokens[i] if i < n else None

    def expect(tok, what):
        nonlocal i
        if i >= n or tokens[i] != tok:
            raise ParseError(f"expected {what} at position {i}, got {tokens[i] if i < n else '<eof>'}")
        i += 1

    def read_value():
        nonlocal i
        if i >= n or tokens[i] not in (vocab.TRUE, vocab.FALSE):
            raise ParseError(f"expected TRUE/FALSE at position {i}")
        value = 1 if tokens[i] == vocab.TRUE else -1
        i += 1
        return value

    def read_var():
        nonlocal i
        if i >= n or not vocab.is_var_token(tokens[i]):
            raise ParseError(f"expected a variable token at position {i}")
        var = vocab.var_index(tokens[i])
        i += 1
        return var

    try:
        expect(vocab.BEGIN_HINTS, "BEGIN_HINTS")
        while peek() is not None and vocab.is_var_token(peek()):
            var = read_var()
            result["hints"][var] = read_value()

        expect(vocab.BEGIN_STATEMENT, "BEGIN_STATEMENT")
        while peek() is not None and vocab.is_clause_token(peek()):
            c = vocab.clause_index(tokens[i])
            i += 1
            literals = {}
            while peek() is not None and (peek() == vocab.NOT or vocab.is_var_token(peek())):
                sign = 1
                if peek() == vocab.NOT:
                    sign = -1
                    i += 1
                literals[read_var()] = sign
            result["clauses"][c] = literals

        emit_status = getattr(vocab, "use_clause_status", False)

        def read_status_tags(tag_tok):
            nonlocal i
            clauses = []
            while emit_status and peek() == tag_tok:
                i += 1  # consume the tag
                if i >= n or not vocab.is_clause_token(tokens[i]):
                    raise ParseError(f"expected a clause token after status tag at position {i}")
                clauses.append(vocab.clause_index(tokens[i]))
                i += 1
            return clauses

        expect(vocab.BEGIN_REASONING, "BEGIN_REASONING")
        while peek() in (vocab.NO_UNIT, vocab.IS_UNIT, vocab.BACKTRACK):
            if peek() == vocab.BACKTRACK:
                i += 1
                var = read_var()
                reactivated = read_status_tags(vocab.REACTIVATED) if emit_status else []
                result["events"].append({
                    "kind": "unassign", "var": var, "value": None, "clause": None,
                    "satisfied": [], "reactivated": reactivated,
                })
                continue
            clause_idx = None
            if peek() == vocab.IS_UNIT:
                i += 1
                if i >= n or not vocab.is_clause_token(tokens[i]):
                    raise ParseError(f"expected a clause token after IS_UNIT at position {i}")
                clause_idx = vocab.clause_index(tokens[i])
                i += 1
            else:
                i += 1  # NO_UNIT
            expect(vocab.SET, "SET")
            var = read_var()
            value = read_value()
            satisfied = read_status_tags(vocab.SATISFIED) if emit_status else []
            kind = "unit_prop" if clause_idx is not None else "decision_or_flip"
            result["events"].append({
                "kind": kind, "var": var, "value": value, "clause": clause_idx,
                "satisfied": satisfied, "reactivated": [],
            })

        if peek() == vocab.SAT:
            i += 1
            result["outcome"] = "SAT"
            while peek() is not None and vocab.is_var_token(peek()):
                var = read_var()
                result["assignment"][var] = read_value()
        elif peek() == vocab.UNSAT:
            i += 1
            result["outcome"] = "UNSAT"
        else:
            raise ParseError(f"expected SAT/UNSAT at position {i}, got {tokens[i] if i < n else '<eof>'}")

        result["well_formed"] = True
    except ParseError as exc:
        result["error"] = str(exc)
    return result


def render(tokens, vocab):
    """One line per event/section, human-readable -- for notebook display
    and quick debugging, analogous to ksat_render.py's trace_to_text."""
    decoded = decode(tokens, vocab)
    lines = []
    hints_str = " ".join(f"X{v + 1}={'T' if val == 1 else 'F'}" for v, val in sorted(decoded["hints"].items()))
    lines.append(f"HINTS: {hints_str or '(none)'}")
    for c, literals in sorted(decoded["clauses"].items()):
        lits = " | ".join(f"X{v + 1}" if s > 0 else f"~X{v + 1}" for v, s in sorted(literals.items()))
        lines.append(f"CLAUSE{c + 1}: ({lits})")
    for event in decoded["events"]:
        val_str = "T" if event["value"] == 1 else "F"
        if event["kind"] == "unassign":
            line = f"UNASSIGN X{event['var'] + 1}"
            if event.get("reactivated"):
                line += "  REACTIVATED: " + ",".join(f"CLAUSE{c + 1}" for c in event["reactivated"])
            lines.append(line)
            continue
        if event["kind"] == "unit_prop":
            line = f"UNIT(CLAUSE{event['clause'] + 1}) SET X{event['var'] + 1}={val_str}"
        else:
            line = f"SET X{event['var'] + 1}={val_str}"
        if event.get("satisfied"):
            line += "  SATISFIED: " + ",".join(f"CLAUSE{c + 1}" for c in event["satisfied"])
        lines.append(line)
    if decoded["outcome"] == "SAT":
        assign_str = " ".join(f"X{v + 1}={'T' if val == 1 else 'F'}" for v, val in sorted(decoded["assignment"].items()))
        lines.append(f"SAT: {assign_str}")
    elif decoded["outcome"] == "UNSAT":
        lines.append("UNSAT")
    if not decoded["well_formed"]:
        lines.append(f"(parse stopped early: {decoded['error']})")
    return "\n".join(lines)


def check_satisfies(assignment, clauses, n_vars):
    """assignment: dict var -> +-1 (missing vars treated as unassigned/0).
    clauses: (m, n_vars) int8 array. True iff every clause has >=1 satisfied
    literal under this (possibly partial) assignment."""
    full = np.zeros(n_vars, dtype=np.int8)
    for v, val in assignment.items():
        full[v] = val
    return bool(np.all((clauses * full).max(axis=1) > 0)) if clauses.shape[0] else True
