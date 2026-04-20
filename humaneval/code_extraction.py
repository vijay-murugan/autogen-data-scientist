"""Extract a 4-space-indented function body from raw LLM output.

Runners instruct the LLM to return only the function body, but in practice
LLMs sometimes:
  * wrap the answer in ```python ... ``` fences,
  * redefine the whole function (``def entry_point(...):``) with imports,
  * use a multi-line signature (``def foo(\\n    x: int,\\n) -> int:``),
  * collapse the whole function onto one line,
  * forget to indent the body, or use 2-space or tab indent,
  * prepend a short commentary line before the code.

This module normalises all those into a body that can be concatenated
directly onto a task's ``PROMPT`` to form a syntactically valid program.

Strategy:
  1. Strip markdown fences.
  2. Drop any preamble text that appears before the target ``def``.
  3. If the result parses as Python and contains a ``FunctionDef`` with
     the target name, extract its body via the AST (handles multi-line
     signatures, single-line bodies, and leading docstrings correctly).
  4. Otherwise treat the input as a raw body.
  5. In every case, force-normalise indentation to exactly 4 spaces so
     it lines up with the docstring closing in PROMPT.
"""

from __future__ import annotations

import ast
import re
import textwrap


# -- Fences and preamble ----------------------------------------------------

def _strip_fences(text: str) -> str:
    """Extract content from a ```python ... ``` block if present; else drop stray fences."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    text = re.sub(r"^```(?:python|py)?\s*\n?", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text


def _drop_preamble(text: str, entry_point: str) -> str:
    """If text starts with non-code commentary, drop everything up to the target ``def``."""
    sig_re = re.compile(rf"^[ \t]*def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
    m = sig_re.search(text)
    if m and m.start() > 0:
        return text[m.start():]
    return text


# -- Indentation normalisation ---------------------------------------------

def _align_first_line(lines: list[str]) -> list[str]:
    """Fix the common LLM mistake where the first body line is flush-left but
    the rest are indented (e.g., ``"x = 1\\n    y = 2"``).

    If the first non-blank line has strictly less leading whitespace than the
    minimum of the remaining non-blank lines, pad the first line so all lines
    share a common prefix. This lets ``textwrap.dedent`` dedent them as a
    single block.
    """
    non_blank = [(i, l) for i, l in enumerate(lines) if l.strip()]
    if len(non_blank) < 2:
        return lines

    first_idx, first_line = non_blank[0]
    first_indent = len(first_line) - len(first_line.lstrip(" "))
    rest_indents = [len(l) - len(l.lstrip(" ")) for _, l in non_blank[1:]]
    min_rest = min(rest_indents)

    if first_indent < min_rest:
        lines = list(lines)
        lines[first_idx] = " " * (min_rest - first_indent) + lines[first_idx]
    return lines


def _normalize_indent(text: str) -> str:
    """Force every non-blank line to start with exactly 4 spaces of base indent.

    ``textwrap.dedent`` removes any common leading whitespace across non-blank
    lines, so relative indentation inside nested blocks is preserved. We then
    prepend 4 spaces to every non-blank line. Tabs are expanded to 4 spaces
    first to avoid mixed-indent errors. If the LLM under-indented just the
    first line (a common failure mode), we align it to the rest first.
    """
    expanded = text.expandtabs(4)
    lines = expanded.split("\n")
    lines = _align_first_line(lines)
    dedented = textwrap.dedent("\n".join(lines))
    lines = dedented.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return "    pass"
    indented = [("    " + l) if l.strip() else "" for l in lines]
    return "\n".join(indented)


# -- AST-based extraction --------------------------------------------------

def _find_function_node(tree: ast.AST, entry_point: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return node
    return None


def _body_via_ast(text: str, entry_point: str) -> str | None:
    """Extract the target function's body using the AST, if ``text`` parses."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    fn = _find_function_node(tree, entry_point)
    if fn is None or not fn.body:
        return None

    body_nodes = fn.body
    first = body_nodes[0]
    # Skip a leading docstring expression
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        body_nodes = body_nodes[1:]

    if not body_nodes:
        return "    pass\n"

    source_lines = text.split("\n")
    start_lineno = body_nodes[0].lineno  # 1-indexed
    end_lineno = getattr(body_nodes[-1], "end_lineno", None) or start_lineno

    # Handle single-line bodies (e.g., ``def foo(): return 1``) where the
    # body lives on the signature line itself: extract everything after the
    # signature ``:`` on that line.
    if start_lineno == fn.lineno:
        sig_line = source_lines[fn.lineno - 1]
        # Balance parens to find the ':' that ends the signature.
        depth = 0
        colon_pos = -1
        for idx, ch in enumerate(sig_line):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ":" and depth == 0:
                colon_pos = idx
                break
        if colon_pos != -1:
            inline_body = sig_line[colon_pos + 1:].strip()
            if inline_body:
                normalized = _normalize_indent(inline_body)
                if not normalized.endswith("\n"):
                    normalized += "\n"
                return normalized
        # Fall through to line-range extraction otherwise.

    body_lines = source_lines[start_lineno - 1: end_lineno]
    raw_body = "\n".join(body_lines)
    normalized = _normalize_indent(raw_body)
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


# -- Public entry point ----------------------------------------------------

def _trim_blank_edges(text: str) -> str:
    """Drop leading/trailing blank lines, but preserve indentation of the first
    non-blank line.

    ``str.strip()`` is too aggressive here: it would eat the leading 4 spaces
    off the first line of a body-only response like ``"    if x:\\n        ..."``,
    which then makes ``textwrap.dedent`` find no common prefix and double-indent
    the rest of the body.
    """
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_function_body(raw: str, entry_point: str) -> str:
    """Return a 4-space-indented body suitable for appending to the task PROMPT."""
    text = _strip_fences(_trim_blank_edges(raw))
    text = _trim_blank_edges(text)
    text = _drop_preamble(text, entry_point)

    body = _body_via_ast(text, entry_point)
    if body is not None:
        return body

    # Raw-body fallback: always normalise indent.
    if not text.strip():
        return "    pass\n"
    normalized = _normalize_indent(text)
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized
