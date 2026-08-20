"""A small, tolerant unified-diff applier.

No third-party patch library is available, and LLM-produced diffs often have
slightly wrong line numbers, so this applies hunks by locating their context
block in the file (with a whitespace-insensitive fallback) rather than trusting
the @@ line offsets. Used by ``apply_patch`` and by the judge when it applies the
agent's final patch to the buggy source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HUNK_RE = re.compile(r"^@@ .* @@")


class PatchError(ValueError):
    """Raised when a diff cannot be parsed or applied cleanly."""


@dataclass
class _Hunk:
    before: list[str]   # context + removed lines (content only)
    after: list[str]    # context + added lines (content only)
    hint: int           # 0-based start line hint from the @@ header


def _parse(diff_text: str) -> list[_Hunk]:
    lines = diff_text.splitlines()
    hunks: list[_Hunk] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("diff "):
            i += 1
            continue
        if _HUNK_RE.match(line):
            m = re.search(r"-(\d+)", line)
            hint = (int(m.group(1)) - 1) if m else 0
            i += 1
            before, after = [], []
            while i < len(lines) and not _HUNK_RE.match(lines[i]) \
                    and not lines[i].startswith("--- "):
                body = lines[i]
                if body.startswith("\\"):  # "\ No newline at end of file"
                    i += 1
                    continue
                tag = body[0] if body else " "
                content = body[1:] if body else ""
                if tag == " ":
                    before.append(content)
                    after.append(content)
                elif tag == "-":
                    before.append(content)
                elif tag == "+":
                    after.append(content)
                else:  # a bare line with no marker — treat as context
                    before.append(body)
                    after.append(body)
                i += 1
            hunks.append(_Hunk(before=before, after=after, hint=max(hint, 0)))
        else:
            i += 1
    if not hunks:
        raise PatchError("no hunks found in diff")
    return hunks


def _find(haystack: list[str], needle: list[str], hint: int) -> int:
    """Return the index where ``needle`` occurs in ``haystack``, or -1."""
    if not needle:
        return min(hint, len(haystack))
    n = len(needle)

    def matches(pos: int, strip: bool) -> bool:
        if pos < 0 or pos + n > len(haystack):
            return False
        for a, b in zip(haystack[pos:pos + n], needle):
            if (a.strip() == b.strip()) if strip else (a == b):
                continue
            return False
        return True

    # Prefer a match near the hint, then exact anywhere, then whitespace-insensitive.
    for strip in (False, True):
        candidates = sorted(range(len(haystack) - n + 1), key=lambda p: abs(p - hint))
        for pos in candidates:
            if matches(pos, strip):
                return pos
    return -1


def apply_unified_diff(original: str, diff_text: str) -> str:
    """Apply ``diff_text`` to ``original`` and return the patched text."""
    trailing_nl = original.endswith("\n")
    work = original.splitlines()
    for hunk in _parse(diff_text):
        pos = _find(work, hunk.before, hunk.hint)
        if pos == -1:
            raise PatchError(
                "could not locate hunk context in file; the patch does not apply"
            )
        work[pos:pos + len(hunk.before)] = hunk.after
    out = "\n".join(work)
    if trailing_nl:
        out += "\n"
    return out
