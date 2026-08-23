#!/usr/bin/env python3
"""
Read-only filesystem MCP server (JP's "solution B", 2026-07-30).

Gives an AI a SAFE, pre-approved way to browse/read JP's projects — no dialog needed
because the scope IS the safety boundary: read-only tools (list_dir/read_file/grep/tree),
hard-whitelisted roots, size caps. No Bash, no Write/Edit — nothing here can change
anything on disk, so there's nothing to "approve" per call.

For anything that needs to actually RUN commands or EDIT files, JP uses a real
interactive session (Claude Code CLI, this very kind of chat) where approval prompts
already work correctly — this server intentionally does NOT try to replicate that.

Transport: stdio (simplest for a single local consumer — Claude subprocess/SDK spawns
this as a child process; add --http if a network transport is ever needed).
"""
import os
import pathlib
import subprocess

from mcp.server.fastmcp import FastMCP

# --- whitelist: only these roots are ever readable. Everything else -> refused. ---
ROOTS = [
    pathlib.Path.home() / "syncthing",
    pathlib.Path.home() / "Projects",
]
MAX_FILE_BYTES = 200_000       # refuse reading anything bigger (binary/huge logs etc.)
MAX_LIST_ENTRIES = 500
MAX_GREP_MATCHES = 200

mcp = FastMCP("jp-filesystem-readonly")


def _resolve_safe(path_str: str) -> pathlib.Path:
    """Resolve a path and verify it's inside a whitelisted root. Raises ValueError if not."""
    p = pathlib.Path(path_str).expanduser().resolve()
    for root in ROOTS:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    allowed = ", ".join(str(r) for r in ROOTS)
    raise ValueError(f"'{path_str}' is outside the allowed roots ({allowed}). Refused.")


@mcp.tool()
def list_dir(path: str) -> str:
    """List files and directories at PATH (must be inside an allowed root).
    Returns names with a trailing '/' for directories."""
    p = _resolve_safe(path)
    if not p.is_dir():
        return f"Not a directory: {p}"
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries[:MAX_LIST_ENTRIES]]
    more = len(entries) - MAX_LIST_ENTRIES
    if more > 0:
        lines.append(f"... and {more} more")
    return "\n".join(lines) if lines else "(empty directory)"


@mcp.tool()
def read_file(path: str, max_bytes: int = MAX_FILE_BYTES) -> str:
    """Read a text file's contents (must be inside an allowed root). Truncated at
    max_bytes (default 200000) to avoid dumping huge/binary files."""
    p = _resolve_safe(path)
    if not p.is_file():
        return f"Not a file: {p}"
    size = p.stat().st_size
    cap = min(max_bytes, MAX_FILE_BYTES)
    with open(p, "rb") as f:
        data = f.read(cap)
    text = data.decode("utf-8", errors="replace")
    if size > cap:
        text += f"\n\n... [truncated, file is {size} bytes, showed first {cap}]"
    return text


@mcp.tool()
def tree(path: str, max_depth: int = 3) -> str:
    """Show a directory tree rooted at PATH (must be inside an allowed root),
    up to max_depth levels deep. Skips hidden dirs and common noise (node_modules,
    __pycache__, .git, venvs)."""
    root = _resolve_safe(path)
    if not root.is_dir():
        return f"Not a directory: {root}"
    skip = {"node_modules", "__pycache__", ".git", ".venv", "venv", ".mypy_cache"}
    lines = []

    def walk(d: pathlib.Path, depth: int, prefix: str):
        if depth > max_depth or len(lines) > MAX_LIST_ENTRIES:
            return
        try:
            entries = sorted(
                [e for e in d.iterdir() if not e.name.startswith(".") and e.name not in skip],
                key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return
        for e in entries:
            lines.append(f"{prefix}{e.name}{'/' if e.is_dir() else ''}")
            if e.is_dir():
                walk(e, depth + 1, prefix + "  ")

    lines.append(f"{root.name}/")
    walk(root, 1, "  ")
    if len(lines) > MAX_LIST_ENTRIES:
        lines = lines[:MAX_LIST_ENTRIES] + ["... (truncated)"]
    return "\n".join(lines)


@mcp.tool()
def grep(pattern: str, path: str, glob: str = "*") -> str:
    """Search for a text PATTERN (plain substring, case-insensitive) inside files
    matching GLOB under PATH (must be inside an allowed root). Returns up to 200
    matching 'file:line: text' rows. Uses ripgrep-style plain matching, not regex,
    to keep it predictable."""
    root = _resolve_safe(path)
    if not root.exists():
        return f"Path does not exist: {root}"
    matches = []
    needle = pattern.lower()
    skip = {"node_modules", "__pycache__", ".git", ".venv", "venv"}
    for fp in root.rglob(glob):
        if not fp.is_file() or any(part in skip for part in fp.parts):
            continue
        try:
            if fp.stat().st_size > MAX_FILE_BYTES:
                continue
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if needle in line.lower():
                        matches.append(f"{fp.relative_to(root)}:{i}: {line.strip()[:200]}")
                        if len(matches) >= MAX_GREP_MATCHES:
                            break
        except OSError:
            continue
        if len(matches) >= MAX_GREP_MATCHES:
            break
    if not matches:
        return "No matches."
    suffix = f"\n... (capped at {MAX_GREP_MATCHES})" if len(matches) >= MAX_GREP_MATCHES else ""
    return "\n".join(matches) + suffix


if __name__ == "__main__":
    mcp.run()  # stdio transport
