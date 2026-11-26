"""Prompt loader and version hashing for committed prompts/*.md files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def get_prompts_dir() -> Path:
    """Resolve the repository's prompts directory."""
    candidates = [
        Path.cwd() / "prompts",
        Path(__file__).resolve().parent.parent / "prompts",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[1]


def load_prompt(name: str) -> str:
    """Load a prompt template from prompts/<name> (or <name>.md)."""
    prompts_dir = get_prompts_dir()
    file_name = name if name.endswith(".md") or name.endswith(".txt") else f"{name}.md"
    path = prompts_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **variables: Any) -> str:
    """Load a prompt template and replace {{var}} placeholders safely."""
    text = load_prompt(name)
    for key, val in variables.items():
        text = text.replace(f"{{{{{key}}}}}", str(val))
    return text


def get_prompt_versions(names: list[str] | None = None) -> dict[str, str]:
    """Compute SHA-256 short hashes of committed prompt files for reproducibility."""
    prompts_dir = get_prompts_dir()
    versions: dict[str, str] = {}
    if not prompts_dir.exists():
        return versions
    files = sorted(prompts_dir.glob("*.md"))
    if names is not None:
        wanted = {n if n.endswith(".md") else f"{n}.md" for n in names}
        files = [f for f in files if f.name in wanted]
    for f in files:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        versions[f.name] = digest
    return versions
