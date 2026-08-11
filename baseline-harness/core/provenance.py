"""Run provenance — everything a published trace needs to be reproducible.

Captured ONCE per run (core/cli.py) and recorded in the session JSONL row
(core/output.py), so each trace is self-describing: the exact code version (git
commit + dirty flag), interpreter and SDK versions, the full resolved CLI config
(minus secrets), the endpoint actually hit, and a hash of the prompt contract —
a silently edited prompt cannot masquerade as the published one.

Stdlib only, best-effort throughout: every field degrades to None (or is omitted)
rather than ever failing a run.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from . import prompts

# Provider SDK distributions whose installed versions are recorded (absent ones omitted).
_SDK_DISTS = ("google-genai", "anthropic", "openai")

# CLI args that must NEVER be recorded in a run artifact.
_SECRET_ARGS = ("api_key",)


def _redact_url(url):
    """Strip userinfo (``user:pass@``) from a URL so credentials embedded in e.g.
    ``--base-url`` never reach a published artifact. Non-URL / non-str values pass through."""
    if not isinstance(url, str) or "@" not in url:
        return url
    try:
        parts = urlsplit(url)
        if parts.username is None and parts.password is None:
            return url
        netloc = parts.hostname or ""
        if parts.port:
            netloc += f":{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except ValueError:
        return "(redacted)"


def _git() -> dict | None:
    """{"commit", "dirty"} for the repo this file lives in, or None (no git / not a checkout)."""
    root = Path(__file__).resolve().parents[1]

    def run(*argv):
        return subprocess.run(["git", "-C", str(root), *argv],
                              capture_output=True, text=True, timeout=5)

    try:
        head = run("rev-parse", "HEAD")
        if head.returncode != 0:
            return None
        status = run("status", "--porcelain")
        return {"commit": head.stdout.strip(),
                "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except Exception:
        return None


def prompt_contract_sha256() -> str:
    """sha256 over the ENTIRE ``core/prompts.py`` source. Hashing selected constants would
    miss model-facing literals inside functions (``seed_text``'s framing text,
    ``creative_mode_instruction``, serialization choices), which could then change without
    changing the fingerprint — the whole module source is the prompt contract. Two runs with
    the same hash saw byte-identical task framing (the per-game description/state is the
    bench server's, recorded separately)."""
    return hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest()


def collect(args, policy, server: str) -> dict:
    """The provenance dict for the session JSONL row. `args` is the RESOLVED namespace
    (after model auto-ID), `policy` the constructed provider policy."""
    sdks = {}
    for dist in _SDK_DISTS:
        try:
            sdks[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    out = {
        "git": _git(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sdk_versions": sdks,
        # Full resolved config: reproduces the exact invocation. Secrets excluded by NAME —
        # never dump credential-bearing args into an artifact — and URL userinfo redacted.
        "config": {k: (_redact_url(v) if k == "base_url" else v)
                   for k, v in vars(args).items() if k not in _SECRET_ARGS},
        "endpoint": {
            "provider": args.provider,
            "server": _redact_url(server),
            "base_url": _redact_url(getattr(policy, "base_url", None)),
            "aws_region": getattr(policy, "resolved_region", None),
            "api_timeout_seconds": args.api_timeout_seconds,
            "stream_idle_timeout": args.stream_idle_timeout,
        },
        # Budget-based thinking families (Anthropic manual, Gemini 2.5): the RESOLVED token
        # budget the level mapped to (None on level/effort-knob providers).
        "thinking_budget": getattr(policy, "thinking_budget", None),
        "prompt_contract_sha256": prompt_contract_sha256(),
    }
    # Open-model serving stack (SGLang): the server's own launch configuration — model
    # revision, tokenizer/template, dtype, parsers — all of which can change results.
    server_info = getattr(policy, "server_info", None)
    models_entry = getattr(policy, "models_entry", None)
    if server_info is not None or models_entry is not None:
        out["sglang_server"] = {"server_info": server_info, "models_entry": models_entry}
    return out
