#!/usr/bin/env python3
"""Generate the baseline-harness API reference from source docstrings + signatures.

Emits:
  - docs/API.md            — one Markdown file (all modules)
  - docs/api/index.html    — HTML index
  - docs/api/<module>.html — one HTML page per module

Run from the repo root:  python tools/gen_api_docs.py
Re-run after code changes to refresh the docs. Importing the client modules does NOT require
provider SDKs (those are imported lazily inside the policy constructors).
"""

from __future__ import annotations

import html
import importlib
import inspect
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    "core.types", "core.prompts", "core.accounting", "core.clientutil",
    "core.bench", "core.history", "core.output", "core.harness", "core.cli",
    "core.provenance",
    "clients.gemini_client", "clients.anthropic_client",
    "clients.openai_client", "clients.sglang_client",
]


def _sig(fn) -> str:
    try:
        # Strip the ` at 0x…` address from repr'd defaults (e.g. `<function _post_json at 0x…>`)
        # so the output is deterministic — otherwise the address changes every run and churns docs.
        return re.sub(r" at 0x[0-9a-fA-F]+", "", str(inspect.signature(fn)))
    except (ValueError, TypeError):
        return "(...)"


def _methods(cls):
    """(name, func, is_static) for methods DEFINED on cls, in source order."""
    out = []
    for name, raw in vars(cls).items():
        if name.startswith("__") and name != "__init__":
            continue
        if isinstance(raw, staticmethod):
            out.append((name, raw.__func__, True))
        elif inspect.isfunction(raw):
            out.append((name, raw, False))
    return out


def collect(modname):
    mod = importlib.import_module(modname)
    funcs = [(n, o) for n, o in inspect.getmembers(mod, inspect.isfunction)
             if getattr(o, "__module__", None) == modname and not n.startswith("_")]
    classes = [(n, o) for n, o in inspect.getmembers(mod, inspect.isclass)
               if getattr(o, "__module__", None) == modname]
    funcs.sort(key=lambda t: t[1].__code__.co_firstlineno)
    classes.sort(key=lambda t: getattr(t[1], "__firstlineno__", 0) or inspect.getsourcelines(t[1])[1])
    return mod, funcs, classes


def first_line(doc):
    return (doc or "").strip().split("\n", 1)[0]


# ---- Markdown ----------------------------------------------------------

def render_md():
    lines = ["# DiG-bench baseline harness — API reference", "",
             "> Auto-generated from source docstrings + signatures by `tools/gen_api_docs.py`. "
             "Re-run that script to refresh.", "", "## Modules", ""]
    data = [(m, collect(m)) for m in MODULES]
    for m, (mod, _, _) in data:
        anchor = m.replace(".", "").replace("_", "")
        lines.append(f"- [`{m}`](#{anchor}) — {first_line(inspect.getdoc(mod))}")
    lines.append("")
    for m, (mod, funcs, classes) in data:
        lines += ["---", "", f"## `{m}`", ""]
        if inspect.getdoc(mod):
            lines += ["> " + inspect.getdoc(mod).replace("\n", "\n> "), ""]
        if funcs:
            lines += ["### Functions", ""]
            for n, o in funcs:
                lines += [f"#### `{n}{_sig(o)}`", "", (inspect.getdoc(o) or "_(no docstring)_"), ""]
        for cn, cls in classes:
            lines += [f"### class `{cn}`", "", (inspect.getdoc(cls) or "_(no docstring)_"), ""]
            for mn, mf, is_static in _methods(cls):
                tag = " *(staticmethod)*" if is_static else ""
                lines += [f"- **`{mn}{_sig(mf)}`**{tag}  ",
                          "  " + (inspect.getdoc(mf) or "_(no docstring)_").replace("\n", "\n  "), ""]
    (ROOT / "docs" / "API.md").write_text("\n".join(lines), encoding="utf-8")


# ---- HTML --------------------------------------------------------------

CSS = """
:root{--fg:#1b1f24;--muted:#5a6675;--bg:#fff;--code:#f4f6f8;--accent:#2456b8;--border:#e3e8ee}
*{box-sizing:border-box}body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);
background:var(--bg);margin:0}.wrap{max-width:920px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:.2em 0}h2{font-size:21px;margin-top:1.6em;border-bottom:1px solid var(--border);padding-bottom:4px}
h3{font-size:17px;margin-top:1.5em}h4{font-size:15px;margin:1.1em 0 .2em}a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}code{background:var(--code);padding:1px 5px;border-radius:4px;font:13px/1.5 ui-monospace,Menlo,monospace}
pre{background:var(--code);padding:12px 14px;border-radius:8px;overflow-x:auto;font:13px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap}
.sig{background:var(--code);border-left:3px solid var(--accent);padding:8px 12px;border-radius:0 6px 6px 0;margin:.4em 0;
font:13px/1.5 ui-monospace,Menlo,monospace;overflow-x:auto}.muted{color:var(--muted)}.static{color:var(--muted);font-size:12px}
.cls{border:1px solid var(--border);border-radius:10px;padding:4px 18px 14px;margin:1.2em 0;background:#fcfdfe}
nav{margin:1em 0 2em}nav a{display:inline-block;margin:0 10px 6px 0}.top{font-size:13px}
"""


def _doc_html(doc):
    return f"<pre>{html.escape(doc.strip())}</pre>" if doc else '<p class="muted">(no docstring)</p>'


def _page(title, body, home="index.html"):
    nav = "" if home is None else f'<p class="top"><a href="{home}">&larr; all modules</a></p>'
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>{nav}{body}</div></body></html>")


def render_html():
    out = ROOT / "docs" / "api"
    out.mkdir(parents=True, exist_ok=True)
    data = [(m, collect(m)) for m in MODULES]

    # index
    idx = ["<h1>DiG-bench baseline harness — API reference</h1>",
           "<p class='muted'>Auto-generated from source docstrings + signatures.</p><nav>"]
    for m, (mod, _, _) in data:
        idx.append(f"<a href='{m.replace('.', '_')}.html'><b>{m}</b></a> "
                   f"<span class='muted'>— {html.escape(first_line(inspect.getdoc(mod)))}</span><br>")
    idx.append("</nav>")
    (out / "index.html").write_text(_page("DiG-bench baseline harness API", "".join(idx), home=None), encoding="utf-8")

    for m, (mod, funcs, classes) in data:
        b = [f"<h1><code>{m}</code></h1>", _doc_html(inspect.getdoc(mod))]
        if funcs:
            b.append("<h2>Functions</h2>")
            for n, o in funcs:
                b += [f"<h4><code>{html.escape(n)}</code></h4>",
                      f"<div class='sig'>{html.escape(n + _sig(o))}</div>", _doc_html(inspect.getdoc(o))]
        for cn, cls in classes:
            b.append(f"<div class='cls'><h2>class <code>{cn}</code></h2>{_doc_html(inspect.getdoc(cls))}")
            for mn, mf, is_static in _methods(cls):
                tag = " <span class='static'>staticmethod</span>" if is_static else ""
                b += [f"<h4><code>{html.escape(mn)}</code>{tag}</h4>",
                      f"<div class='sig'>{html.escape(mn + _sig(mf))}</div>", _doc_html(inspect.getdoc(mf))]
            b.append("</div>")
        (out / f"{m.replace('.', '_')}.html").write_text(_page(m, "".join(b)), encoding="utf-8")


if __name__ == "__main__":
    render_md()
    render_html()
    n_mod = len(MODULES)
    print(f"Generated docs/API.md + docs/api/ ({n_mod} module pages + index.html)")
