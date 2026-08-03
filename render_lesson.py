"""Render a lesson to one self-contained HTML file you can open in any browser.

Images are embedded as base64, so the output is a single file that works offline
with no plugin, no server, and no IDE support required.

    python render_lesson.py 01-attention/lesson1
    python render_lesson.py --all

Then open the printed path in a browser.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import pathlib
import re
import sys

try:
    import markdown
except ImportError:
    sys.exit("pip install markdown")

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 2.5rem 1.5rem 6rem; max-width: 46rem;
  font: 16px/1.65 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #1f2328; background: #fff;
  -webkit-text-size-adjust: 100%;
}
h1, h2, h3 { line-height: 1.25; margin: 2.2rem 0 .8rem; }
h1 { font-size: 1.85rem; padding-bottom: .3rem; border-bottom: 2px solid #d8dee4; }
h2 { font-size: 1.35rem; margin-top: 2.6rem; }
h3 { font-size: 1.08rem; }
p, li { margin: .7rem 0; }
a { color: #0969da; }
hr { border: 0; border-top: 1px solid #d8dee4; margin: 2.6rem 0; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .875em; background: #f0f1f3; padding: .15em .35em; border-radius: 4px;
}
pre {
  background: #f6f8fa; border: 1px solid #e4e7eb; border-radius: 8px;
  padding: .9rem 1rem; overflow-x: auto; line-height: 1.5;
}
pre code { background: none; padding: 0; font-size: .82rem; }
blockquote {
  margin: 1.1rem 0; padding: .1rem 1rem; border-left: 4px solid #d0d7de; color: #57606a;
}
table { border-collapse: collapse; margin: 1.1rem 0; width: 100%; font-size: .93rem;
        display: block; overflow-x: auto; }
th, td { border: 1px solid #d8dee4; padding: .45rem .7rem; text-align: left;
         vertical-align: top; }
th { background: #f6f8fa; }
tr:nth-child(2n) td { background: #fafbfc; }
img { max-width: 100%; height: auto; display: block; margin: 1.4rem auto;
      border: 1px solid #e4e7eb; border-radius: 8px; background: #fff; }
.part { border-top: 3px double #d0d7de; margin-top: 4rem; padding-top: .5rem; }
.part:first-of-type { border-top: 0; margin-top: 0; }
.toc { background: #f6f8fa; border: 1px solid #e4e7eb; border-radius: 8px;
       padding: .5rem 1.2rem; margin-bottom: 2.5rem; }
@media (prefers-color-scheme: dark) {
  body { color: #e6edf3; background: #0d1117; }
  h1 { border-color: #30363d; }
  hr, .part { border-color: #30363d; }
  a { color: #4493f8; }
  code { background: #262c36; }
  pre { background: #161b22; border-color: #30363d; }
  blockquote { border-color: #3d444d; color: #9198a1; }
  th, td { border-color: #30363d; }
  th { background: #161b22; }
  tr:nth-child(2n) td { background: #11151a; }
  img { background: #fff; border-color: #30363d; }
  .toc { background: #161b22; border-color: #30363d; }
}
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body>{body}</body></html>
"""


def embed_images(html: str, base: pathlib.Path) -> tuple[str, int, list[str]]:
    """Replace <img src="relative.png"> with inline base64 data URIs."""
    done, missing = 0, []

    def repl(m: re.Match) -> str:
        nonlocal done
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.is_file():
            missing.append(src)
            return m.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        done += 1
        return m.group(0).replace(src, f"data:{mime};base64,{data}")

    return re.sub(r'<img[^>]*src="([^"]+)"', repl, html), done, missing


def render(lesson_dir: pathlib.Path) -> pathlib.Path:
    parts = sorted(lesson_dir.glob("[0-9]*.md"),
                   key=lambda p: int(re.match(r"(\d+)", p.name).group(1)))
    if not parts:
        sys.exit(f"no numbered parts found in {lesson_dir}")

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    chunks, toc = [], []
    for p in parts:
        md.reset()
        text = p.read_text()
        heading = text.splitlines()[0].lstrip("# ").strip()
        anchor = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
        toc.append(f'<li><a href="#{anchor}">{heading}</a></li>')
        # local .md links are meaningless in a single page — point them at anchors
        body = re.sub(r'\[([^\]]+)\]\(\d[^)]*\.md\)', r"<b>\1</b>", text)
        chunks.append(f'<section class="part" id="{anchor}">{md.convert(body)}</section>')

    title = f"{lesson_dir.parent.name} · {lesson_dir.name}"
    body = (f"<h1>{title}</h1>"
            f'<nav class="toc"><ul>{"".join(toc)}</ul></nav>'
            + "".join(chunks))
    html, n_img, missing = embed_images(body, lesson_dir)

    out = lesson_dir / f"{lesson_dir.name}.html"
    out.write_text(PAGE.format(title=title, css=CSS, body=html))
    size = out.stat().st_size / 1e6
    print(f"{out}  ({len(parts)} parts, {n_img} images embedded, {size:.1f} MB)")
    for m in missing:
        print(f"  !! missing image: {m}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lesson", nargs="?", help="path to a lesson directory")
    ap.add_argument("--all", action="store_true", help="render every lesson in the repo")
    args = ap.parse_args()

    root = pathlib.Path(__file__).parent
    if args.all:
        targets = sorted(d for d in root.glob("[0-9][0-9]-*/lesson*") if d.is_dir())
    elif args.lesson:
        targets = [pathlib.Path(args.lesson).resolve()]
    else:
        ap.error("give a lesson directory, or --all")

    for t in targets:
        render(t)


if __name__ == "__main__":
    main()
