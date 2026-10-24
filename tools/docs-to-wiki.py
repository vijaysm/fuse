#!/usr/bin/env python3
"""Convert the MyST documentation under docs/ into GitHub wiki pages.

    python3 tools/docs-to-wiki.py OUTDIR

Writes OUTDIR/Home.md, OUTDIR/_Sidebar.md and one page per Markdown file under
docs/user. The developer guide (docs/dev) and docs/history stay in the repository.
Sphinx/MyST constructs that GitHub does not render are rewritten: literalinclude
becomes a code block holding the included source, admonitions become block
quotes, toctree directives become the sidebar, and links between pages become
wiki links. Links to files outside docs/ point at the repository on GitHub.

The wiki is a separate git repository (<repo>.wiki.git). To publish:

    python3 tools/docs-to-wiki.py /tmp/wiki-out
    git clone https://github.com/vijaysm/fuse.wiki.git /tmp/wiki
    rsync -a --delete --exclude .git /tmp/wiki-out/ /tmp/wiki/
    git -C /tmp/wiki add -A && git -C /tmp/wiki commit -m "Sync docs" && git -C /tmp/wiki push

The wiki is generated output: edit docs/, not the wiki.
"""
import json
import os
import re
import sys

REPO = "https://github.com/vijaysm/fuse"
BRANCH = "main"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
SECTIONS = ("user",)
SKIP_PARTS = ("reviews",)

LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
FENCE = re.compile(r"^(`{3,})(\{(\w[\w-]*)\})?\s*(.*)$")
COLON = re.compile(r"^(:{3,})\{(\w[\w-]*)\}\s*(.*)$")
LABEL = re.compile(r"^\([A-Za-z0-9_.-]+\)=\s*$")


def unquote(v):
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return json.loads(v)
    return v


def pages():
    out = {}
    for sec in SECTIONS:
        base = os.path.join(DOCS, sec)
        for dp, dn, fn in os.walk(base):
            rel = os.path.relpath(dp, base)
            if any(p in SKIP_PARTS for p in rel.split(os.sep)):
                continue
            for f in sorted(fn):
                if f.endswith(".md"):
                    out[os.path.join(dp, f)] = None
    names = {}
    for path in sorted(out):
        stem = os.path.splitext(os.path.basename(path))[0]
        sec = os.path.relpath(path, DOCS).split(os.sep)[0]
        if stem == "index":
            name = "User-guide" if sec == "user" else "Developer-guide"
        else:
            stem = stem[0].upper() + stem[1:]
            name = stem if sec == "user" else "Dev-" + stem
        names[path] = name
    names[os.path.join(DOCS, "index.md")] = "Home"
    names[os.path.join(DOCS, "contents.md")] = "Home"
    return names


def title_of(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# "):
                return line[2:].strip()
    return os.path.basename(path)


class Converter:
    def __init__(self, names):
        self.names = names
        self.titles = {p: title_of(p) for p in names}

    def link(self, src, text, target):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            return "[%s](%s)" % (text or target, target)
        path, _, anchor = target.partition("#")
        full = os.path.normpath(os.path.join(os.path.dirname(src), path))
        suffix = "#" + anchor if anchor else ""
        if full in self.names:
            label = text or self.titles[full]
            return "[%s](%s%s)" % (label, self.names[full], suffix)
        rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
        kind = "tree" if os.path.isdir(full) else "blob"
        return "[%s](%s/%s/%s/%s%s)" % (text or rel, REPO, kind, BRANCH, rel, suffix)

    def inline(self, src, line):
        return LINK.sub(lambda m: self.link(src, m.group(1), m.group(2)), line)

    def literalinclude(self, src, arg, opts):
        path = os.path.normpath(os.path.join(os.path.dirname(src), arg))
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        sel = opts.get("lines")
        if sel:
            keep = []
            for part in sel.split(","):
                a, _, b = part.strip().partition("-")
                lo = int(a)
                hi = int(b) if b else lo
                keep.extend(lines[lo - 1:hi])
            lines = keep

        def find(pat, start=0):
            for i in range(start, len(lines)):
                if pat in lines[i]:
                    return i
            raise SystemExit("%s: %r not found in %s" % (src, pat, arg))

        lo, hi = 0, len(lines)
        if "start-at" in opts:
            lo = find(opts["start-at"])
        elif "start-after" in opts:
            lo = find(opts["start-after"]) + 1
        if "end-at" in opts:
            hi = find(opts["end-at"], lo) + 1
        elif "end-before" in opts:
            hi = find(opts["end-before"], lo)
        body = lines[lo:hi]
        return ["```" + opts.get("language", "")] + body + ["```"]

    def convert(self, src, lines):
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if LABEL.match(line):
                i += 1
                continue
            indent = line[:len(line) - len(line.lstrip(" "))]
            m = FENCE.match(line.lstrip(" "))
            if m:
                fence, _, directive, arg = m.groups()
                j = i + 1
                body = []
                while j < len(lines) and lines[j].strip() != fence:
                    body.append(lines[j][len(indent):] if lines[j].startswith(indent) else lines[j])
                    j += 1
                i = j + 1
                if not directive:
                    out.append(line)
                    out.extend(indent + b if b else b for b in body)
                    out.append(indent + fence)
                    continue
                opts = {}
                while body and body[0].startswith(":"):
                    k, _, v = body.pop(0)[1:].partition(":")
                    opts[k.strip()] = unquote(v.strip())
                if directive == "literalinclude":
                    out.extend(indent + b if b else b for b in self.literalinclude(src, arg, opts))
                elif directive == "toctree":
                    pass
                else:
                    out.extend(self.convert(src, body))
                continue
            m = COLON.match(line)
            if m:
                fence, directive, arg = m.groups()
                depth = len(fence)
                j = i + 1
                body = []
                while j < len(lines) and not re.match(r"^:{%d,}\s*$" % depth, lines[j]):
                    body.append(lines[j])
                    j += 1
                i = j + 1
                if directive in ("note", "tip", "warning", "important", "caution"):
                    inner = self.convert(src, body)
                    out.append("> **%s**" % directive.capitalize())
                    out.append(">")
                    out.extend(("> " + t) if t else ">" for t in inner)
                else:
                    out.extend(self.convert(src, body))
                continue
            out.append(self.inline(src, line))
            i += 1
        return out


def toctrees(path):
    """Yield (caption, [target, ...]) for each toctree in a file."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("```{toctree}"):
            caption, entries = "", []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                t = lines[i].strip()
                if t.startswith(":caption:"):
                    caption = t[len(":caption:"):].strip()
                elif t and not t.startswith(":"):
                    entries.append(t)
                i += 1
            yield caption, entries
        i += 1


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    names = pages()
    conv = Converter(names)
    contents = os.path.join(DOCS, "contents.md")

    side = ["**[Home](Home)**", ""]
    toc = []
    for caption, entries in toctrees(contents):
        side.append("**%s**" % caption)
        toc.append("### %s" % caption)
        toc.append("")
        for e in entries:
            target = os.path.join(DOCS, e + ".md")
            if target not in names:
                continue
            item = "- [%s](%s)" % (conv.titles[target], names[target])
            side.append(item)
            toc.append(item)
        side.append("")
        toc.append("")
    with open(os.path.join(outdir, "_Sidebar.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(side).rstrip() + "\n")

    for path, name in names.items():
        if name == "Home":
            continue
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        body = conv.convert(path, lines)
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        body.append("")
        body.append("---")
        body.append("*Generated from [`%s`](%s/blob/%s/%s). Edit the repository, not the wiki.*"
                    % (rel, REPO, BRANCH, rel))
        with open(os.path.join(outdir, name + ".md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(body) + "\n")

    index = os.path.join(DOCS, "index.md")
    with open(index, encoding="utf-8") as fh:
        home = conv.convert(index, fh.read().split("\n"))
    # The wiki is the published site, so its home page carries the full contents.
    paras = "\n".join(home).split("\n\n")
    home = "\n\n".join(q for q in paras if "table of contents" not in q).split("\n")
    home += ["", "## Contents", ""] + toc
    with open(os.path.join(outdir, "Home.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(home).rstrip() + "\n")
    print("wrote %d pages to %s" % (len(names) - 1, outdir))


if __name__ == "__main__":
    main()
