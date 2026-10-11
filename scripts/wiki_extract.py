#!/usr/bin/env python3
"""Stream a MediaWiki pages-articles XML dump (optionally .bz2) to plain text.

Output format (what lang/dataset_stream.c expects): one *document* per article,
documents separated by exactly one blank line, paragraphs inside an article
separated by single newlines (so a blank line never splits an article).

Standard library only. This is deliberately a simple markup stripper, not a
full wikitext parser: templates, tables, refs, files/categories and html are
dropped; links keep their label.

    python3 scripts/wiki_extract.py DUMP.xml.bz2 OUT.txt [--max-mb N]
    curl ... | python3 scripts/wiki_extract.py - OUT.txt --max-mb 100
"""
import argparse
import bz2
import re
import sys
import xml.etree.ElementTree as ET

RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
RE_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_FILE = re.compile(r"\[\[(?:File|Image|Category|Media):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
RE_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
RE_EXT = re.compile(r"\[(?:https?:)?//\S+(?:\s+([^\]]*))?\]")
RE_QUOTES = re.compile(r"'{2,5}")
RE_HEAD = re.compile(r"^=+\s*(.*?)\s*=+\s*$")
RE_WS = re.compile(r"[ \t\u00a0]+")


def clean(text):
    text = RE_COMMENT.sub("", text)
    text = RE_REF.sub("", text)
    text = RE_TABLE.sub("", text)
    for _ in range(8):  # nested templates, innermost first
        new = RE_TEMPLATE.sub("", text)
        if new == text:
            break
        text = new
    text = RE_FILE.sub("", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_EXT.sub(lambda m: m.group(1) or "", text)
    text = RE_TAG.sub("", text)
    text = RE_QUOTES.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith(("{|", "|", "!", "__")):
            continue
        m = RE_HEAD.match(line)
        if m:
            continue  # drop section headings; keep prose only
        line = line.lstrip("*#:; ").strip()
        line = RE_WS.sub(" ", line)
        if len(line) >= 60:  # prose paragraphs, not stubs/lists
            lines.append(line)
    return lines


def local(tag):
    return tag.rsplit("}", 1)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="XML or XML.bz2 path, or - for bz2 on stdin")
    ap.add_argument("out")
    ap.add_argument("--max-mb", type=float, default=0, help="stop after N MB of text (0 = all)")
    ap.add_argument("--min-chars", type=int, default=400, help="drop articles shorter than this")
    args = ap.parse_args()

    if args.dump == "-":
        src = bz2.BZ2File(sys.stdin.buffer)
    elif args.dump.endswith(".bz2"):
        src = bz2.open(args.dump, "rb")
    else:
        src = open(args.dump, "rb")
    limit = int(args.max_mb * 1024 * 1024)
    written = docs = 0
    title = ns = None
    redirect = False
    with open(args.out, "w", encoding="utf-8") as out:
        for ev, el in ET.iterparse(src, events=("end",)):
            tag = local(el.tag)
            if tag == "title":
                title = el.text or ""
            elif tag == "ns":
                ns = el.text
            elif tag == "redirect":
                redirect = True
            elif tag == "text" and ns == "0" and not redirect:
                lines = clean(el.text or "")
                body = "\n".join(lines)
                if len(body) >= args.min_chars:
                    doc = title.strip() + "\n" + body
                    out.write(doc + "\n\n")
                    written += len(doc.encode("utf-8")) + 2
                    docs += 1
            elif tag == "page":
                title = ns = None
                redirect = False
                el.clear()
                if limit and written >= limit:
                    break
            if tag in ("text", "revision"):
                el.clear()
    sys.stderr.write("wrote %d articles, %.1f MB -> %s\n" % (docs, written / 1048576.0, args.out))


if __name__ == "__main__":
    main()
