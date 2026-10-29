"""
Render the conference paper into the supplied IEEE Word template.

The template is a valid OOXML package but declares its relationships under the
`purl.oclc.org` namespace, which python-docx does not recognise, so it cannot be
opened with that library.  This script therefore edits the package directly:
the template's own `word/styles.xml`, theme and section geometry are preserved
byte-for-byte and only `word/document.xml` is replaced, with the three figures
added as new media parts.

Layout follows IEEE practice: the title block spans the page, the body -- from
the abstract onwards -- is two columns, and full-width figures sit in their own
single-column continuous sections so they span both columns instead of being
squeezed into one.

Markdown conventions read here: `**[FIGURE n: file.png]**` (add ` | full` for a
figure spanning both columns) followed by a `*Fig. n. ...*` caption line;
`**TABLE n. ...**` followed by a pipe table; `> expression (n)` for a numbered
equation; the first bold and italic lines after the title for the author block.
The document properties are written blank, so the file names no person, as
double-blind review requires.

Usage:  python build_paper_docx.py [template.docx]
        Without an argument the IEEE template is read from the repository
        history, where it has been kept since it left the working tree.
Output: paper/Fisher-Rao-Curvature-IEEE.docx
        (export to PDF from Word; see paper/README.md)
"""

from __future__ import annotations

import io
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_MD = os.path.join("paper", "fisher-rao-conference-paper.md")
TEMPLATE = "IEEE_Template.docx"
TEMPLATE_COMMIT = "a5648ef"   # the commit that added the template to the repository
OUT = os.path.join("paper", "Fisher-Rao-Curvature-IEEE.docx")
FIGDIR = os.path.join("paper", "figures")

EMU_PER_IN = 914400
COL_IN = 3.4          # one IEEE column
FULL_IN = 7.0         # both columns


# ----------------------------------------------------------------- helpers
def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def png_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", head[16:24])


def runs(text: str, font: str | None = None) -> str:
    """Convert inline **bold**, *italic*, `code`, <sub>..</sub> and <sup>..</sup>
    into w:r runs.  `font` overrides the paragraph style's font (the template's
    equation style uses Symbol, which would turn Latin letters into Greek)."""
    out = []
    face = (f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>' if font else "")
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|<sub>.*?</sub>|<sup>.*?</sup>)", text):
        if not part:
            continue
        props = ""
        body = part
        if part.startswith("**") and part.endswith("**"):
            props, body = "<w:b/>", part[2:-2]
        elif part.startswith("*") and part.endswith("*"):
            props, body = "<w:i/>", part[1:-1]
        elif part.startswith("`") and part.endswith("`"):
            props, body = "<w:i/>", part[1:-1]
        elif part.startswith("<sub>"):
            props, body = '<w:vertAlign w:val="subscript"/>', part[5:-6]
        elif part.startswith("<sup>"):
            props, body = '<w:vertAlign w:val="superscript"/>', part[5:-6]
        rpr = f"<w:rPr>{face}{props}</w:rPr>" if (props or face) else ""
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{esc(body)}</w:t></w:r>')
    return "".join(out)


def visible(text: str) -> str:
    """Cell text without inline markup, for sizing table columns."""
    return re.sub(r"\*\*|\*|`|</?su[bp]>", "", text)


def sectpr(cols: int) -> str:
    """A continuous section break with `cols` columns, same page geometry."""
    c = ('<w:cols w:space="36pt"/>' if cols == 1 else
         '<w:cols w:num="2" w:space="24pt" w:equalWidth="1"/>')
    return ("<w:sectPr><w:type w:val=\"continuous\"/>"
            '<w:pgSz w:w="595.30pt" w:h="841.90pt" w:code="9"/>'
            '<w:pgMar w:top="36pt" w:right="44.65pt" w:bottom="44pt" '
            'w:left="44.65pt" w:header="36pt" w:footer="36pt" w:gutter="0pt"/>'
            f"{c}<w:docGrid w:linePitch=\"360\"/></w:sectPr>")


def para(text: str, style: str = "BodyText", sect: str | None = None,
         jc: str | None = None, keep: bool = False) -> str:
    ppr = f'<w:pStyle w:val="{style}"/>'
    if keep:
        ppr += "<w:keepNext/>"
    if jc:
        ppr += f'<w:jc w:val="{jc}"/>'
    if sect:
        ppr += sect
    return f"<w:p><w:pPr>{ppr}</w:pPr>{runs(text)}</w:p>"


def equation(expr: str, number: str) -> str:
    """The template's equation style: the expression on its centre tab, the
    number on its right tab, in Times New Roman rather than the style's Symbol."""
    tab = "<w:r><w:tab/></w:r>"
    tnr = "Times New Roman"
    return (f'<w:p><w:pPr><w:pStyle w:val="equation"/></w:pPr>{tab}{runs(expr, tnr)}'
            f"{tab}{runs(f'({number})', tnr)}</w:p>")


def image_para(rid: str, w_in: float, path: str, name: str) -> str:
    px_w, px_h = png_size(path)
    cx = int(w_in * EMU_PER_IN)
    cy = int(cx * px_h / px_w)
    return (
        '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" '
        'w:after="60"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" '
        'distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{10 + int(re.sub(r"[^0-9]", "", rid))}" '
        f'name="{name}"/>'
        "<a:graphic xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\">"
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/>'
        "</a:stretch></pic:blipFill>"
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>")


def table(rows: list[list[str]], width_in: float) -> str:
    ncol = max(len(r) for r in rows)
    tw = int(width_in * 1440)                      # twips
    # Columns share the width in proportion to their longest body entry, with the
    # header counting half (headers may wrap, numbers should not), bounded so no
    # column starves or dominates, and never narrower than the column's longest
    # unbreakable word; the constant leaves room for Word's cell padding.
    def need_of(j):
        cells = [visible(r[j]) for r in rows if j < len(r)]
        body = max((len(c) for c in cells[1:]), default=0)
        head = len(cells[0]) if cells else 0
        word = max((len(w) for c in cells for w in c.split()), default=0)
        return max(min(body, 30), (head + 1) // 2, word) + 5
    need = [need_of(j) for j in range(ncol)]
    widths = [int(tw * n / sum(need)) for n in need]
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    body = []
    for i, row in enumerate(rows):
        cells = []
        # keep-with-next on every row but the last keeps the table in one piece
        keep = "<w:keepNext/>" if i < len(rows) - 1 else ""
        for j in range(ncol):
            txt = row[j] if j < len(row) else ""
            style = "tablecolhead" if i == 0 else "tablecopy"
            jc = "center" if (i == 0 or j > 0) else "left"
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>'
                '<w:vAlign w:val="center"/></w:tcPr>'
                + f"<w:p><w:pPr><w:pStyle w:val=\"{style}\"/>{keep}"
                  f'<w:jc w:val="{jc}"/><w:spacing w:before="20" w:after="20"/>'
                  f"</w:pPr>{runs(txt)}</w:p></w:tc>")
        trpr = ('<w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' if i == 0
                else "<w:trPr><w:cantSplit/></w:trPr>")
        body.append(f"<w:tr>{trpr}{''.join(cells)}</w:tr>")
    return (
        "<w:tbl><w:tblPr>"
        f'<w:tblW w:w="{tw}" w:type="dxa"/>'
        "<w:tblBorders>"
        + "".join(f'<w:{e} w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                  for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + "</w:tblBorders>"
        '<w:tblLayout w:type="fixed"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(body)}</w:tbl>")


# ------------------------------------------------------------------- parse
def build_body(md: str, figs: dict) -> str:
    """Walk the markdown and emit OOXML in one pass."""
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    cols_state = 1            # the title block is single-column
    pending_table: list[list[str]] = []
    cap_idx: int | None = None

    def flush_table(full: bool):
        """Emit the pending table.

        A table with many narrow columns is unreadable squeezed into one IEEE
        column, so tables past a column threshold are promoted to full width by
        wrapping them -- and their caption -- in their own single-column
        continuous section.  In OOXML the paragraph *carrying* a sectPr is the
        LAST paragraph of that section, hence the break goes in before the
        caption and the closing break after the table.
        """
        nonlocal pending_table, cap_idx
        if not pending_table:
            return
        wide = full or len(pending_table[0]) >= 7
        if wide:
            at = cap_idx if cap_idx is not None else len(out)
            out.insert(at, para("", "BodyText", sect=sectpr(2)))
            out.append(table(pending_table, FULL_IN))
            out.append(para("", "BodyText", sect=sectpr(1)))
        else:
            out.append(table(pending_table, COL_IN))
        pending_table = []
        cap_idx = None

    while i < len(lines):
        ln = lines[i].rstrip()
        i += 1

        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            pending_table.append(cells)
            continue
        flush_table(full=False)

        if not ln or ln in ("---",):
            continue

        # ---- title block -------------------------------------------------
        if ln.startswith("# "):
            out.append(para(ln[2:], "papertitle", jc="center"))
            continue
        if cols_state == 1 and re.fullmatch(r"\*\*[^*]+\*\*", ln):
            out.append(para(ln.strip("*"), "Author", jc="center"))
            continue
        if cols_state == 1 and re.fullmatch(r"\*[^*]+\*", ln):
            out.append(para(ln.strip("*"), "Affiliation", jc="center"))
            continue

        # ---- headings ----------------------------------------------------
        if ln.startswith("## "):
            head = re.sub(r"^[IVX]+\.\s*", "", ln[3:])
            if head == "Abstract":
                # IEEE practice: the abstract already sits in the two-column
                # body, so the title block's single-column section closes here.
                if cols_state == 1:
                    out.append(para("", "BodyText", sect=sectpr(1)))
                    cols_state = 2
                i, txt = swallow(lines, i)
                out.append(para("Abstract— " + txt, "Abstract"))
                continue
            if head.startswith("Index Terms"):
                continue
            if head in ("Acknowledgment", "References"):
                out.append(para(head, "Heading5", keep=True))   # IEEE component heads: unnumbered
                continue
            # fallback: if there was no abstract, the first numbered section
            # opens the two-column body instead
            sect = None
            if cols_state == 1 and re.match(r"^[IVX]+\.", ln[3:]):
                sect = sectpr(1)
                cols_state = 2
            if sect:
                out.append(para("", "BodyText", sect=sect))
            out.append(para(head, "Heading1"))
            continue
        if ln.startswith("### "):
            out.append(para(re.sub(r"^[A-Z]\.\s*", "", ln[4:]), "Heading2"))
            continue

        # ---- index terms -------------------------------------------------
        if ln.startswith("**Index Terms**"):
            out.append(para("Index Terms—" + ln.split("—", 1)[1].strip(), "Keywords"))
            continue

        # ---- figures: their own single-column section --------------------
        m = re.match(FIG_RE, ln)
        if m:
            key, full = m.group(1), bool(m.group(2))
            cap = ""
            if i < len(lines) and lines[i].startswith("*Fig."):
                cap = re.sub(r"^Fig\.\s*\d+\.\s*", "",
                             lines[i].strip().strip("*"))
                i += 1
            if full:                    # its own single-column section, as for wide tables
                out.append(para("", "BodyText", sect=sectpr(2)))
            out.append(image_para(figs[key]["rid"], FULL_IN if full else COL_IN,
                                  figs[key]["path"], key))
            out.append(para(cap, "figurecaption", jc="both" if full else "center"))
            if full:
                out.append(para("", "BodyText", sect=sectpr(1)))
            continue

        # ---- table captions ----------------------------------------------
        if ln.startswith("**TABLE "):
            cap_idx = len(out)
            out.append(para(re.sub(r"^TABLE\s+[IVX]+\.\s*", "",
                                   ln.strip("*")), "tablehead", jc="center", keep=True))
            continue

        # ---- equations ----------------------------------------------------
        if ln.startswith("> "):
            m = re.fullmatch(r"> (.*?)\s+\((\d+)\)", ln)
            out.append(equation(m.group(1), m.group(2)) if m else
                       para(ln[2:], "BodyText", jc="center"))
            continue

        # ---- lists --------------------------------------------------------
        if re.match(r"^[-*] ", ln):
            out.append(para(ln[2:], "bulletlist"))
            continue
        if re.match(r"^\d+\. ", ln):
            out.append(para(ln, "BodyText", jc="both"))
            continue

        # ---- references ---------------------------------------------------
        if re.match(r"^\[\d+\] ", ln):
            out.append(para(re.sub(r"^\[\d+\]\s*", "", ln), "references"))
            continue

        out.append(para(ln, "BodyText", jc="both"))

    flush_table(full=False)
    return "".join(out)


def swallow(lines: list[str], i: int) -> tuple[int, str]:
    """Collect the paragraph following a heading."""
    while i < len(lines) and not lines[i].strip():
        i += 1
    buf = []
    while i < len(lines) and lines[i].strip():
        buf.append(lines[i].strip())
        i += 1
    return i, " ".join(buf)


# ------------------------------------------------------------------- build
FIG_RE = r"^\*\*\[FIGURE \d+: ([^\]|\s]+)(?:\s*\|\s*(full))?\]\*\*"


def properties(title: str) -> tuple[str, str]:
    """Blank core and app properties: no author, editor or company is named."""
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{esc(title)}</dc:title><dc:creator></dc:creator><cp:lastModifiedBy></cp:lastModifiedBy>"
            "<cp:revision>1</cp:revision></cp:coreProperties>")
    app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
           'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
           "<Template>Normal.dotm</Template><Application>Microsoft Office Word</Application>"
           "<Company></Company></Properties>")
    return core, app


def main() -> int:
    template = sys.argv[1] if len(sys.argv) > 1 else TEMPLATE
    md = open(SRC_MD, encoding="utf-8").read()

    figs = {}
    for n, m in enumerate(re.finditer(FIG_RE, md, re.M), start=1):
        fname = m.group(1)
        figs[fname] = dict(rid=f"rIdFig{n}", path=os.path.join(FIGDIR, fname),
                           media=f"media/figure{n}.png")
        if not os.path.exists(figs[fname]["path"]):
            print(f"missing figure: {figs[fname]['path']} -- run make_figures.py")
            return 1

    body = build_body(md, figs)
    title = re.search(r"^# (.+)$", md, re.M).group(1).strip()

    doc = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        f"<w:body>{body}{sectpr(2)}"
        "</w:body></w:document>")

    if os.path.exists(template):
        src = zipfile.ZipFile(template)
    else:                       # not in the working tree: take it from the repository history
        data = subprocess.run(["git", "show", f"{TEMPLATE_COMMIT}:{TEMPLATE}"], capture_output=True,
                              check=True).stdout
        src = zipfile.ZipFile(io.BytesIO(data))
        print(f"template read from the repository history ({TEMPLATE_COMMIT}:{TEMPLATE})")
    rels = src.read("word/_rels/document.xml.rels").decode("utf-8")
    add = "".join(
        f'<Relationship Id="{f["rid"]}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="{f["media"]}"/>' for f in figs.values())
    rels = rels.replace("</Relationships>", add + "</Relationships>")

    ct = src.read("[Content_Types].xml").decode("utf-8")
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Types ", "<Types ", 1).replace(
            "</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    skip = {"word/document.xml", "word/_rels/document.xml.rels", "[Content_Types].xml",
            "docProps/core.xml", "docProps/app.xml"}
    core, app = properties(title)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for item in src.infolist():
            if item.filename in skip:
                continue
            z.writestr(item, src.read(item.filename))
        z.writestr("[Content_Types].xml", ct)
        z.writestr("word/document.xml", doc)
        z.writestr("word/_rels/document.xml.rels", rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        for f in figs.values():
            z.writestr("word/" + f["media"], open(f["path"], "rb").read())
    src.close()

    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
    print(f"  figures embedded: {len(figs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
