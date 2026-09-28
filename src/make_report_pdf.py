"""Regenerate reports/laporan_tugas2.pdf dari reports/laporan_tugas2.md.

Jalur konversi (tanpa pandoc/weasyprint):
    markdown -> HTML (+CSS cetak A4, gambar di-embed base64)
             -> Chrome headless --print-to-pdf

Caption "Gambar N" / "Tabel N" yang ditulis miring di md dipetakan ke
kelas CSS caption agar tampil sebagai keterangan rapi di PDF.

Pemakaian:
    python src/make_report_pdf.py
"""

from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dukung `python src/make_report_pdf.py`

from src.config import REPORTS_DIR, ROOT

MD_PATH = REPORTS_DIR / "laporan_tugas2.md"
PDF_PATH = REPORTS_DIR / "laporan_tugas2.pdf"
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 11pt; line-height: 1.55; color: #1a1a1a; margin: 0;
}
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 4pt 0; }
h2 { font-size: 14pt; border-bottom: 1.5pt solid #4c72b0;
     padding-bottom: 3pt; margin: 22pt 0 8pt 0;
     page-break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 6pt 0; page-break-after: avoid; }
p { margin: 6pt 0; text-align: justify; }
ul, ol { margin: 6pt 0 6pt 2pt; padding-left: 18pt; }
li { margin: 2.5pt 0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt;
       background: #f4f6f8; padding: 1pt 3pt; border-radius: 2pt; }
pre { background: #f4f6f8; border: 0.5pt solid #d8dee6; border-radius: 3pt;
      padding: 7pt 9pt; font-size: 8.5pt; line-height: 1.4;
      overflow-x: hidden; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0;
        font-size: 9pt; page-break-inside: avoid; }
th { background: #eef3fa; border: 0.5pt solid #b9c6d8; padding: 4pt 6pt;
     text-align: left; }
td { border: 0.5pt solid #cfd8e4; padding: 3.5pt 6pt; }
tr:nth-child(even) td { background: #f8fafc; }
img { max-width: 100%; height: auto; display: block;
      margin: 10pt auto 4pt auto; }
p.caption {
  text-align: center; font-size: 9pt; color: #444;
  margin: 2pt 0 12pt 0; page-break-before: avoid; page-break-inside: avoid;
}
blockquote { border-left: 3pt solid #4c72b0; margin: 8pt 0;
             padding: 2pt 10pt; color: #333; background: #f6f8fb; }
hr { border: none; border-top: 0.75pt solid #cfd8e4; margin: 16pt 0; }
"""

COVER_TITLE = "Prediksi Arah Nilai Tukar USD/IDR dengan Fitur Pasar dan NLP Berita Geopolitik"
COVER_SUBTITLE = "Laporan Tugas 2 - Proposal Pipeline & Eksperimen Baseline"


def embed_images(html: str) -> str:
    """Ganti <img src="..."> relatif dengan data-URI agar PDF mandiri."""

    def _embed(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith("data:"):
            return match.group(0)
        path = (REPORTS_DIR / src).resolve()
        if not path.exists():
            path = (ROOT / src).resolve()
        if not path.exists():
            return match.group(0)
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', _embed, html)


def wrap_captions(html: str) -> str:
    """P -> p.caption untuk paragraf miring 'Gambar N' / 'Tabel N'."""

    def _wrap(match: re.Match) -> str:
        text = match.group(1)
        return f'<p class="caption">{text}</p>'

    return re.sub(
        r"<p><em>((?:Gambar|Tabel)\s+\d+\..*?)</em></p>", _wrap, html, flags=re.S
    )


def build_cover(team_block: str) -> str:
    lines = team_block.strip().splitlines()
    members = "\n".join(f"<li>{ln.strip('- ').strip()}</li>" for ln in lines if ln.strip())
    return (
        f'<div class="cover">'
        f'<h1>{COVER_TITLE}</h1>'
        f'<p class="subtitle">{COVER_SUBTITLE}</p>'
        f'<div class="team"><p><strong>Kelompok Ayam Bakar Pak Yanto</strong></p>'
        f"<ul>{members}</ul></div>"
        f"</div>"
    )


COVER_CSS = """
.cover { text-align: center; margin: 30mm 0 10mm 0; }
.cover h1 { font-size: 20pt; border: none; }
.cover .subtitle { font-size: 13pt; color: #4c72b0; margin-top: 6pt; }
.cover .team { margin-top: 24mm; font-size: 11pt; }
.cover .team ul { list-style: none; padding: 0; }
.cover { page-break-after: always; }
"""


def find_chrome() -> Path | None:
    for cand in CHROME_CANDIDATES:
        if cand.exists():
            return cand
    which = shutil.which("chrome") or shutil.which("msedge")
    return Path(which) if which else None


def md_to_html() -> str:
    text = MD_PATH.read_text(encoding="utf-8")

    # Pisahkan blok tim (untuk cover) dari isi setelah heading judul utama.
    lines = text.splitlines()
    team_lines = [ln for ln in lines[:12] if ln.startswith("- ")]
    body_start = next(
        i for i, ln in enumerate(lines) if ln.startswith("# ")
    ) + 1
    body = "\n".join(lines[body_start:]).strip()

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "sane_lists", "smarty"],
        extension_configs={"smarty": {"smart_quotes": False}},
    )
    html_body = md.convert(body)
    html_body = embed_images(html_body)
    html_body = wrap_captions(html_body)

    return f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<style>{CSS}{COVER_CSS}</style></head>
<body>
{build_cover("\n".join(team_lines))}
{html_body}
</body></html>"""


def html_to_pdf(html_path: Path, pdf_path: Path, chrome: Path) -> None:
    cmd = [
        str(chrome),
        "--headless=new" if "chrome" in chrome.name.lower() else "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        raise RuntimeError(f"Chrome gagal membuat PDF: {result.stderr[-500:]}")


def main() -> None:
    chrome = find_chrome()
    if chrome is None:
        raise SystemExit("Chrome/Edge tidak ditemukan; install Chrome atau sesuaikan CHROME_CANDIDATES.")
    html = md_to_html()
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "laporan_tugas2.html"
        html_path.write_text(html, encoding="utf-8")
        html_to_pdf(html_path, PDF_PATH, chrome)
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"PDF dibuat: {PDF_PATH} ({size_kb:.0f} KB) via {chrome.name}")


if __name__ == "__main__":
    main()
