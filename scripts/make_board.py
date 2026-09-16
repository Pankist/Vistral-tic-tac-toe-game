"""Generate boards.pdf — an A4 page with a bold printable '#' grid, for when
hand-drawing on the spot isn't convenient. The detector is markerless and
borderless, so the printed page is just a tidy version of what you'd draw.
Run: make print-board
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "boards.pdf"


def main() -> None:
    w, h = A4
    c = canvas.Canvas(str(OUT), pagesize=A4)
    grid = 420                       # grid span in points (~14.8cm)
    cx, cy = w / 2, h / 2
    third = grid / 3
    c.setLineWidth(6)
    c.setLineCap(1)
    left, bottom = cx - grid / 2, cy - grid / 2
    for i in (1, 2):
        x = left + i * third
        c.line(x, bottom, x, bottom + grid)
        y = bottom + i * third
        c.line(left, y, left + grid, y)
    c.setFont("Helvetica", 9)
    c.drawCentredString(cx, bottom - 24,
                        "visual-gamer-agent — draw thick marks that fill the cell")
    c.showPage()
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
