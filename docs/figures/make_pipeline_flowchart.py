"""Build docs/figures/citrus_pipeline.pptx, the flowchart of the scatter-corrected MBIR pipeline of CITRUS (one
16:9 slide, editable in PowerPoint), and export it to docs/source/_static/citrus_pipeline.png with PowerPoint.

    python docs/figures/make_pipeline_flowchart.py            # writes the .pptx and, on Windows with PowerPoint, the .png

Edit the slide in PowerPoint and re-export (File > Save As > PNG, or rerun this script) when the pipeline changes.
"""
from __future__ import annotations

import os
import subprocess
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(HERE, "citrus_pipeline.pptx")
PNG = os.path.join(HERE, "..", "source", "_static", "citrus_pipeline.png")

FONT = "Calibri"
# fill, border, text colours per role
STYLE = {
    "data": ("DEEAF6", "2F5597", "1F3864"),      # measured data, initial and final volumes (blue)
    "density": ("E2F0D9", "548235", "385723"),   # density model and grids (green)
    "scatter": ("FBE5D6", "C55A11", "843C0C"),   # LBTE scatter solver (orange)
    "primary": ("E4DFEC", "7030A0", "3F1F5E"),   # primary (purple)
    "correct": ("FFF2CC", "BF9000", "7F6000"),   # ratio and correction (yellow)
    "mbir": ("DEEAF6", "2F5597", "1F3864"),      # MBIR (blue)
    "note": ("F2F2F2", "7F7F7F", "404040"),      # notes (grey)
}
EMU_PER_INCH = 914400


def rgb(hex6: str) -> RGBColor:
    return RGBColor.from_string(hex6)


def add_box(slide, x, y, w, h, text, role, *, size=11, bold_first=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            dashed=False, fill=True):
    fill_c, line_c, text_c = STYLE[role]
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.12
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill_c)
    else:
        s.fill.background()
    s.line.color.rgb = rgb(line_c)
    s.line.width = Pt(1.25)
    if dashed:
        ln = s.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
    s.shadow.inherit = False
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bool(bold_first and i == 0)
        r.font.color.rgb = rgb(text_c)
    return s


def add_label(slide, x, y, w, h, text, color="404040", size=10, bold=False, align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = rgb(color)
    return tb


def _arrow_end(line_elm, kind="triangle"):
    ln = line_elm._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": kind, "w": "med", "len": "med"}))


def add_arrow(slide, x1, y1, x2, y2, color="595959", width=1.5, head=True, dashed=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dashed:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
    if head:
        _arrow_end(c.line)
    return c


def add_polyline_arrow(slide, points, color="595959", width=1.5, dashed=False):
    """Axis-aligned segments through `points` [(x, y), ...] in inches; the last segment carries the arrowhead."""
    for i in range(len(points) - 1):
        (x1, y1), (x2, y2) = points[i], points[i + 1]
        add_arrow(slide, x1, y1, x2, y2, color=color, width=width, head=(i == len(points) - 2), dashed=dashed)


def left(s):
    return s.left / EMU_PER_INCH


def top(s):
    return s.top / EMU_PER_INCH


def right(s):
    return Emu(s.left + s.width).inches


def bottom(s):
    return Emu(s.top + s.height).inches


def cx(s):
    return Emu(s.left + s.width // 2).inches


def cy(s):
    return Emu(s.top + s.height // 2).inches


def build() -> str:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # ---- inputs (left column) -----------------------------------------------------------------------------------
    y_data = add_box(slide, 0.4, 1.6, 2.2, 0.95,
                     "Measured projections Y\nprimary + scatter, all views", "data", size=12)
    x0 = add_box(slide, 0.4, 3.3, 2.2, 0.95, "Initial reconstruction X₀\n(FDK)", "data", size=12)
    add_arrow(slide, cx(y_data), bottom(y_data), cx(x0), top(x0))

    # ---- outer loop container ------------------------------------------------------------------------------------
    L, R = 3.0, 13.0
    loop = add_box(slide, L, 0.4, R - L, 6.35, "", "note", shape=MSO_SHAPE.ROUNDED_RECTANGLE, dashed=True,
                   fill=False)
    loop.adjustments[0] = 0.04
    loop.line.color.rgb = rgb("595959")
    add_label(slide, L + 0.25, 0.45, 9.0, 0.3, "Outer iteration k", color="404040", size=13, bold=True)

    # step 1: densities
    dens = add_box(slide, 3.35, 1.0, 2.45, 1.05,
                   "Material densities\nρ(Xₖ): each voxel a mixture of\ntwo anchor materials", "density", size=11)

    # step 2: the age-ordered recursion (the central element)
    rec = add_box(slide, 6.2, 0.95, 6.5, 2.2, "", "scatter", size=12)
    tf = rec.text_frame
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_top = Inches(0.08)
    lines = [
        ("Age-ordered scatter recursion:  one transport sweep Gₖ per view", 13, True, "843C0C"),
        ("", 4, False, "843C0C"),
        ("Φᵥ⁽ᵏ⁾  =  Gₖ φ₁(Xₖ)  +  Gₖ Φᵥ⁽ᵏ⁻¹⁾", 18, True, "843C0C"),
        ("newest first-collision source        previous accumulator", 9, False, "C55A11"),
        ("", 4, False, "843C0C"),
        ("=  Gₖ φ₁⁽ᵏ⁾  +  GₖGₖ₋₁ φ₁⁽ᵏ⁻¹⁾  +  GₖGₖ₋₁Gₖ₋₂ φ₁⁽ᵏ⁻²⁾  +  …",
         13, False, "843C0C"),
        ("2nd-order scatter from Xₖ,  3rd order from Xₖ₋₁,  4th order from Xₖ₋₂, …", 10, False, "843C0C"),
        ("the dominant low orders are refreshed from the newest reconstruction;", 10, False, "843C0C"),
        ("stale information is pushed into the weak, smooth high orders", 10, False, "843C0C"),
    ]
    for i, (text, size, bold, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = text
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = rgb(color)
    add_arrow(slide, right(dens), cy(dens), left(rec), cy(dens))

    # the accumulator carried across outer iterations: a loop around the recursion box
    add_polyline_arrow(slide, [(right(rec) - 1.2, bottom(rec)), (right(rec) - 1.2, bottom(rec) + 0.3),
                               (left(rec) + 1.2, bottom(rec) + 0.3), (left(rec) + 1.2, bottom(rec) + 0.01)],
                       color="C55A11", width=2.0, dashed=True)
    add_label(slide, left(rec) + 1.3, bottom(rec) + 0.06, 3.9, 0.2,
              "Φᵥ⁽ᵏ⁾ kept per view → enters the next outer iteration as Φᵥ⁽ᵏ⁾",
              color="843C0C", size=9, italic=True, align=PP_ALIGN.LEFT)

    # step 3: scatter and primary -> ratio -> corrected projections
    spr = add_box(slide, 3.35, 4.35, 3.95, 1.1,
                  "Scatter-to-primary ratio of every view\nSPRᵥ = sᵥ / pᵥ\nsᵥ: projected scatter of Φᵥ⁽ᵏ⁾;  pᵥ: primary of Xₖ",
                  "correct", size=11)
    corr = add_box(slide, 7.65, 4.35, 2.4, 1.1,
                   "Corrected projections\nYᵥ / (1 + SPRᵥ)", "correct", size=11)
    mbir = add_box(slide, 10.4, 4.35, 2.3, 1.1,
                   "n MBIR iterations\nwarm-started at Xₖ  →  Xₖ₊₁", "mbir", size=11)
    add_polyline_arrow(slide, [(cx(rec) - 2.0, bottom(rec) + 0.3), (cx(rec) - 2.0, 4.1), (cx(spr) + 0.6, 4.1),
                               (cx(spr) + 0.6, top(spr))])
    add_label(slide, cx(spr) + 0.68, 4.1, 0.5, 0.2, "sᵥ", color="843C0C", size=10, italic=True)
    add_polyline_arrow(slide, [(cx(dens) - 0.6, bottom(dens)), (cx(dens) - 0.6, top(spr))], color="7030A0", width=1.5)
    add_label(slide, cx(dens) - 0.55, 3.6, 0.9, 0.2, "pᵥ(Xₖ)", color="7030A0", size=10, italic=True)
    add_arrow(slide, right(spr), cy(spr), left(corr), cy(corr))
    add_arrow(slide, right(corr), cy(corr), left(mbir), cy(mbir))

    # X0 into the loop; X_{k+1} back to the densities
    add_polyline_arrow(slide, [(right(x0), cy(x0)), (2.8, cy(x0)), (2.8, cy(dens)), (left(dens), cy(dens))])
    add_polyline_arrow(slide, [(cx(mbir), bottom(mbir)), (cx(mbir), 6.1), (3.2, 6.1), (3.2, cy(dens) + 0.3),
                               (left(dens), cy(dens) + 0.3)], color="2F5597", width=1.5)
    add_label(slide, 5.5, 5.88, 4.5, 0.2, "Xₖ₊₁ → next outer iteration (k ← k + 1)",
              color="1F3864", size=10, italic=True, align=PP_ALIGN.CENTER)
    add_label(slide, 10.55, 5.55, 2.3, 0.3, "n ≥ 1: several MBIR iterations\nper scatter update", color="1F3864",
              size=9, italic=True, align=PP_ALIGN.CENTER)

    # output
    out = add_box(slide, 10.4, 6.9, 2.3, 0.5, "Scatter-corrected reconstruction", "data", size=11)
    add_polyline_arrow(slide, [(right(mbir), cy(mbir)), (12.85, cy(mbir)), (12.85, cy(out)), (right(out), cy(out))],
                       color="2F5597", width=1.5)

    # caption
    add_label(slide, 0.4, 6.9, 9.6, 0.5,
              "With X held fixed, N outer iterations reproduce N source iterations of the transport solver: the scatter "
              "estimate converges together with the reconstruction,\nfor the cost of one sweep per view per outer "
              "iteration instead of a converged scatter solve for every new volume.",
              color="404040", size=9, italic=True)

    prs.save(PPTX)
    return PPTX


def export_png(pptx_path: str, png_path: str, width_px: int = 2400) -> bool:
    """Export slide 1 to PNG with PowerPoint (Windows COM via PowerShell). Returns True on success."""
    if sys.platform != "win32":
        return False
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    ps = (
        "$app = New-Object -ComObject PowerPoint.Application; "
        f"$p = $app.Presentations.Open('{os.path.abspath(pptx_path)}', $true, $true, $false); "
        f"$p.Slides.Item(1).Export('{os.path.abspath(png_path)}', 'PNG', {width_px}, {int(width_px * 7.5 / 13.333)}); "
        "$p.Close(); $app.Quit()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr, file=sys.stderr)
    return r.returncode == 0 and os.path.exists(png_path)


if __name__ == "__main__":
    path = build()
    print("wrote", path)
    if export_png(path, PNG):
        print("exported", os.path.abspath(PNG))
    else:
        print("PNG export skipped or failed (needs PowerPoint on Windows)")
