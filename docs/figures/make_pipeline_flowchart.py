"""Build docs/figures/citrus_pipeline.pptx, the flowchart of the scatter-corrected MBIR pipeline of CITRUS (one
16:9 slide, editable in PowerPoint), and export it to docs/source/_static/citrus_pipeline.png with PowerPoint.

    python docs/figures/make_pipeline_flowchart.py            # exports the existing .pptx to .png (PowerPoint, Windows)
    python docs/figures/make_pipeline_flowchart.py --rebuild  # regenerates the .pptx from this script first

The slide is edited by hand in PowerPoint; the default run only re-exports it, so hand edits are never overwritten.
``--rebuild`` discards them and regenerates the slide from the layout code below.
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
    L, R = 3.05, 13.0  # the loop container

    # ---- inputs (left column) -----------------------------------------------------------------------------------
    y_data = add_box(slide, 0.35, 1.4, 2.25, 0.95, "Measured projections Y\nprimary + scatter, all views", "data")
    x0 = add_box(slide, 0.35, 3.3, 2.25, 0.95, "Initial reconstruction X₀\n(FDK)", "data")
    add_arrow(slide, cx(y_data), bottom(y_data), cx(x0), top(x0))

    # ---- outer loop container ------------------------------------------------------------------------------------
    loop = add_box(slide, L, 0.45, R - L, 6.1, "", "note", shape=MSO_SHAPE.ROUNDED_RECTANGLE, dashed=True,
                   fill=False)
    loop.adjustments[0] = 0.04
    loop.line.color.rgb = rgb("595959")
    add_label(slide, L + 0.25, 0.5, 9.0, 0.3, "Outer iteration k", color="404040", size=12, bold=True)

    # row A: material densities; schedule note
    dens = add_box(slide, 3.5, 1.0, 5.2, 1.05,
                   "Material densities ρ(Xₖ)\neach voxel a mixture of the two anchor materials\n"
                   "whose attenuations bracket its value", "density")
    add_box(slide, 9.0, 1.0, 3.75, 1.05,
            "Schedule\nscatter step at every n-th outer iteration;\nscatter on every m-th view, the others\n"
            "interpolated in angle", "note", size=9)

    # row B: the LBTE scatter solver, one step per view
    band_left = 3.95
    band = add_box(slide, band_left, 2.35, 12.85 - band_left, 2.3, "", "scatter", shape=MSO_SHAPE.RECTANGLE,
                   fill=True)
    band.fill.fore_color.rgb = rgb("FDF3EC")
    band.line.color.rgb = rgb("C55A11")
    band.line.width = Pt(0.75)
    add_label(slide, band_left + 0.1, 2.38, 8.5, 0.28,
              "LBTE scatter solver, one step per view v (deterministic, multigroup, discrete ordinates)",
              color="843C0C", size=10, bold=True)
    uncol = add_box(slide, 4.1, 2.72, 2.75, 1.2,
                    "Uncollided fluence\nand first-collision source\nφ₁⁽ᵏ⁾ of the densities ρ(Xₖ)",
                    "scatter")
    sweep = add_box(slide, 7.15, 2.72, 2.95, 1.2,
                    "One transport sweep Gₖ\nover φ₁⁽ᵏ⁾ + the aged accumulator\n"
                    "Φᵥ⁽ᵏ⁻¹⁾  →  Φᵥ⁽ᵏ⁾ (all collision orders)",
                    "scatter", size=10)
    proj = add_box(slide, 10.4, 2.72, 2.35, 1.2,
                   "Collapsed-cone projection\nof the scatter emission\nto the panel → sᵥ", "scatter", size=10)
    add_arrow(slide, right(uncol), cy(uncol), left(sweep), cy(sweep))
    add_arrow(slide, right(sweep), cy(sweep), left(proj), cy(proj))

    # THE AGE-ORDERED RECURSION: a thick loop under the sweep box with the update equation
    add_polyline_arrow(slide, [(cx(sweep) + 1.2, bottom(sweep)), (cx(sweep) + 1.2, bottom(sweep) + 0.32),
                               (cx(sweep) - 1.2, bottom(sweep) + 0.32), (cx(sweep) - 1.2, bottom(sweep) + 0.02)],
                       color="C55A11", width=3.0)
    add_label(slide, band_left + 0.1, bottom(sweep) + 0.36, 8.6, 0.3,
              "Age-ordered recursion:   Φᵥ⁽ᵏ⁾ = Gₖ φ₁⁽ᵏ⁾ + Gₖ Φᵥ⁽ᵏ⁻¹⁾"
              "      (Φᵥ kept per view across outer iterations)",
              color="843C0C", size=10.5, bold=True)
    add_label(slide, band_left + 0.1, bottom(sweep) + 0.6, 8.6, 0.2,
              "= Gₖ φ₁⁽ᵏ⁾ + GₖGₖ₋₁ φ₁⁽ᵏ⁻¹⁾ + GₖGₖ₋₁Gₖ₋₂ φ₁⁽ᵏ⁻²⁾ + … :"
              "  2nd order from Xₖ, 3rd from Xₖ₋₁, …  (low orders refreshed first)",
              color="843C0C", size=9, italic=True)
    add_label(slide, cx(sweep) + 1.25, bottom(sweep) + 0.08, 1.6, 0.2, "Φᵥ⁽ᵏ⁾ out",
              color="843C0C", size=9, italic=True)
    add_label(slide, cx(sweep) - 2.05, bottom(sweep) + 0.08, 0.8, 0.2, "Φᵥ⁽ᵏ⁻¹⁾ in",
              color="843C0C", size=9, italic=True, align=PP_ALIGN.RIGHT)

    # row C: primary, ratio, correction, MBIR
    prim = add_box(slide, 3.5, 5.05, 2.3, 1.05,
                   "Primary pᵥ of every view\nprojection of the densities ρ(Xₖ)\nwith the spectrum", "primary",
                   size=10)
    ratio = add_box(slide, 6.05, 5.05, 2.2, 1.05,
                    "Scatter-to-primary ratio\nSPRᵥ = sᵥ / pᵥ", "correct", size=10)
    corr = add_box(slide, 8.5, 5.05, 2.0, 1.05, "Corrected projections\nYᵥ / (1 + SPRᵥ)", "correct", size=10)
    mbir = add_box(slide, 10.75, 5.05, 2.0, 1.05, "n MBIR iterations\nwarm-started at Xₖ\n→ Xₖ₊₁",
                   "mbir", size=10)
    add_arrow(slide, right(prim), cy(prim), left(ratio), cy(ratio))
    add_arrow(slide, right(ratio), cy(ratio), left(corr), cy(corr))
    add_arrow(slide, right(corr), cy(corr), left(mbir), cy(mbir))

    # vertical flow: densities -> uncollided, densities -> primary (left margin), projection -> ratio
    add_arrow(slide, cx(uncol), bottom(dens), cx(uncol), top(uncol))
    add_polyline_arrow(slide, [(3.7, bottom(dens)), (3.7, top(prim))], color="548235", width=1.25)
    add_label(slide, 3.26, 3.4, 0.5, 0.2, "ρ(Xₖ)", color="385723", size=8, italic=True)
    add_polyline_arrow(slide, [(cx(proj), bottom(proj)), (cx(proj), 4.85), (cx(ratio) + 0.3, 4.85),
                               (cx(ratio) + 0.3, top(ratio))])
    add_label(slide, cx(ratio) + 0.38, 4.84, 0.6, 0.2, "sᵥ", color="843C0C", size=9, italic=True)

    # X0 into the loop; X_{k+1} back to the densities
    add_polyline_arrow(slide, [(right(x0), cy(x0)), (2.85, cy(x0)), (2.85, cy(dens) - 0.2),
                               (left(dens), cy(dens) - 0.2)])
    add_polyline_arrow(slide, [(right(mbir), cy(mbir)), (12.9, cy(mbir)), (12.9, 6.32), (3.25, 6.32),
                               (3.25, cy(dens) + 0.2), (left(dens), cy(dens) + 0.2)], color="2F5597", width=1.5)
    add_label(slide, 6.5, 6.12, 3.5, 0.2, "Xₖ₊₁ → next outer iteration", color="1F3864", size=9,
              italic=True, align=PP_ALIGN.CENTER)

    # output
    out = add_box(slide, 10.75, 6.8, 2.0, 0.55, "Scatter-corrected\nreconstruction", "data", size=10)
    add_arrow(slide, cx(mbir), bottom(mbir), cx(out), top(out))

    # caption
    add_label(slide, 0.35, 6.75, 9.5, 0.5,
              "CITRUS: Coupled Iterative Tomographic reconstruction and Recursively-Updated Scatter.  With X held fixed, "
              "N outer iterations reproduce N source\niterations of the transport solver, so the scatter estimate "
              "converges together with the reconstruction for one sweep per view per outer iteration.",
              color="404040", size=9, italic=True)

    prs.save(PPTX)
    return PPTX


def export_png(pptx_path: str, png_path: str, width_px: int = 2400) -> bool:
    """Export slide 1 to PNG with PowerPoint (Windows COM via PowerShell). Returns True on success."""
    if sys.platform != "win32":
        return False
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    deck = Presentation(pptx_path)  # the slide's own aspect ratio (the hand-edited slide may not be 16:9)
    height_px = int(round(width_px * deck.slide_height / deck.slide_width))
    ps = (
        "$app = New-Object -ComObject PowerPoint.Application; "
        f"$p = $app.Presentations.Open('{os.path.abspath(pptx_path)}', $true, $true, $false); "
        f"$p.Slides.Item(1).Export('{os.path.abspath(png_path)}', 'PNG', {width_px}, {height_px}); "
        "$p.Close(); $app.Quit()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr, file=sys.stderr)
    return r.returncode == 0 and os.path.exists(png_path)


if __name__ == "__main__":
    if "--rebuild" in sys.argv[1:] or not os.path.exists(PPTX):
        path = build()
        print("wrote", path)
    else:
        path = PPTX
        print("keeping the hand-edited slide", path, "(--rebuild regenerates it)")
    if export_png(path, PNG):
        print("exported", os.path.abspath(PNG))
    else:
        print("PNG export skipped or failed (needs PowerPoint on Windows)")
