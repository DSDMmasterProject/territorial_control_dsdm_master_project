"""
pub_style.py — Reusable publication-quality matplotlib style for the Myanmar paper.

Usage:
    import pub_style
    pub_style.apply()
    # ... make your figure ...
    pub_style.save(fig, FIGURES_DIR / "A2_confusion_matrices_overall")

The style targets a 6.5in-wide figure (full text width in a typical LaTeX article).
Font sizes are absolute so they are correct when \includegraphics[width=\textwidth]{}.
"""

from contextlib import contextmanager
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Palette (same colours as reporting.ipynb) ─────────────────────────────────
MODEL_COLORS = {
    "focal_mask"        : "#2266CC",
    "random_proportion" : "#AAAAAA",
    "no_change"         : "#22BB44",
}

# ── rcParams dict — apply once at the top of any export script ────────────────
PUB_RC = {
    # Backgrounds
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "savefig.facecolor" : "white",
    "savefig.transparent": False,
    # Font
    "font.family"       : "serif",
    "font.serif"        : ["DejaVu Serif", "Times New Roman", "serif"],
    # Sizes — absolute pt, designed for a 6.5in-wide figure
    "figure.titlesize"  : 12,
    "axes.titlesize"    : 11,
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.title_fontsize": 9,
    # Lines and borders
    "axes.linewidth"    : 0.7,
    "xtick.major.width" : 0.6,
    "ytick.major.width" : 0.6,
    "xtick.minor.width" : 0.4,
    "ytick.minor.width" : 0.4,
    "lines.linewidth"   : 1.5,
    "patch.linewidth"   : 0.7,
    # Tick direction
    "xtick.direction"   : "out",
    "ytick.direction"   : "out",
    # Grid
    "axes.grid"         : False,
    # PDF/PS: embed fonts (no Type 3 bitmaps)
    "pdf.fonttype"      : 42,
    "ps.fonttype"       : 42,
    # Figure spacing
    "figure.constrained_layout.use": False,
}


def apply():
    """Apply publication rcParams globally."""
    mpl.rcParams.update(PUB_RC)


def reset():
    """Restore matplotlib defaults."""
    mpl.rcParams.update(mpl.rcParamsDefault)


@contextmanager
def pub_context():
    """Context manager: apply pub style inside `with` block, restore after."""
    original = {k: mpl.rcParams[k] for k in PUB_RC if k in mpl.rcParams}
    apply()
    try:
        yield
    finally:
        mpl.rcParams.update(original)


def save(fig, stem: Path, dpi: int = 300, pad_inches: float = 0.05):
    """
    Save figure as both PDF (vector) and PNG (raster).

    Parameters
    ----------
    fig      : matplotlib Figure
    stem     : Path without extension, e.g. FIGURES_DIR / "A2_confusion_matrices"
    dpi      : raster DPI for PNG (default 300)
    pad_inches: whitespace margin around tight bounding box (default 0.05in)
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    common = dict(bbox_inches="tight", pad_inches=pad_inches, facecolor="white")

    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")

    fig.savefig(pdf_path, format="pdf", **common)
    fig.savefig(png_path, format="png", dpi=dpi, **common)

    print(f"  Saved PDF: {pdf_path}")
    print(f"  Saved PNG: {png_path}  (dpi={dpi})")
    return pdf_path, png_path
