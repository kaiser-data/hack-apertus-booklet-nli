"""The Büchli pipeline as a block diagram, in two forms from one definition:

- paper/pipeline.excalidraw : editable at https://excalidraw.com (File > Open), clean style
- paper/fig-pipeline.pdf    : the figure in the paper (matplotlib, vector, serif like the text)

    uv run --with matplotlib python paper/pipeline_diagram.py
"""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
INPUT, CPU, GPU, OUTPUT = "#f1f3f5", "#e7f1fb", "#fdf0e1", "#e8f5ec"
EDGE, INK, MUTED = "#343a40", "#111111", "#495057"

# (x, y, w, h, fill, title, body, tag) in diagram units, y pointing down
BOXES = [
    (0, 10, 260, 150, INPUT, "Booklet PDF", "official, DE · FR · IT\n16–144 pages", ""),
    (0, 230, 260, 150, INPUT, "Claim + vote", "in DE, FR or IT\n(may differ from\nthe booklet)", ""),
    (310, 10, 290, 150, CPU, "1  Split", "≈900-character passages\nwith page and section\n(PyMuPDF)", "CPU"),
    (650, 125, 290, 150, CPU, "2  Retrieve", "16 nearest passages,\ncross-lingual\n(MiniLM)", "CPU"),
    (990, 125, 290, 150, GPU, "3  Verify", "Apertus v1.5 (8B / 70B):\nlabel + cited passages\n(JSON schema)", "GPU"),
    (1330, 105, 290, 190, OUTPUT, "Verdict + evidence", "entailment / neutral /\ncontradiction,\n≤ 5 verbatim sentences\nwith page numbers", ""),
]
# polylines (orthogonal); the last segment carries the arrowhead
ARROWS = [
    [(260, 85), (310, 85)],                            # booklet -> split
    [(600, 85), (625, 85), (625, 170), (650, 170)],    # split -> retrieve
    [(260, 305), (625, 305), (625, 230), (650, 230)],  # claim -> retrieve
    [(940, 200), (990, 200)],                          # retrieve -> verify
    [(1280, 200), (1330, 200)],                        # verify -> verdict
]
NOTE = (650, 292, 630, "≈ 3,400 input tokens per claim (whole booklet: ≈ 31,400)")
W, H = 1620, 385


def write_excalidraw() -> None:
    rng = random.Random(7)
    els = []

    def base(kind, x, y, w, h, **kw):
        return {"id": f"{kind}-{len(els)}", "type": kind, "x": x, "y": y, "width": w, "height": h,
                "angle": 0, "strokeColor": kw.get("stroke", EDGE),
                "backgroundColor": kw.get("fill", "transparent"), "fillStyle": "solid",
                "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
                "groupIds": [], "frameId": None, "roundness": None,
                "seed": rng.randint(1, 2**31), "version": 1, "versionNonce": rng.randint(1, 2**31),
                "isDeleted": False, "boundElements": None, "updated": 1, "link": None, "locked": False}

    def text(x, y, w, s, size, color=INK, align="center"):
        el = base("text", x, y, w, size * 1.25 * len(s.split("\n")), stroke=color)
        el.update({"text": s, "originalText": s, "fontSize": size, "fontFamily": 2,
                   "textAlign": align, "verticalAlign": "top", "containerId": None,
                   "lineHeight": 1.25, "autoResize": True})
        els.append(el)

    for x, y, w, h, fill, title, body, tag in BOXES:
        els.append(base("rectangle", x, y, w, h, fill=fill))
        text(x + 10, y + 14, w - 20, title, 22)
        text(x + 10, y + 48, w - 20, body, 16, MUTED)
        if tag:
            text(x + w - 60, y + 6, 52, tag, 12, MUTED, "right")
    for pts in ARROWS:
        (x0, y0) = pts[0]
        el = base("arrow", x0, y0, pts[-1][0] - x0, pts[-1][1] - y0, stroke=INK)
        el.update({"points": [[px - x0, py - y0] for px, py in pts], "lastCommittedPoint": None,
                   "startBinding": None, "endBinding": None, "startArrowhead": None,
                   "endArrowhead": "arrow"})
        els.append(el)
    x, y, w, s = NOTE
    text(x, y, w, s, 15, MUTED)
    doc = {"type": "excalidraw", "version": 2, "source": "booklet-nli/paper/pipeline_diagram.py",
           "elements": els, "appState": {"viewBackgroundColor": "#ffffff"}, "files": {}}
    (HERE / "pipeline.excalidraw").write_text(json.dumps(doc, indent=1, ensure_ascii=False))


def tex(s: str) -> str:
    """STIX has no plain "≈"/"≤" glyphs; render them as math symbols."""
    return s.replace("≈", r"$\approx$").replace("≤", r"$\leq$")


def draw_pdf() -> None:
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.9, 6.9 * H / W))
    ax.set_xlim(-6, W + 6)
    ax.set_ylim(H + 6, -6)
    ax.axis("off")
    for x, y, w, h, fill, title, body, tag in BOXES:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=EDGE, linewidth=0.8))
        ax.text(x + w / 2, y + 28, title, ha="center", va="center", fontsize=7.6, color=INK,
                fontweight="bold")
        ax.text(x + w / 2, y + 50, tex(body), ha="center", va="top", fontsize=5.9, color=MUTED,
                linespacing=1.3)
        if tag:
            ax.text(x + w - 8, y + 8, tag, ha="right", va="top", fontsize=4.8, color=MUTED,
                    family="DejaVu Sans")
    for pts in ARROWS:
        for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x1, x2], [y1, y2], color=INK, linewidth=0.8, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color=INK, shrinkA=0, shrinkB=0))
    x, y, w, s = NOTE
    ax.text(x + w / 2, y + 8, tex(s), ha="center", va="top", fontsize=5.9, color=MUTED, style="italic")
    fig.savefig(HERE / "fig-pipeline.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    write_excalidraw()
    draw_pdf()
    print("wrote pipeline.excalidraw and fig-pipeline.pdf")
