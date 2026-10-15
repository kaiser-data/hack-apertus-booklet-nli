"""The Büchli pipeline as a block diagram, in two forms from one definition:

- paper/pipeline.excalidraw : editable at https://excalidraw.com (File > Open)
- paper/fig-pipeline.pdf    : hand-drawn style for the paper (matplotlib sketch mode, offline)

    uv run --with matplotlib python paper/pipeline_diagram.py
"""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
YELLOW, BLUE, ORANGE, GREEN = "#ffec99", "#a5d8ff", "#ffd8a8", "#b2f2bb"
INK, MUTED = "#1e1e1e", "#495057"

# (x, y, w, h, fill, title, body) in Excalidraw pixels, y pointing down
BOXES = [
    (0, 0, 250, 140, YELLOW, "Booklet PDF", "official, DE · FR · IT\n16–144 pages"),
    (0, 230, 250, 140, YELLOW, "Claim + vote", "in any of the\nthree languages"),
    (310, 0, 260, 170, BLUE, "1  Split", "≈900-char passages\nwith page + section\n(PyMuPDF, CPU)"),
    (630, 115, 270, 170, BLUE, "2  Search", "16 closest passages,\nacross languages\n(MiniLM, CPU)"),
    (960, 115, 280, 170, ORANGE, "3  Apertus", "reads only those 16,\nlabel + cited passages\n(8B / 70B, GPU)"),
    (1300, 60, 280, 280, GREEN, "Verdict", "supports /\ncontradicts /\ndoesn't settle\n+ 5 verbatim\nsentences with pages"),
]
ARROWS = [((250, 70), (310, 70)), ((570, 110), (630, 175)), ((250, 300), (630, 240)),
          ((900, 200), (960, 200)), ((1240, 200), (1300, 200))]
NOTE = (630, 320, 610, "≈ 3,400 tokens instead of ≈ 31,000 for the whole booklet")


def write_excalidraw() -> None:
    rng = random.Random(7)
    els = []

    def base(kind, x, y, w, h, **kw):
        return {"id": f"{kind}-{len(els)}", "type": kind, "x": x, "y": y, "width": w, "height": h,
                "angle": 0, "strokeColor": kw.get("stroke", INK),
                "backgroundColor": kw.get("fill", "transparent"), "fillStyle": "hachure",
                "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
                "groupIds": [], "frameId": None, "roundness": kw.get("roundness"),
                "seed": rng.randint(1, 2**31), "version": 1, "versionNonce": rng.randint(1, 2**31),
                "isDeleted": False, "boundElements": None, "updated": 1, "link": None, "locked": False}

    def text(x, y, w, s, size, color=INK):
        el = base("text", x, y, w, size * 1.25 * len(s.split("\n")), stroke=color)
        el.update({"text": s, "originalText": s, "fontSize": size, "fontFamily": 1,
                   "textAlign": "center", "verticalAlign": "top", "containerId": None,
                   "lineHeight": 1.25, "autoResize": True})
        els.append(el)

    for x, y, w, h, fill, title, body in BOXES:
        els.append(base("rectangle", x, y, w, h, fill=fill, roundness={"type": 3}))
        text(x + 10, y + 14, w - 20, title, 24)
        text(x + 10, y + 50, w - 20, body, 18, MUTED)
    for (x1, y1), (x2, y2) in ARROWS:
        el = base("arrow", x1, y1, x2 - x1, y2 - y1, roundness={"type": 2})
        el.update({"points": [[0, 0], [x2 - x1, y2 - y1]], "lastCommittedPoint": None,
                   "startBinding": None, "endBinding": None, "startArrowhead": None,
                   "endArrowhead": "arrow"})
        els.append(el)
    x, y, w, s = NOTE
    text(x, y, w, s, 18, MUTED)
    doc = {"type": "excalidraw", "version": 2, "source": "booklet-nli/paper/pipeline_diagram.py",
           "elements": els, "appState": {"viewBackgroundColor": "#ffffff"}, "files": {}}
    (HERE / "pipeline.excalidraw").write_text(json.dumps(doc, indent=1, ensure_ascii=False))


def hand_font() -> str:
    """A handwriting font that ships with macOS, else the default."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Chalkboard SE", "Chalkboard", "Marker Felt", "Bradley Hand", "Comic Sans MS"):
        if name in have:
            return name
    return plt.rcParams["font.family"][0]


def draw_pdf() -> None:
    plt.rcParams.update({"font.family": hand_font(), "pdf.fonttype": 42,
                         "path.sketch": (1.2, 60, 2)})  # Excalidraw-like wobble
    W, H = 1590, 370
    fig, ax = plt.subplots(figsize=(6.9, 6.9 * H / W))
    ax.set_xlim(-10, W)
    ax.set_ylim(H + 5, -10)
    ax.axis("off")
    for x, y, w, h, fill, title, body in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                    facecolor=fill, edgecolor=INK, linewidth=1.3))
        ax.text(x + w / 2, y + 28, title, ha="center", va="center", fontsize=7.6, color=INK,
                fontweight="bold")
        ax.text(x + w / 2, y + 56, body, ha="center", va="top", fontsize=5.7, color=MUTED,
                linespacing=1.25)
    for (x1, y1), (x2, y2) in ARROWS:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.2, color=INK))
    x, y, w, s = NOTE
    ax.text(x + w / 2, y + 10, s, ha="center", va="top", fontsize=6.0, color=MUTED, style="italic")
    fig.savefig(HERE / "fig-pipeline.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    write_excalidraw()
    draw_pdf()
    print("wrote pipeline.excalidraw and fig-pipeline.pdf, font:", plt.rcParams["font.family"])
