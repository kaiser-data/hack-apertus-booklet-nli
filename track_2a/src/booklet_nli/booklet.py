"""Turn a voting booklet PDF into page-anchored passages."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pymupdf

from .config import CACHE_DIR

# Lines in these PDFs break words without a hyphen ("landwirt\nschaftliche"); real line ends
# carry a trailing space. Joining a lowercase-to-lowercase break without a space restores the word.
_SOFT_BREAK = re.compile(r"(?<=[a-zäöüßéèêàâçùûôîïë])-?\n(?=[a-zäöüßéèêàâçùûôîïë])")
_SOFT_HYPHEN = re.compile(r"­\s*")
_WS = re.compile(r"\s+")
_NON_SECTION = re.compile(r"(?i)^(volksabstimmung|votation|votazione|inhalt|sommaire|indice)")
# Short blocks are kept (rates, recommendations, table cells) unless they are pure page furniture.
_NOISE = re.compile(r"^[\W\d_]{0,6}$")  # page numbers, bullets, footnote markers
_SENTENCE_END = re.compile(r"(?<=[.!?;:»])\s+")
OVERLAP_CHARS = 150  # tail of the previous passage on the same page, so a fact split by a cut survives
PARSER_VERSION = 4  # bump when passage extraction changes; part of the cache key
USER_AGENT = "Mozilla/5.0 (compatible; booklet-nli/0.1; +https://hackapertus.ch)"


@dataclass(frozen=True)
class Passage:
    id: int
    page: int  # 1-based
    section: str  # outline path, e.g. "<proposal> › In Kürze › Ausgangslage"; "" if none
    text: str


def fetch_pdf(source: str) -> Path:
    """Return a local path for a booklet given as a path or URL; URLs are cached by hash."""
    if not source.startswith(("http://", "https://")):
        return Path(source)
    target = CACHE_DIR / "pdf" / f"{hashlib.sha1(source.encode()).hexdigest()[:16]}.pdf"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        # bk.admin.ch drops connections from httpx's default user agent.
        transport = httpx.HTTPTransport(retries=3)
        with httpx.Client(transport=transport, headers={"User-Agent": USER_AGENT},
                          follow_redirects=True, timeout=120) as client:
            resp = client.get(source)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    return target


def _clean(text: str) -> str:
    return _WS.sub(" ", _SOFT_BREAK.sub("", _SOFT_HYPHEN.sub("", text))).strip()


def _section_by_page(doc: pymupdf.Document) -> dict[int, str]:
    """Map each page to the outline path (proposal › part › subsection) open at that page.

    Outline depth differs between booklets, so the path is kept whole, minus the cover heading
    ("Volksabstimmung vom …") and the table of contents.
    """
    toc = [(lvl, _clean(title), page) for lvl, title, page in doc.get_toc() if page > 0]
    sections: dict[int, str] = {}
    path: list[str] = []
    entries = iter(toc)  # outline order is reading order
    nxt = next(entries, None)
    for page in range(1, doc.page_count + 1):
        while nxt and nxt[2] <= page:
            lvl, title, _ = nxt
            path = path[: max(lvl - 1, 0)] + [title]
            nxt = next(entries, None)
        shown = [t for t in path if not _NON_SECTION.match(t)]
        sections[page] = " › ".join(shown[-3:])
    return sections


def extract_passages(pdf: Path, max_chars: int = 900) -> list[Passage]:
    """Split a booklet into passages of at most ~max_chars, never crossing a page."""
    cache = CACHE_DIR / "passages" / f"{_file_hash(pdf)}-{max_chars}-v{PARSER_VERSION}.json"
    if cache.exists():
        return [Passage(**p) for p in json.loads(cache.read_text())]

    doc = pymupdf.open(pdf)
    sections = _section_by_page(doc)
    passages: list[Passage] = []
    for page_no, page in enumerate(doc, start=1):
        section = sections[page_no]
        units = [u for block in page.get_text("blocks") if block[6] == 0  # skip image blocks
                 for u in _units(_clean(block[4]), max_chars)]
        buf = ""
        for text in units:
            if buf and len(buf) + len(text) + 1 > max_chars:
                passages.append(Passage(len(passages), page_no, section, buf))
                tail = _tail(buf, OVERLAP_CHARS)
                buf = tail if len(tail) + len(text) + 1 <= max_chars else ""
            buf = f"{buf} {text}".strip()
        if buf:
            passages.append(Passage(len(passages), page_no, sections[page_no], buf))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(p) for p in passages], ensure_ascii=False))
    return passages


def _units(text: str, max_chars: int) -> list[str]:
    """A cleaned block as pieces of at most ~max_chars, split at sentence ends; [] for noise."""
    if not text or _NOISE.match(text):
        return []
    if len(text) <= max_chars:
        return [text]
    pieces, cur = [], ""
    for sentence in _SENTENCE_END.split(text):
        if cur and len(cur) + len(sentence) + 1 > max_chars:
            pieces.append(cur)
            cur = ""
        cur = f"{cur} {sentence}".strip()
    pieces.append(cur)
    # A single sentence longer than max_chars is cut at a word boundary.
    return [chunk for piece in pieces for chunk in _hard_wrap(piece, max_chars)]


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    chunks = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        cut = cut if cut > max_chars // 2 else max_chars
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return chunks + [text] if text else chunks


def _tail(text: str, n: int) -> str:
    """The last ~n characters of text, starting at a word boundary."""
    if len(text) <= n:
        return text
    start = text.find(" ", len(text) - n - 1)
    return text[start + 1:] if start != -1 else ""


def _file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]
