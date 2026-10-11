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
MIN_BLOCK_CHARS = 25
PARSER_VERSION = 2  # bump when passage extraction changes; part of the cache key
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
        buf = ""
        for block in page.get_text("blocks"):
            if block[6] != 0:  # image block
                continue
            text = _clean(block[4])
            if len(text) < MIN_BLOCK_CHARS:
                continue  # page numbers, bullets, footnote markers
            if buf and len(buf) + len(text) > max_chars:
                passages.append(Passage(len(passages), page_no, sections[page_no], buf))
                buf = ""
            buf = f"{buf} {text}".strip()
        if buf:
            passages.append(Passage(len(passages), page_no, sections[page_no], buf))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(p) for p in passages], ensure_ascii=False))
    return passages


def _file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]
