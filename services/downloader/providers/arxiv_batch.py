#!/usr/bin/env python3
"""
Headless ArXiv Batch Downloader Engine.
Extracted and adapted from download_Manager.py (GUI removed).
Supports: discover papers by category/month, download HTML+PDF with retry/backoff.
"""

import re
import time
import math
import hashlib
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, Dict, Any, List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

UA = "KayaResearchDownloader/1.0 (Kaya Compute Hub; mailto:admin@kaya.local)"

MAX_RETRIES   = 5
MAX_WORKERS   = 6
MIN_LIST_DELAY = 3.0  # polite delay between ArXiv list-page requests


# ── utility helpers ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_html(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 1000:
        return False
    try:
        b = p.read_bytes()[:65536].lower()
        return b"<html" in b or b"<!doctype html" in b
    except OSError:
        return False


def _valid_pdf(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 10000:
        return False
    try:
        with p.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1_048_576), b""):
            h.update(chunk)
    return h.hexdigest()


def _eta_str(seconds) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


# ── adaptive concurrency limiter ─────────────────────────────────────────────

class _AdaptiveLimiter:
    def __init__(self, initial: int = 4):
        self._target = max(1, min(initial, MAX_WORKERS))
        self._stable = 0
        self._errors = 0
        self._lock = threading.Lock()

    def failure(self) -> int:
        with self._lock:
            self._errors += 1
            self._stable = 0
            self._target = max(1, self._target - 1)
            return self._target

    def success(self) -> int:
        with self._lock:
            self._stable += 1
            if self._stable >= 25 and self._errors == 0:
                self._target = min(MAX_WORKERS, self._target + 1)
                self._stable = 0
            elif self._stable >= 25:
                self._stable = 0
                self._errors = 0
            return self._target

    def get(self) -> int:
        with self._lock:
            return self._target


# ── main headless engine ─────────────────────────────────────────────────────

class ArXivBatchEngine:
    """
    Headless batch downloader.  Thread-safe; safe to call from Celery workers.

    :param output_dir:  Root directory to store html/ and pdf/ sub-folders.
    :param workers:     Parallel download threads (1–6).
    :param delay:       Base delay (s) between individual paper downloads.
    :param on_progress: Optional callback(stat_dict) called after every paper.
    :param stop_event:  Optional threading.Event to cancel a running job.
    """

    def __init__(
        self,
        output_dir: str | Path,
        workers: int = 4,
        delay: float = 1.0,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        self.html_dir = Path(output_dir) / "html"
        self.pdf_dir  = Path(output_dir) / "pdf"
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.workers     = max(1, min(workers, MAX_WORKERS))
        self.delay       = max(0.0, delay)
        self.on_progress = on_progress
        self.stop_event  = stop_event or threading.Event()
        self._limiter    = _AdaptiveLimiter(self.workers)
        self._local      = threading.local()

        self.stats: Dict[str, Any] = {
            "total": 0, "discovered": 0, "processed": 0,
            "html": 0, "pdf": 0, "existing": 0, "failed": 0,
            "bytes": 0,
        }

    # -- session --

    def _sess(self) -> requests.Session:
        if not hasattr(self._local, "sess"):
            s = requests.Session()
            s.headers["User-Agent"] = UA
            a = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
            s.mount("https://", a)
            s.mount("http://",  a)
            self._local.sess = s
        return self._local.sess

    def _stopped(self) -> bool:
        return self.stop_event.is_set()

    # -- retry helper --

    def _retry_wait(self, attempt: int, response=None) -> bool:
        self._limiter.failure()
        wait = None
        if response is not None:
            try:
                wait = float(response.headers.get("Retry-After", ""))
            except (ValueError, TypeError):
                pass
        wait = min(wait if wait is not None else 2 ** (attempt - 1), 60)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if self._stopped():
                return False
            time.sleep(0.2)
        return True

    # -- discover --

    def discover(self, category: str, month: str) -> List[str]:
        """Crawl ArXiv list pages and return all paper IDs for category/month."""
        ses = self._sess()
        skip = 0
        page = 1
        seen: List[str] = []
        expected_total: Optional[int] = None

        while not self._stopped():
            url = f"https://arxiv.org/list/{category}/{month}?skip={skip}&show=50"
            logger.info(f"[DISCOVER] Page {page} | {url}")

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    r = ses.get(url, timeout=(10, 60))
                    r.raise_for_status()
                    break
                except requests.RequestException as exc:
                    logger.warning(f"[DISCOVER] Retry {attempt}/{MAX_RETRIES}: {exc}")
                    if attempt >= MAX_RETRIES or not self._retry_wait(attempt):
                        return seen

            soup = BeautifulSoup(r.text, "html.parser")

            if expected_total is None:
                m = re.search(r"[Tt]otal of\s+([\d,]+)\s+entries", soup.get_text())
                if m:
                    expected_total = int(m.group(1).replace(",", ""))
                    self.stats["total"] = expected_total

            entries = soup.select("dt")
            new_ids = []
            for e in entries:
                a = e.select_one("a[href*='/abs/']")
                if a:
                    m = re.search(r"/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", a.get("href", ""))
                    if m and m.group(1) not in seen:
                        seen.append(m.group(1))
                        new_ids.append(m.group(1))

            self.stats["discovered"] = len(seen)
            if not self.stats.get("total"):
                self.stats["total"] = len(seen)
            self._emit()

            logger.info(f"[DISCOVER] HTTP {r.status_code} | +{len(new_ids)} papers | total={len(seen)}" +
                        (f"/{expected_total}" if expected_total else ""))

            if (expected_total and len(seen) >= expected_total) or not entries:
                break
            if len(entries) < 50 and expected_total is None:
                break

            skip += 50
            page += 1
            time.sleep(MIN_LIST_DELAY)

        return seen

    # -- single paper download --

    def _download_html(self, arxiv_id: str):
        out  = self.html_dir / f"{arxiv_id}.html"
        part = self.html_dir / f"{arxiv_id}.html.part"
        if _valid_html(out):
            return "existing", out
        part.unlink(missing_ok=True)
        for attempt in range(1, MAX_RETRIES + 1):
            if self._stopped():
                return "stopped", None
            try:
                r = self._sess().get(f"https://arxiv.org/html/{arxiv_id}", timeout=(10, 90))
                if (r.status_code == 200 and
                        "text/html" in r.headers.get("content-type", "").lower() and
                        len(r.content) >= 1000):
                    part.write_bytes(r.content)
                    if _valid_html(part):
                        part.replace(out)
                        return "html", out
                if r.status_code in (404, 410):
                    return "no_html", None
                if r.status_code in (403, 429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    if self._retry_wait(attempt, r):
                        continue
                    return "stopped", None
                return "no_html", None
            except requests.RequestException as exc:
                logger.warning(f"[HTML] {arxiv_id} retry {attempt}: {exc}")
                if attempt < MAX_RETRIES:
                    if self._retry_wait(attempt):
                        continue
                    return "stopped", None
        return "no_html", None

    def _download_pdf(self, arxiv_id: str):
        out  = self.pdf_dir / f"{arxiv_id}.pdf"
        part = self.pdf_dir / f"{arxiv_id}.pdf.part"
        if _valid_pdf(out):
            return "existing", out
        part.unlink(missing_ok=True)
        for attempt in range(1, MAX_RETRIES + 1):
            if self._stopped():
                return "stopped", None
            try:
                r = self._sess().get(
                    f"https://arxiv.org/pdf/{arxiv_id}",
                    timeout=(10, 180), stream=True
                )
                if r.status_code in (403, 429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    if self._retry_wait(attempt, r):
                        continue
                    return "stopped", None
                if r.status_code != 200:
                    return "failed", None
                with r:
                    it = r.iter_content(262144)
                    first = next(it, b"")
                    if not first.startswith(b"%PDF-"):
                        return "failed", None
                    with part.open("wb") as f:
                        f.write(first)
                        for chunk in it:
                            if self._stopped():
                                part.unlink(missing_ok=True)
                                return "stopped", None
                            if chunk:
                                f.write(chunk)
                if _valid_pdf(part):
                    part.replace(out)
                    return "pdf", out
            except (requests.RequestException, StopIteration) as exc:
                logger.warning(f"[PDF] {arxiv_id} retry {attempt}: {exc}")
                if attempt < MAX_RETRIES:
                    if self._retry_wait(attempt):
                        continue
                    return "stopped", None
        return "failed", None

    def _download_one(self, arxiv_id: str):
        hp = self.html_dir / f"{arxiv_id}.html"
        pp = self.pdf_dir  / f"{arxiv_id}.pdf"
        if _valid_html(hp):
            return "existing", hp.stat().st_size, str(hp)
        if _valid_pdf(pp):
            return "existing", pp.stat().st_size, str(pp)

        typ, p = self._download_html(arxiv_id)
        if typ in ("html", "existing") and p:
            return typ, p.stat().st_size, str(p)
        if typ == "stopped":
            return "stopped", 0, ""

        typ, p = self._download_pdf(arxiv_id)
        if typ in ("pdf", "existing") and p:
            return typ, p.stat().st_size, str(p)
        if typ == "stopped":
            return "stopped", 0, ""

        return "failed", 0, ""

    # -- emit progress --

    def _emit(self):
        if self.on_progress:
            try:
                self.on_progress(dict(self.stats))
            except Exception:
                pass

    # -- run batch ────────────────────────────────────────────────────────────

    def run(self, category: str, month: str) -> Dict[str, Any]:
        """
        Full pipeline: discover → download all papers.
        Returns final stats dict.
        """
        logger.info(f"[ARXIV BATCH] Starting: category={category} month={month}")
        paper_ids = self.discover(category, month)

        if self._stopped():
            self.stats["status"] = "stopped"
            return self.stats

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._download_one, aid): aid for aid in paper_ids}
            for future in as_completed(futures):
                aid = futures[future]
                if self._stopped():
                    break
                try:
                    typ, size, path = future.result()
                except Exception as exc:
                    typ, size, path = "failed", 0, ""
                    logger.error(f"[BATCH] {aid}: {exc}")

                if typ == "stopped":
                    continue

                self.stats["processed"] += 1
                self.stats["bytes"]     += size
                if typ == "html":     self.stats["html"]     += 1
                elif typ == "pdf":    self.stats["pdf"]      += 1
                elif typ == "existing": self.stats["existing"] += 1
                elif typ == "failed": self.stats["failed"]   += 1

                elapsed = max(time.monotonic() - start, 0.001)
                speed   = self.stats["processed"] / elapsed * 60
                remaining = self.stats["total"] - self.stats["processed"]
                self.stats["eta"]   = _eta_str(remaining / speed * 60 if speed else None)
                self.stats["speed"] = round(speed, 1)

                symbol = "✓" if typ != "failed" else "✗"
                logger.info(f"[BATCH] {symbol} {aid} [{typ}] | {self.stats['processed']}/{self.stats['total']}")
                self._emit()

                # adaptive inter-paper delay
                extra = (4 - self._limiter.get()) * 0.5
                delay = max(self.delay, extra)
                if delay > 0:
                    time.sleep(delay)

        self.stats["status"] = "stopped" if self._stopped() else "complete"
        logger.info(f"[ARXIV BATCH] Finished: {self.stats}")
        return self.stats
