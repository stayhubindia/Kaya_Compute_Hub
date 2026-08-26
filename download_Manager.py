#!/usr/bin/env python3
import os

os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")

import sys, re, time, math, sqlite3, hashlib, threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QProgressBar, QPlainTextEdit, QGroupBox, QSpinBox, QMessageBox

ROOT=Path.cwd(); HTML=ROOT/"html"; PDF=ROOT/"pdf"; STATE=ROOT/"state"; DB=STATE/"arxiv_downloads.sqlite3"
for p in (HTML,PDF,STATE): p.mkdir(parents=True,exist_ok=True)
UA="PhysicsDatasetDownloader/2.0 (mailto:set-your-contact-email@example.com)"; REFRESH_EVERY=50


def now(): return datetime.now(timezone.utc).isoformat()
def valid_html(p):
    if not p.exists() or p.stat().st_size<1000:return False
    try:
        b=p.read_bytes()[:65536].lower(); return b"<html" in b or b"<!doctype html" in b
    except OSError:return False
def valid_pdf(p):
    if not p.exists() or p.stat().st_size<10000:return False
    try:
        with p.open("rb") as f:return f.read(5)==b"%PDF-"
    except OSError:return False
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1048576),b""):h.update(c)
    return h.hexdigest()
def eta(sec):
    if sec is None or not math.isfinite(sec):return "--"
    sec=max(0,int(sec)); h,r=divmod(sec,3600); m,s=divmod(r,60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

class DBM:
    def __init__(self,p):
        self.p=p; self.local=threading.local()
        c=sqlite3.connect(p); c.execute("PRAGMA journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS collections(
        id INTEGER PRIMARY KEY,category TEXT,month TEXT,total INTEGER DEFAULT 0,discovered INTEGER DEFAULT 0,updated TEXT,
        UNIQUE(category,month));
        CREATE TABLE IF NOT EXISTS papers(
        id INTEGER PRIMARY KEY,collection_id INTEGER,arxiv_id TEXT,title TEXT DEFAULT '',status TEXT DEFAULT 'queued',
        format TEXT DEFAULT '',path TEXT DEFAULT '',size INTEGER DEFAULT 0,sha256 TEXT DEFAULT '',
        attempts INTEGER DEFAULT 0,last_error TEXT DEFAULT '',updated TEXT,downloaded TEXT,
        UNIQUE(collection_id,arxiv_id));
        """); c.commit(); c.close()
    def c(self):
        if not hasattr(self.local,"c"): self.local.c=sqlite3.connect(self.p,timeout=30,check_same_thread=False)
        return self.local.c
    def cid(self,cat,mon):
        c=self.c(); c.execute("INSERT OR IGNORE INTO collections(category,month,updated) VALUES(?,?,?)",(cat,mon,now())); c.commit()
        return c.execute("SELECT id FROM collections WHERE category=? AND month=?",(cat,mon)).fetchone()[0]
    def add(self,cat,mon,aid):
        cid=self.cid(cat,mon); c=self.c()
        c.execute("INSERT OR IGNORE INTO papers(collection_id,arxiv_id,updated) VALUES(?,?,?)",(cid,aid,now())); c.commit()
    def collection(self,cat,mon,total):
        c=self.c(); cid=self.cid(cat,mon); c.execute("UPDATE collections SET total=?,discovered=?,updated=? WHERE id=?",(total,total,now(),cid)); c.commit()
    def update(self,cat,mon,aid,**kw):
        cid=self.cid(cat,mon); kw["updated"]=now(); c=self.c(); ks=list(kw)
        c.execute("UPDATE papers SET "+",".join(k+"=?" for k in ks)+" WHERE collection_id=? AND arxiv_id=?",(*[kw[k] for k in ks],cid,aid)); c.commit()
    def ids(self,cat,mon):
        cid=self.cid(cat,mon); return [x[0] for x in self.c().execute("SELECT arxiv_id FROM papers WHERE collection_id=? ORDER BY id",(cid,))]
    def counts(self,cat,mon):
        cid=self.cid(cat,mon); d={}
        for s,n in self.c().execute("SELECT status,count(*) FROM papers WHERE collection_id=? GROUP BY status",(cid,)):d[s]=n
        return d

class S(QObject):
    log=Signal(str); status=Signal(str); stats=Signal(dict); done=Signal(); stopped=Signal(); error=Signal(str)

DEFAULT_WORKERS = 4
MIN_WORKERS = 1
MAX_WORKERS = 8
MAX_RETRIES = 5
MAX_BACKOFF = 90
MIN_LIST_DELAY = 3.0  # seconds between /list/ page requests — arXiv asks for polite, non-burst access

class AdaptiveLimiter:
    """Reduce concurrency after transport/rate-limit errors and recover slowly."""
    def __init__(self, initial=DEFAULT_WORKERS):
        self.target = max(MIN_WORKERS, min(initial, MAX_WORKERS))
        self.stable = 0
        self.errors = 0
        self.lock = threading.Lock()

    def failure(self):
        with self.lock:
            self.errors += 1
            self.stable = 0
            self.target = max(MIN_WORKERS, self.target - 1)
            return self.target

    def success(self):
        with self.lock:
            self.stable += 1
            if self.stable >= 25 and self.errors == 0:
                self.target = min(MAX_WORKERS, self.target + 1)
                self.stable = 0
            elif self.stable >= 25:
                self.stable = 0
                self.errors = 0
            return self.target

    def get(self):
        with self.lock:
            return self.target


class Manager:
    def __init__(self):
        self.s=S(); self.db=DBM(DB); self.stop=threading.Event(); self.pause=threading.Event(); self.pause.set(); self.thread=None
        self.local=threading.local(); self.limiter=AdaptiveLimiter(DEFAULT_WORKERS)
        self.stat={"total":0,"discovered":0,"processed":0,"html":0,"pdf":0,"existing":0,"failed":0,"bytes":0}
    def sess(self):
        if not hasattr(self.local,"sess"):
            x=requests.Session(); x.headers["User-Agent"]=UA
            adapter=requests.adapters.HTTPAdapter(pool_connections=10,pool_maxsize=10)
            x.mount("https://",adapter); x.mount("http://",adapter)
            self.local.sess=x
        return self.local.sess
    def wait(self):
        while not self.pause.is_set():
            if self.stop.is_set():return False
            time.sleep(.2)
        return not self.stop.is_set()
    def start(self,cat,mon,workers,delay):
        self.stop.clear(); self.pause.set()
        self.thread=threading.Thread(target=self.run,args=(cat,mon,workers,delay),daemon=True); self.thread.start()
    def pause_(self):self.pause.clear();self.s.status.emit("Paused")
    def resume(self):self.pause.set();self.s.status.emit("Running")
    def stop_(self):self.stop.set();self.pause.set()
    def retry_wait(self,n,r=None):
        self.limiter.failure()  # back off future requests after any retry-triggering error
        w=None
        if r is not None:
            try:w=float(r.headers.get("Retry-After",""))
            except ValueError:pass
        w=min(w if w is not None else 2**(n-1),60)
        end=time.monotonic()+w
        while time.monotonic()<end:
            if not self.wait():return False
            time.sleep(.1)
        return True
    def adaptive_delay(self,base):
        """Extra pacing on top of the user-set delay: widens after recent errors, narrows on a clean streak."""
        self.limiter.success()
        extra=(DEFAULT_WORKERS-self.limiter.get())*0.5  # 0s when healthy, up to ~3.5s when badly throttled
        return max(base,extra)
    def discover(self,cat,mon):
        ses=self.sess(); skip=0; page=1; seen=[]; expected_total=None
        while not self.stop.is_set():
            if not self.wait():return []
            url=f"https://arxiv.org/list/{cat}/{mon}?skip={skip}&show=50"
            self.s.status.emit(f"Discovering page {page} (skip={skip})"); self.s.log.emit("[LIST] GET "+url)
            for n in range(1,MAX_RETRIES+1):
                try:
                    r=ses.get(url,timeout=(10,60)); r.raise_for_status(); break
                except requests.RequestException as e:
                    self.s.log.emit(f"[LIST] retry {n}/{MAX_RETRIES} ({type(e).__name__})")
                    if n>=MAX_RETRIES or not self.retry_wait(n):return seen
            soup=BeautifulSoup(r.text,"html.parser")
            if expected_total is None:
                m=re.search(r"[Tt]otal of\s+([\d,]+)\s+entries",soup.get_text())
                if m:expected_total=int(m.group(1).replace(",",""))
            es=soup.select("dt")
            ids=[]
            for e in es:
                a=e.select_one("a[href*='/abs/']")
                if a:
                    m=re.search(r"/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",a.get("href",""))
                    if m and m.group(1) not in seen:seen.append(m.group(1));ids.append(m.group(1));self.db.add(cat,mon,m.group(1))
            self.stat["discovered"]=len(seen)
            self.stat["total"]=expected_total if expected_total is not None else len(seen)
            self.s.stats.emit(dict(self.stat))
            self.s.log.emit(f"[LIST] HTTP {r.status_code} | {len(r.content):,} bytes | +{len(ids)} | total={len(seen)}"+(f"/{expected_total}" if expected_total else ""))
            # stop once we've matched the page's own reported count, or a page comes back with nothing new (safety net)
            if (expected_total is not None and len(seen)>=expected_total) or len(es)==0:break
            if len(es)<50 and expected_total is None:break  # fallback for pages without a parseable total
            skip+=50;page+=1
            if not self.wait():return seen
            time.sleep(MIN_LIST_DELAY)  # politeness delay between list pages
        self.db.collection(cat,mon,len(seen)); return seen
    def html(self,aid):
        out=HTML/f"{aid}.html"; part=HTML/f"{aid}.html.part"
        if valid_html(out):return "existing",out
        part.unlink(missing_ok=True)
        for n in range(1,MAX_RETRIES+1):
            if not self.wait():return "stopped",None
            try:
                r=self.sess().get(f"https://arxiv.org/html/{aid}",timeout=(10,90))
                if r.status_code==200 and "text/html" in r.headers.get("content-type","").lower() and len(r.content)>=1000:
                    part.write_bytes(r.content)
                    if valid_html(part):part.replace(out);return "html",out
                if r.status_code in (404,410):return "no_html",None
                if r.status_code in (403,429,500,502,503,504) and n<MAX_RETRIES:
                    if self.retry_wait(n,r):continue
                    return "stopped",None
                return "no_html",None
            except requests.RequestException as e:
                self.s.log.emit(f"[HTML] {aid}: retry {n}/{MAX_RETRIES} ({type(e).__name__})")
                if n<MAX_RETRIES:
                    if self.retry_wait(n):continue
                    return "stopped",None
        return "no_html",None
    def pdf(self,aid):
        out=PDF/f"{aid}.pdf"; part=PDF/f"{aid}.pdf.part"
        if valid_pdf(out):return "existing",out
        part.unlink(missing_ok=True)
        for n in range(1,MAX_RETRIES+1):
            if not self.wait():return "stopped",None
            try:
                r=self.sess().get(f"https://arxiv.org/pdf/{aid}",timeout=(10,180),stream=True)
                if r.status_code in (403,429,500,502,503,504) and n<MAX_RETRIES:
                    if self.retry_wait(n,r):continue
                    return "stopped",None
                if r.status_code!=200:return "failed",None
                with r:
                    it=r.iter_content(262144); first=next(it,b"")
                    if not first.startswith(b"%PDF-"):return "failed",None
                    with part.open("wb") as f:
                        f.write(first)
                        for c in it:
                            if self.stop.is_set():part.unlink(missing_ok=True);return "stopped",None
                            if c:f.write(c)
                if valid_pdf(part):part.replace(out);return "pdf",out
            except (requests.RequestException,StopIteration) as e:
                self.s.log.emit(f"[PDF] {aid}: retry {n}/{MAX_RETRIES} ({type(e).__name__})")
                if n<MAX_RETRIES:
                    if self.retry_wait(n):continue
                    return "stopped",None
        return "failed",None
    def one(self,cat,mon,aid):
        hp=HTML/f"{aid}.html"; pp=PDF/f"{aid}.pdf"
        if valid_html(hp):return "existing",hp.stat().st_size
        if valid_pdf(pp):return "existing",pp.stat().st_size
        self.s.status.emit("HTML → "+aid); typ,p=self.html(aid)
        if typ in ("html","existing"):
            z=p.stat().st_size;self.db.update(cat,mon,aid,status="html",format="html",path=str(p),size=z,sha256=sha(p),downloaded=now());return typ,z
        if typ=="stopped":return "stopped",0
        self.db.update(cat,mon,aid,status="html_unavailable",last_error="HTML unavailable")
        self.s.status.emit("PDF fallback → "+aid);typ,p=self.pdf(aid)
        if typ in ("pdf","existing"):
            z=p.stat().st_size;self.db.update(cat,mon,aid,status="pdf",format="pdf",path=str(p),size=z,sha256=sha(p),downloaded=now());return typ,z
        if typ=="stopped":return "stopped",0
        self.db.update(cat,mon,aid,status="failed",last_error="HTML and PDF failed");return "failed",0
    def run(self,cat,mon,workers,delay):
        try:
            ids=self.discover(cat,mon)
            if self.stop.is_set():self.s.stopped.emit();return
            start=time.monotonic();batch=0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fs={ex.submit(self.one,cat,mon,aid):aid for aid in ids}
                for f in as_completed(fs):
                    aid=fs[f]
                    if self.stop.is_set():break
                    try:t,z=f.result()
                    except Exception as e:t,z="failed",0;self.s.log.emit(f"[ERROR] {aid}: {e}")
                    if t=="stopped":continue  # user stopped mid-retry — don't count as processed/failed
                    self.stat["processed"]+=1;self.stat["bytes"]+=z
                    if t=="html":self.stat["html"]+=1
                    elif t=="pdf":self.stat["pdf"]+=1
                    elif t=="existing":self.stat["existing"]+=1
                    elif t=="failed":self.stat["failed"]+=1
                    batch+=1
                    self.s.stats.emit(dict(self.stat))
                    speed=self.stat["processed"]/max(time.monotonic()-start,.001)*60
                    rem=self.stat["total"]-self.stat["processed"]; self.s.status.emit(f"{speed:.1f} papers/min | ETA {eta(rem/speed*60) if speed else '--'}")
                    self.s.log.emit(("✓" if t!="failed" else "✗")+f" {aid} [{t}]")
                    if batch>=REFRESH_EVERY:batch=0;self.s.log.emit("──────── 50 transactions completed / activity refresh ────────")
                    d=self.adaptive_delay(delay)
                    if d:time.sleep(d)
            if self.stop.is_set():self.s.stopped.emit()
            else:self.s.done.emit()
        except Exception as e:self.s.error.emit(str(e))

class Win(QWidget):
    def __init__(self):
        super().__init__();self.setWindowTitle("arXiv Download Manager v2");self.resize(1100,780);self.m=Manager();self.build();self.connect()
    def build(self):
        r=QVBoxLayout(self);g=QGridLayout();box=QGroupBox("Collection / Speed");box.setLayout(g)
        self.cat=QLineEdit("astro-ph");self.mon=QLineEdit("2026-01");self.work=QSpinBox();self.work.setRange(1,16);self.work.setValue(4);self.delay=QSpinBox();self.delay.setRange(0,30);self.delay.setValue(2)
        for i,(n,w) in enumerate([("Category",self.cat),("Month",self.mon),("Workers",self.work),("Delay",self.delay)]):g.addWidget(QLabel(n),0,i*2);g.addWidget(w,0,i*2+1)
        r.addWidget(box);sg=QGridLayout();sb=QGroupBox("Pinned status");sb.setLayout(sg);self.lab={}
        for i,k in enumerate(["total","discovered","processed","html","pdf","existing","failed","bytes"]):
            self.lab[k]=QLabel(k.title()+": 0");self.lab[k].setStyleSheet("font-weight:600;font-size:14px");sg.addWidget(self.lab[k],i//4,i%4)
        r.addWidget(sb);self.prog=QProgressBar();r.addWidget(self.prog);self.status=QLabel("Ready");self.status.setStyleSheet("font-weight:700;font-size:15px");r.addWidget(self.status)
        b=QHBoxLayout();self.start=QPushButton("▶ Start");self.pause=QPushButton("⏸ Pause");self.resume=QPushButton("▶ Resume");self.stop=QPushButton("■ Stop")
        for x in [self.pause,self.resume,self.stop]:x.setEnabled(False)
        self.start.clicked.connect(self.go);self.pause.clicked.connect(self.do_pause);self.resume.clicked.connect(self.do_resume);self.stop.clicked.connect(self.do_stop);[b.addWidget(x) for x in [self.start,self.pause,self.resume,self.stop]];r.addLayout(b)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(500);r.addWidget(self.log,1)
    def do_pause(self):self.m.pause_();self.pause.setEnabled(False);self.resume.setEnabled(True)
    def do_resume(self):self.m.resume();self.resume.setEnabled(False);self.pause.setEnabled(True)
    def do_stop(self):self.m.stop_()
    def connect(self):
        s=self.m.s;s.log.connect(self.log.appendPlainText);s.status.connect(self.status.setText);s.stats.connect(self.stats);s.done.connect(self.done);s.stopped.connect(self.stopped);s.error.connect(self.error)
    def stats(self,d):
        for k,l in self.lab.items():
            v=d.get(k,0);l.setText(f"{k.title()}: {v/1024**2:.1f} MB" if k=="bytes" else f"{k.title()}: {v:,}")
        self.prog.setValue(int(d.get("processed",0)*100/max(d.get("total",1),1)))
    def go(self):
        self.start.setEnabled(False);self.pause.setEnabled(True);self.stop.setEnabled(True);self.resume.setEnabled(False);self.log.clear()
        if self.m.thread and self.m.thread.is_alive():
            self.m.stop_();self.m.thread.join(3)
        self.m=Manager();self.connect();self.m.start(self.cat.text().strip(),self.mon.text().strip(),self.work.value(),self.delay.value())
    def done(self):self.start.setEnabled(True);self.pause.setEnabled(False);self.resume.setEnabled(False);self.stop.setEnabled(False);self.status.setText("✓ Complete")
    def stopped(self):self.start.setEnabled(True);self.pause.setEnabled(False);self.resume.setEnabled(False);self.stop.setEnabled(False);self.status.setText("■ Stopped safely; valid files preserved")
    def error(self,e):self.start.setEnabled(True);self.pause.setEnabled(False);self.stop.setEnabled(False);self.status.setText("✗ Error");self.log.appendPlainText("[ERROR] "+e)
    def closeEvent(self,e):
        self.m.stop_()
        if self.m.thread:self.m.thread.join(3)
        e.accept()

if __name__=="__main__":
    app=QApplication(sys.argv);w=Win();w.show();sys.exit(app.exec())