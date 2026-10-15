import json, sys
from pathlib import Path
import pymupdf
from booklet_nli.booklet import _clean, extract_passages
from booklet_nli.retrieve import rank
E=Path(sys.argv[1]); cases={c["id"]:c for c in map(json.loads,open(E/"dev150/cases.jsonl"))}
preds=[json.loads(l) for l in open(E/"pred-vote.jsonl")]
pages={}
def page_text(path,n):
    if (path,n) not in pages:
        doc=pymupdf.open(path); pages.update({(path,i+1):_clean(p.get_text()) for i,p in enumerate(doc)})
    return pages[(path,n)]
out={v:[] for v in ("page","page+pad3","page+pad5")}
for p in preds:
    c=cases[p["id"]]
    if "booklet" not in c:
        for v in out: out[v].append(p)
        continue
    path=str(E/"dev150"/c["booklet"]["path"])
    cited=[]
    for e in p["evidence"]:
        if e["page"] not in cited: cited.append(e["page"])
    ranked_pages=[]
    if p["label"]!=1:
        for ps in rank(c["claim"]["text"], extract_passages(Path(path)))[:16]:
            if ps.page not in ranked_pages: ranked_pages.append(ps.page)
    for v,n in (("page",0),("page+pad3",3),("page+pad5",5)):
        pg=list(cited)
        for r in ranked_pages:
            if len(pg)>=max(n,len(cited)): break
            if r not in pg: pg.append(r)
        ev=[{"page":q,"text":page_text(path,q)[:5000]} for q in pg[:5]]
        out[v].append({**p,"evidence":ev})
for v,rows in out.items():
    (E/f"pred-{v}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    print(v, "mean items", sum(len(r["evidence"]) for r in rows)/len(rows), "long pages", sum(len(e["text"])>=5000 for r in rows for e in r["evidence"]))
