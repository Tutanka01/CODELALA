"""Runtime smoke check of generated single-file web apps in headless Firefox.

For every record of a gen.jsonl it extracts the HTML document, loads it, and records:
  extract_ok, js_errors, raf_frames, nonblank, idle_change, input_change, dialogs
`smoke_pass` only means "loads, renders, reacts without throwing" -- it does NOT prove the
game logic is right; judge.py covers that.

Usage: webcheck.py --gen runs/base/gen.jsonl --out runs/base/web.jsonl [--workdir runs/base/html]
"""
import argparse
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageStat
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

FIREFOX = "/snap/firefox/current/usr/lib/firefox/firefox"
GECKO = "/snap/bin/geckodriver"

PROBE = """<script>
window.__errs=[];window.__raf=0;window.__dialogs=0;
window.addEventListener('error',function(e){window.__errs.push(String(e.message||e.type)+' @'+(e.lineno||''));});
window.addEventListener('unhandledrejection',function(e){window.__errs.push('rejection: '+String(e.reason));});
(function(){var ce=console.error;console.error=function(){window.__errs.push('console.error: '+Array.prototype.join.call(arguments,' '));ce.apply(console,arguments);};
var r=window.requestAnimationFrame;window.requestAnimationFrame=function(f){window.__raf++;return r.call(window,f);};
window.alert=function(){window.__dialogs++;};window.confirm=function(){window.__dialogs++;return true;};window.prompt=function(){window.__dialogs++;return '';};})();
</script>"""

FENCE = re.compile(r"```([a-zA-Z0-9]*)\s*\n(.*?)```", re.S)


def extract_html(text: str):
    """Return (html, how). Accepts one full document, or separate html/css/js blocks."""
    blocks = FENCE.findall(text)
    if not blocks:
        m = re.search(r"<!DOCTYPE html.*</html>", text, re.S | re.I) or re.search(r"<html.*</html>", text, re.S | re.I)
        return (m.group(0), "raw") if m else (None, "none")
    docs = [b for lang, b in blocks if re.search(r"<html|<!doctype|<body|<canvas|<script", b, re.I)]
    if not docs:
        return None, "none"
    html = max(docs, key=len)
    css = "\n".join(b for lang, b in blocks if lang.lower() == "css")
    js = "\n".join(b for lang, b in blocks if lang.lower() in ("js", "javascript"))
    how = "single"
    if css and "<style" not in html.lower():
        html = html.replace("</head>", f"<style>{css}</style></head>") if "</head>" in html else f"<style>{css}</style>" + html
        how = "assembled"
    if js and not re.search(r"<script(?![^>]*src=)[^>]*>\s*\S", html, re.I):
        html = html.replace("</body>", f"<script>{js}</script></body>") if "</body>" in html else html + f"<script>{js}</script>"
        how = "assembled"
    return html, how


def instrument(html: str) -> str:
    m = re.search(r"<head[^>]*>", html, re.I)
    if m:
        return html[: m.end()] + PROBE + html[m.end():]
    return PROBE + html


def new_driver():
    o = Options()
    o.add_argument("-headless")
    o.add_argument("--width=900")
    o.add_argument("--height=750")
    o.binary_location = FIREFOX
    d = webdriver.Firefox(options=o, service=Service(GECKO))
    d.set_page_load_timeout(20)
    d.set_script_timeout(10)
    return d


def shot(d):
    return Image.open(io.BytesIO(d.get_screenshot_as_png())).convert("L").resize((160, 120))


def diff(a, b):
    return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]


def probe(d, path: Path):
    res = {}
    d.get(path.as_uri())
    time.sleep(1.2)
    s0 = shot(d)
    time.sleep(1.5)
    s1 = shot(d)
    res["nonblank"] = ImageStat.Stat(s1).stddev[0] > 1.0
    res["idle_change"] = round(diff(s0, s1), 3)
    body = d.find_element(By.TAG_NAME, "body")
    targets = d.find_elements(By.CSS_SELECTOR, "canvas, button, [onclick], .card, .cell, td, input")[:12]
    for el in targets[:6]:
        try:
            ActionChains(d).move_to_element(el).click().perform()
        except Exception:
            pass
    for _ in range(6):
        for k in (Keys.SPACE, Keys.ARROW_UP, Keys.ARROW_RIGHT, Keys.ARROW_DOWN, Keys.ARROW_LEFT, Keys.ENTER):
            try:
                ActionChains(d).send_keys(k).perform()
            except Exception:
                pass
        time.sleep(0.25)
    for el in targets[6:]:
        try:
            ActionChains(d).move_to_element(el).click().perform()
        except Exception:
            pass
    time.sleep(0.5)
    s2 = shot(d)
    res["input_change"] = round(diff(s1, s2), 3)
    res["js_errors"] = d.execute_script("return window.__errs||[]")[:8]
    res["raf_frames"] = d.execute_script("return window.__raf||0")
    res["dialogs"] = d.execute_script("return window.__dialogs||0")
    res["n_canvas"] = len(d.find_elements(By.TAG_NAME, "canvas"))
    res["visible_text"] = (body.text or "")[:300]
    return res


def probe_isolated(path: Path, timeout: int = 90):
    """Probe one page in its own process: a page that freezes the browser (e.g. an endless
    loop in a key handler) is killed after `timeout` seconds instead of stalling the run."""
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--single", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True, text=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return json.loads(out.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-KILL", "-s", str(proc.pid)], check=False)
        proc.wait()
        return {"probe_error": "timeout: the page froze the browser"}
    except Exception as e:
        subprocess.run(["pkill", "-KILL", "-s", str(proc.pid)], check=False)
        return {"probe_error": repr(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen")
    ap.add_argument("--out")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--single", default=None, help="internal: probe one instrumented HTML file and print JSON")
    args = ap.parse_args()
    if args.single:
        d = new_driver()
        try:
            res = probe(d, Path(args.single))
        except Exception as e:
            res = {"probe_error": repr(e)[:200]}
        finally:
            try:
                d.quit()
            except Exception:
                pass
        print(json.dumps(res, ensure_ascii=False))
        return
    gen = [json.loads(l) for l in Path(args.gen).read_text().splitlines() if l.strip()]
    work = Path(args.workdir or Path(args.out).parent / "html").resolve()
    work.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fout:
        for r in gen:
            out = {"id": r["id"], "sample": r["sample"], "finish": r.get("finish"), "completion_tokens": r.get("completion_tokens")}
            html, how = extract_html(r.get("content") or "")
            out["extract"] = how
            out["extract_ok"] = html is not None
            if html is not None:
                p = work / f"{r['id']}_{r['sample']}.html"
                p.write_text(instrument(html))
                (work / f"{r['id']}_{r['sample']}.orig.html").write_text(html)
                out.update(probe_isolated(p))
            out["smoke_pass"] = bool(
                out.get("extract_ok")
                and not out.get("probe_error")
                and not out.get("js_errors")
                and out.get("nonblank")
                and (out.get("idle_change", 0) > 0.05 or out.get("input_change", 0) > 0.05)
            )
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            print(out["id"], out["sample"], "PASS" if out["smoke_pass"] else "FAIL", out.get("js_errors") or out.get("probe_error") or "", flush=True)


if __name__ == "__main__":
    main()
