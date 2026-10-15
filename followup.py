#!/usr/bin/env python3
"""FollowUp Local - paste a customer chat, get the stuck stage, the real concern,
and a next follow-up message, all from a Gemma model running locally via Ollama.

Standard library only. Usage:
  python3 followup.py samples/candle_shop_sample.txt          # CLI
  python3 followup.py --serve                                   # web UI on :8765
"""
import argparse, json, os, sys, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("FOLLOWUP_MODEL", "gemma3:4b")

STAGES = ["new_lead", "asking_questions", "comparing_options", "price_hesitation",
          "waiting_on_someone_else", "ready_to_buy", "gone_quiet", "lost"]

SYSTEM = """You are a sales assistant for a small seller (the SELLER in the chat).
The other person is the CUSTOMER. The customer has not bought yet.
Fill in the JSON:
- stage: which step the customer is stuck at. One of: %s
- stuck_on: the concrete thing blocking the purchase, in this chat's words
- real_concern: the worry behind it (e.g. risk, price vs value, needs approval from someone, timing)
- evidence: copy one CUSTOMER line exactly
- next_message: what the SELLER should send next. Written as the seller, to the customer.
  1-3 sentences. Address the real concern directly using facts already in the chat.
  Only mention prices, products and services the SELLER already stated, word for word. Do not add new ones.
  End with one easy yes/no question. No pressure, no fake urgency, no new discounts or promises.
- send_when: e.g. "now", "tomorrow morning", "in 2 days"
Write stuck_on, real_concern and next_message in the customer's language.""" % ", ".join(STAGES)

EXAMPLE_CHAT = """CUSTOMER: How much for a logo?
SELLER: $300, includes 3 revisions.
CUSTOMER: My partner handles the budget, I'll ask him"""
EXAMPLE_OUT = {"stage": "waiting_on_someone_else",
               "stuck_on": "Her partner has to approve the $300 budget.",
               "real_concern": "She needs something simple to show her partner so the cost looks justified.",
               "evidence": "My partner handles the budget, I'll ask him",
               "next_message": "No rush at all! Would it help if I sent a one-page summary with the price, the 3 revisions and two past logos, so it's easy to show him?",
               "send_when": "tomorrow morning"}

SCHEMA = {"type": "object", "required": ["stage", "stuck_on", "real_concern", "evidence", "next_message", "send_when"],
          "properties": {"stage": {"type": "string", "enum": STAGES}, "stuck_on": {"type": "string"},
                         "real_concern": {"type": "string"}, "evidence": {"type": "string"},
                         "next_message": {"type": "string"}, "send_when": {"type": "string"}}}

TONES = {"warm": "Tone: warm and personal.",
         "brief": "Tone: very brief, one or two short sentences.",
         "direct": "Tone: polite but direct, propose a concrete next step."}


def ollama_chat(messages, model=MODEL, fmt="json"):
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 4096}}
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


def normalise(chat):
    """Drop comment lines, unify full-width colons, keep the last ~60 lines."""
    lines = [l.rstrip().replace("：", ":", 1) for l in chat.strip().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    return "\n".join(lines[-60:])


# Cheap keyword signals. They are only hints for the model, which makes the
# 1B model much more reliable at picking the stage. Easy to extend per niche.
SIGNALS = {
    "waiting_on_someone_else": ["ask my", "check with", "my wife", "my husband", "partner", "fiancé", "boss",
                                "问问", "商量", "爸爸", "妈妈", "老公", "老婆", "领导"],
    "price_hesitation": ["expensive", "cheaper", "too much", "budget", "said $", "discount", "贵", "便宜", "优惠", "预算"],
    "gone_quiet": ["think about it", "get back to you", "let me think", "考虑", "再说", "再看看"],
    "comparing_options": ["other shop", "near my", "elsewhere", "compare", "别家", "对比"],
}


def signals(chat):
    cust = " ".join(l.split(":", 1)[1] for l in normalise(chat).splitlines()
                    if ":" in l and l.split(":", 1)[0].strip().upper() in CUSTOMER_TAGS).lower()
    return [k for k, words in SIGNALS.items() if any(w.lower() in cust for w in words)]


CUSTOMER_TAGS = ("CUSTOMER", "客户", "CLIENT", "BUYER", "顾客")


def analyse(chat, tone="warm", model=MODEL):
    t0 = time.time()
    sig = signals(chat)
    msgs = [{"role": "system", "content": SYSTEM + "\n" + TONES.get(tone, TONES["warm"])},
            {"role": "user", "content": "CHAT:\n" + EXAMPLE_CHAT},
            {"role": "assistant", "content": json.dumps(EXAMPLE_OUT)},
            {"role": "user", "content": "CHAT:\n" + normalise(chat) +
             ("\n\nIMPORTANT: the customer writes in Chinese. Write stuck_on, real_concern and next_message in Simplified Chinese."
              if any("\u4e00" <= ch <= "\u9fff" for ch in chat) else "") +
             ("\n\nHINT - keyword signals found in customer lines: " + ", ".join(sig) if sig else "")}]
    raw = ollama_chat(msgs, model, fmt=SCHEMA)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"next_message": raw}
    if data.get("stage") not in STAGES:
        data["stage_raw"] = data.get("stage")
        data["stage"] = "unclear"
    # small models sometimes paraphrase; fall back to the last real customer line
    cust = [l.split(":", 1)[1].strip() for l in normalise(chat).splitlines()
            if ":" in l and l.split(":", 1)[0].strip().upper() in CUSTOMER_TAGS]
    ev = (data.get("evidence") or "").strip().strip('"')
    if cust and (not ev or ev not in chat):
        data["evidence"] = cust[-1]
    data["signals"] = sig
    data["model"] = model
    data["seconds"] = round(time.time() - t0, 1)
    return data


def render_text(d):
    return (f"Stage        : {d.get('stage')}   (signals: {', '.join(d.get('signals') or []) or 'none'})\n"
            f"Stuck on     : {d.get('stuck_on','')}\n"
            f"Real concern : {d.get('real_concern','')}\n"
            f"Evidence     : \"{d.get('evidence','')}\"\n"
            f"Send when    : {d.get('send_when','')}\n\n"
            f"Next message :\n  {d.get('next_message','')}\n\n"
            f"({d['model']} on this machine, {d['seconds']}s, nothing left your computer)")


PAGE = """<!doctype html><meta charset=utf-8><title>FollowUp Local</title>
<style>body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#222}
textarea{width:100%;height:220px;font:14px monospace}button{padding:.5rem 1rem;font-size:1rem}
.card{background:#f6f7f9;border-radius:8px;padding:1rem;margin-top:1rem;white-space:pre-wrap}
.msg{background:#e7f6ea;border-left:4px solid #2e9e4f}small{color:#666}</style>
<h1>FollowUp Local</h1><p><small>Paste a customer chat. A Gemma model on <b>this</b> computer reads it. Nothing is sent to the cloud.</small></p>
<textarea id=chat placeholder="CUSTOMER: hi, how much is...&#10;SELLER: ..."></textarea>
<p>Tone <select id=tone><option>warm<option>brief<option>direct</select>
<button onclick=go()>Analyse chat</button> <span id=st></span></p><div id=out></div>
<script>
async function go(){st.textContent='Thinking locally...';out.innerHTML='';
const r=await fetch('/api/analyse',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({chat:chat.value,tone:tone.value})});const d=await r.json();
if(d.error){st.textContent='Error: '+d.error;return}
const e=s=>String(s??'').replace(/[&<]/g,c=>({'&':'&amp;','<':'&lt;'}[c]));
out.innerHTML=`<div class=card><b>Stage:</b> ${e(d.stage)}<br><b>Stuck on:</b> ${e(d.stuck_on)}<br><b>Real concern:</b> ${e(d.real_concern)}<br><b>Evidence:</b> "${e(d.evidence)}"<br><b>Send:</b> ${e(d.send_when)}</div>
<div class="card msg"><b>Next message</b>\\n${e(d.next_message)}</div><p><small>${e(d.model)} - ${d.seconds}s - ran on this machine</small></p>`;
st.textContent='';}
</script>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/analyse":
            return self._send(404, "{}", "application/json")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not req.get("chat", "").strip():
                raise ValueError("paste a chat first")
            out = analyse(req["chat"], req.get("tone", "warm"))
        except Exception as e:  # show the error in the UI
            out = {"error": str(e)}
        self._send(200, json.dumps(out, ensure_ascii=False), "application/json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="chat text file (default: stdin)")
    p.add_argument("--tone", default="warm", choices=TONES)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--json", action="store_true", help="print raw JSON")
    p.add_argument("--serve", action="store_true", help="start the web UI")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    if a.serve:
        print(f"FollowUp Local on http://127.0.0.1:{a.port}  (model {MODEL})")
        ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    chat = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
    d = analyse(chat, a.tone, a.model)
    print(json.dumps(d, ensure_ascii=False, indent=2) if a.json else render_text(d))


if __name__ == "__main__":
    main()
