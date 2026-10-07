#!/usr/bin/env python3
"""openRP-API.py port for KoboldCpp - queued chat API with threaded workers.

Endpoints (same contract as openRP-API.py):

  POST /chat      body: {"chat_completion": [{"role": "...", "content": "..."}, ...]}
                  -> {"status": "queued", "request_id": "...", "messages": N}
                  errors -> {"status": "error", "message": "..."}

  POST /response  body: {"request_id": "..."}
                  -> {"status": "processing", "request_id": "..."}
                  -> {"status": "completed", "request_id": "...", "response": "..."}
                  -> {"status": "failed", "request_id": "...", "error": "..."}

Nothing blocks on the way in: jobs are pushed onto a queue and daemon worker
threads drain it and forward the messages to the local KoboldCpp server
(its /v1/chat/completions endpoint), like the cloud OpenAI client in
openRP-API.py, just pointed at a local model host. KoboldCpp keeps serving
its usual UI/API; this queue lives on top of it.
"""

import os
import time
import uuid
import json
import threading

import requests
from fastapi import FastAPI, Request

KCPP_HOST = os.getenv("KCPP_HOST", "http://127.0.0.1:5001").rstrip("/")
GEN_TIMEOUT = int(os.getenv("BRIDGE_GEN_TIMEOUT", "3600"))
KCPP_WAIT = int(os.getenv("BRIDGE_KCPP_WAIT", "10800"))
WORKERS = int(os.getenv("BRIDGE_WORKERS", "2"))

# Extra options a /chat request may carry besides the messages (optional).
PASSTHROUGH_KEYS = ("temperature", "top_p", "top_k", "min_p", "tfs", "typical",
                    "max_tokens", "stop", "seed", "presence_penalty",
                    "frequency_penalty", "repetition_penalty")

app = FastAPI()

chat_queue = []
chat_data = {}
queue_lock = threading.Lock()

def banner(title, *lines):
    print("")
    print("=" * 40)
    print(title)
    for line in lines:
        print(line)
    print("=" * 40)

@app.get("/")
async def index():
    return {"status": "ok", "service": "openrp-bridge", "workers": WORKERS}

@app.post("/chat")
async def chat(req: Request):
    # --------------------------------------------------------
    # Receive body
    # --------------------------------------------------------
    raw = (await req.body()).decode("utf-8", "replace")

    # --------------------------------------------------------
    # Parse body
    # --------------------------------------------------------
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": "Invalid JSON: %s" % e}

    if not isinstance(body, dict):
        return {"status": "error", "message": "Body must be a JSON object"}

    # --------------------------------------------------------
    # Get chat completion
    # --------------------------------------------------------
    chat_completion = body.get("chat_completion")

    if chat_completion is None:
        return {"status": "error", "message": "Missing chat_completion"}

    # --------------------------------------------------------
    # Make sure it is an array
    # --------------------------------------------------------
    if not isinstance(chat_completion, list):
        return {"status": "error", "message": "chat_completion is not an array"}

    if not chat_completion:
        return {"status": "error", "message": "chat_completion is empty"}

    # Any sampling options the request may carry ride along.
    extras = {k: body[k] for k in PASSTHROUGH_KEYS if k in body}

    # --------------------------------------------------------
    # Create request ID
    # --------------------------------------------------------
    request_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # Store request
    # --------------------------------------------------------
    chat_data[request_id] = {
        "request_id": request_id,
        "status": "queued",
        "messages": chat_completion,
        "extras": extras,
        "response": None,
        "error": None,
    }

    # --------------------------------------------------------
    # Add to queue
    # --------------------------------------------------------
    with queue_lock:
        chat_queue.append(request_id)

    banner("New chat request",
           "Request ID: %s" % request_id,
           "Messages: %d" % len(chat_completion),
           "Queue: %d" % len(chat_queue))

    # --------------------------------------------------------
    # Return immediately
    # --------------------------------------------------------
    return {"status": "queued", "request_id": request_id, "messages": len(chat_completion)}

@app.post("/response")
async def response(req: Request):
    raw = (await req.body()).decode("utf-8", "replace")

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": "Invalid JSON: %s" % e}

    if not isinstance(body, dict):
        return {"status": "error", "message": "Body must be a JSON object"}

    request_id = body.get("request_id")

    if not request_id or request_id not in chat_data:
        return {"status": "not_found"}

    data = chat_data[request_id]

    # --------------------------------------------------------
    # Queued / processing
    # --------------------------------------------------------
    if data["status"] in ("queued", "processing"):
        return {"status": "processing", "request_id": request_id}

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------
    if data["status"] == "completed":
        return {"status": "completed", "request_id": request_id, "response": data["response"]}

    # --------------------------------------------------------
    # Failed
    # --------------------------------------------------------
    return {"status": "failed", "request_id": request_id, "error": data["error"]}

def wait_for_koboldcpp():
    # Be patient: the KoboldCpp cell may still be downloading and loading the
    # model when the first requests arrive.
    polled = False
    deadline = time.time() + KCPP_WAIT
    while time.time() < deadline:
        try:
            r = requests.get(KCPP_HOST + "/api/extra/version", timeout=10)
            if r.ok and r.json().get("llm"):
                if polled:
                    print("KoboldCpp is online, releasing the queue.")
                return True
        except Exception:
            pass
        polled = True
        time.sleep(2)
    return False

def generate_chat(messages, extras):
    if not wait_for_koboldcpp():
        raise RuntimeError("KoboldCpp at %s never came online (waited %d s)" % (KCPP_HOST, KCPP_WAIT))

    payload = {"model": "koboldcpp", "messages": messages}
    payload.update(extras)

    r = requests.post(KCPP_HOST + "/v1/chat/completions", json=payload, timeout=GEN_TIMEOUT)
    r.raise_for_status()

    try:
        content = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("KoboldCpp sent a malformed response: %s" % str(r.text)[:500])
    return content

def prune_chat_data():
    # Drop the oldest finished jobs so long-running bridges stay slim,
    # but never touch ones that are queued or still processing.
    if len(chat_data) <= 200:
        return
    finished = [k for k, v in chat_data.items() if v["status"] in ("completed", "failed")]
    for request_id in finished[:len(chat_data) - 200]:
        chat_data.pop(request_id, None)

def chat_worker():
    print("Chat worker started")
    while True:
        with queue_lock:
            request_id = chat_queue.pop(0) if chat_queue else None

        if not request_id:
            time.sleep(0.5)
            continue

        data = chat_data[request_id]

        data["status"] = "processing"

        banner("Processing chat request", "Request ID: %s" % request_id)

        try:
            start = time.time()
            result = generate_chat(data["messages"], data["extras"])

            data["response"] = result
            data["status"] = "completed"

            banner("Chat completed: %s" % request_id,
                   "Took %.1f s, queue: %d" % (time.time() - start, len(chat_queue)),
                   "Response:", result)

            prune_chat_data()
        except Exception as e:
            banner("Chat failed: %s" % request_id, repr(e))

            data["status"] = "failed"
            data["error"] = str(e) or repr(e)

            prune_chat_data()

for i in range(WORKERS):
    threading.Thread(target=chat_worker, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("BRIDGE_PORT", "5002")))
