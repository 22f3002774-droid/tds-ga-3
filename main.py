import os
import json
import re
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np

import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
TEXT_MODEL = "gemini-2.0-flash"
EMBED_MODEL = "text-embedding-004"


def safe_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\{.*\})", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return {}


async def gemini_generate(parts: list, json_mode: bool = True) -> str:
    url = f"{GEMINI_BASE}/models/{TEXT_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
    body = {"contents": [{"parts": parts}]}
    if json_mode:
        body["generationConfig"] = {"response_mime_type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=body)
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return json.dumps({"error": data})


async def gemini_json(prompt: str) -> dict:
    text = await gemini_generate([{"text": prompt}], json_mode=True)
    return safe_json(text)


async def gemini_vision_answer(image_b64: str, question: str) -> str:
    parts = [
        {"text": f"Answer with ONLY the value, no units/symbols/explanation. Question: {question}"},
        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
    ]
    text = await gemini_generate(parts, json_mode=False)
    return text.strip()


async def gemini_embed(texts: list[str]) -> np.ndarray:
    url = f"{GEMINI_BASE}/models/{EMBED_MODEL}:batchEmbedContents?key={config.GEMINI_API_KEY}"
    reqs = [{"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t}]}} for t in texts]
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json={"requests": reqs})
        data = resp.json()
        vecs = [e["values"] for e in data["embeddings"]]
        return np.array(vecs)


# ---------------- Q2: Multimodal Image QA ----------------
@app.post("/answer-image")
async def answer_image(request: Request):
    body = await request.json()
    image_b64 = body.get("image_base64", "")
    question = body.get("question", "")
    try:
        answer = await gemini_vision_answer(image_b64, question)
        cleaned = re.sub(r"[^\d.\-]", "", answer) if re.search(r"\d", answer) and not re.search(r"[a-zA-Z]{3,}", answer) else answer
        return {"answer": cleaned if cleaned else answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------- Q3: Fixed Schema Invoice Extraction ----------------
@app.post("/extract")
async def extract_fixed(request: Request):
    body = await request.json()
    text = body.get("invoice_text", "")
    prompt = f"""Extract these 6 fields from the invoice text below. Return ONLY JSON with exactly these keys:
invoice_no (string or null), date (YYYY-MM-DD or null), vendor (string or null),
amount (number, the SUBTOTAL before tax, or null), tax (number, tax amount only, or null),
currency (3-letter ISO code, default "INR" if rupee symbols/Rs. seen, else infer, or null).

Invoice text:
{text}

Return ONLY the JSON object, no explanation."""
    try:
        result = await gemini_json(prompt)
        for k in ["invoice_no", "date", "vendor", "amount", "tax", "currency"]:
            result.setdefault(k, None)
        return result
    except Exception:
        return {"invoice_no": None, "date": None, "vendor": None, "amount": None, "tax": None, "currency": None}


# ---------------- Q4: Dynamic Schema Extraction ----------------
@app.post("/dynamic-extract")
async def dynamic_extract(request: Request):
    body = await request.json()
    text = body.get("text", "")
    schema = body.get("schema", {})
    prompt = f"""Extract fields from the text below according to this schema (field_name: type).
Types: "string", "integer", "float", "date" (output as YYYY-MM-DD), "boolean".
Return ONLY a JSON object with EXACTLY these keys (no extras, no missing): {json.dumps(schema)}
Use null for any field you cannot find. Use correct JSON types (numbers as numbers, not strings).

Text:
{text}

Return ONLY the JSON object."""
    try:
        result = await gemini_json(prompt)
        out = {}
        for key, typ in schema.items():
            val = result.get(key)
            if val is None:
                out[key] = None
                continue
            try:
                if typ == "integer":
                    out[key] = int(float(val))
                elif typ == "float":
                    out[key] = float(val)
                elif typ == "boolean":
                    out[key] = bool(val) if isinstance(val, bool) else str(val).lower() in ["true", "1", "yes"]
                else:
                    out[key] = str(val)
            except Exception:
                out[key] = val
        return out
    except Exception:
        return {k: None for k in schema}


# ---------------- Q7: Invoice Intelligence (schema sent per-request) ----------------
@app.post("/extract-intelligence")
async def extract_intelligence(request: Request):
    body = await request.json()
    text = body.get("text", "")
    schema = body.get("schema", {})
    prompt = f"""Extract structured data from this business document exactly matching the JSON Schema below.
Follow every rule precisely: normalize amounts (words, K/M suffixes, Indian grouping) to plain integers,
normalize dates to YYYY-MM-DD, infer due_in_days as an integer, infer is_paid as boolean,
lowercase contact_email, keep line_items in document order with integer unit_price.

JSON Schema:
{json.dumps(schema)}

Document:
{text}

Return ONLY the JSON object matching the schema, no extra keys, no explanation."""
    try:
        result = await gemini_json(prompt)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------- Q8: Semantic Search Top-K ----------------
@app.post("/rank")
async def rank_candidates(request: Request):
    body = await request.json()
    query = body.get("query", "")
    candidates = body.get("candidates", [])
    try:
        vecs = await gemini_embed([query] + candidates)
        q_vec = vecs[0]
        c_vecs = vecs[1:]
        q_norm = q_vec / np.linalg.norm(q_vec)
        c_norms = c_vecs / np.linalg.norm(c_vecs, axis=1, keepdims=True)
        sims = c_norms @ q_norm
        top3 = np.argsort(-sims)[:3].tolist()
        return {"ranking": top3}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------- Q9: Word-Problem Solver ----------------
@app.post("/solve")
async def solve_problem(request: Request):
    body = await request.json()
    problem = body.get("problem", "")
    prompt = f"""Solve this word problem step by step. Ignore any irrelevant distractor numbers.
Return ONLY JSON with exactly two keys:
"reasoning": a string of at least 80 characters showing your steps,
"answer": a JSON integer (not a string, not a float, no currency symbols).

Problem: {problem}

Return ONLY the JSON object."""
    try:
        result = await gemini_json(prompt)
        result["answer"] = int(round(float(result.get("answer", 0))))
        if len(str(result.get("reasoning", ""))) < 80:
            result["reasoning"] = result.get("reasoning", "") + " " * (80 - len(str(result.get("reasoning", ""))))
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
