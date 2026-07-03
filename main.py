import os
import json
import re
import base64
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

OPENAI_BASE = "https://api.openai.com/v1"
HEADERS = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}


def safe_json(s: str) -> dict:
    """Extract a JSON object from a raw LLM string, tolerant of markdown fences."""
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


async def llm_json(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """Call an OpenAI-compatible chat completion and force JSON output."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OPENAI_BASE}/chat/completions",
            headers=HEADERS,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return safe_json(content)


async def llm_vision_json(image_b64: str, question: str, model: str = "gpt-4o-mini") -> str:
    """Call a vision-capable model with a base64 image + question, return plain text answer."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OPENAI_BASE}/chat/completions",
            headers=HEADERS,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Answer with ONLY the value, no units/symbols/explanation. Question: {question}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
                "temperature": 0,
            },
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OPENAI_BASE}/embeddings",
            headers=HEADERS,
            json={"model": model, "input": texts},
        )
        data = resp.json()
        vecs = [d["embedding"] for d in data["data"]]
        return np.array(vecs)


# ---------------- Q2: Multimodal Image QA ----------------
@app.post("/answer-image")
async def answer_image(request: Request):
    body = await request.json()
    image_b64 = body.get("image_base64", "")
    question = body.get("question", "")
    try:
        answer = await llm_vision_json(image_b64, question)
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
        result = await llm_json(prompt)
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
        result = await llm_json(prompt)
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
        result = await llm_json(prompt, model="gpt-4o")
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
        vecs = await embed_texts([query] + candidates)
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
        result = await llm_json(prompt, model="gpt-4o")
        result["answer"] = int(round(float(result.get("answer", 0))))
        if len(str(result.get("reasoning", ""))) < 80:
            result["reasoning"] = result.get("reasoning", "") + " " * (80 - len(str(result.get("reasoning", ""))))
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
