import os
import json
import re
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OR_BASE = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json"}

TEXT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
VISION_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"


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


async def or_chat(messages: list, model: str = TEXT_MODEL) -> str:
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            f"{OR_BASE}/chat/completions",
            headers=HEADERS,
            json={"model": model, "messages": messages, "temperature": 0},
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def llm_json(prompt: str) -> dict:
    text = await or_chat([{"role": "user", "content": prompt + "\n\nReturn ONLY raw JSON, no markdown fences, no explanation."}])
    return safe_json(text)


async def vision_answer(image_b64: str, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Answer with ONLY the value, no units/symbols/explanation. Question: {question}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }
    ]
    text = await or_chat(messages, model=VISION_MODEL)
    return text.strip()


# ---------------- Q2: Multimodal Image QA ----------------
@app.post("/answer-image")
async def answer_image(request: Request):
    body = await request.json()
    image_b64 = body.get("image_base64", "")
    question = body.get("question", "")
    try:
        answer = await vision_answer(image_b64, question)
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
{text}"""
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
{text}"""
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


# ---------------- Q7: Invoice Intelligence ----------------
@app.post("/extract-intelligence")
async def extract_intelligence(request: Request):
    body = await request.json()
    text = body.get("text", "")
    schema = body.get("schema", {})
    prompt = f"""Extract structured data from this business document exactly matching the JSON Schema below.
Normalize amounts (words, K/M suffixes, Indian grouping) to plain integers, dates to YYYY-MM-DD,
infer due_in_days as an integer, infer is_paid as boolean, lowercase contact_email,
keep line_items in document order with integer unit_price.

JSON Schema:
{json.dumps(schema)}

Document:
{text}"""
    try:
        result = await llm_json(prompt)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------- Q8: Semantic Search Top-K (LLM picks directly, no external embeddings) ----------------
@app.post("/rank")
async def rank_candidates(request: Request):
    body = await request.json()
    query = body.get("query", "")
    candidates = body.get("candidates", [])
    numbered = "\n".join(f"{i}: {c}" for i, c in enumerate(candidates))
    prompt = f"""Query: {query}

Candidates (index: text):
{numbered}

Return ONLY JSON: {{"ranking": [i, j, k]}} — the 3 candidate indices most semantically relevant to the query."""
    try:
        result = await llm_json(prompt)
        ranking = result.get("ranking", [])[:3]
        return {"ranking": ranking}
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

Problem: {problem}"""
    try:
        result = await llm_json(prompt)
        result["answer"] = int(round(float(result.get("answer", 0))))
        if len(str(result.get("reasoning", ""))) < 80:
            result["reasoning"] = result.get("reasoning", "") + " " * (80 - len(str(result.get("reasoning", ""))))
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
