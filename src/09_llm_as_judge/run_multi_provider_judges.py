#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Provider SDKs
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types as genai_types

# Tokenizer fallback for approximate counts
import tiktoken


# =========================================================
# 🔷 GLOBALS
# =========================================================
ENCODING = tiktoken.get_encoding("cl100k_base")


# =========================================================
# 🔷 HELPERS
# =========================================================
def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except:
        return None

def make_serializable(obj):
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Fallback — convert unknown objects to string representation
        try:
            return obj.model_dump()   # pydantic objects (OpenAI SDK)
        except AttributeError:
            return str(obj)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def approx_token_count(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        return len(ENCODING.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def load_processed_ids(output_file: Path) -> set:
    processed = set()
    if not output_file.exists():
        return processed
    for row in load_jsonl(output_file):
        cid = row.get("custom_id")
        if cid:
            processed.add(cid)
    print(f"🔁 Resume loaded: {len(processed)} completed requests from {output_file}")
    return processed


def model_tag(model_name: str) -> str:
    return model_name.lower().replace("/", "__").replace("-", "_").replace(":", "_")


def extract_openai_usage(resp: Any, prompt: str, output_text: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Try provider-reported usage first, fallback to approximation.
    """
    usage = getattr(resp, "usage", None)
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)
        if in_tok is not None or out_tok is not None:
            return in_tok, out_tok, "provider_reported_or_partial"
    return approx_token_count(prompt), approx_token_count(output_text), "tiktoken_cl100k_fallback"


def extract_anthropic_usage(resp: Any, prompt: str, output_text: str) -> Tuple[Optional[int], Optional[int], str]:
    usage = getattr(resp, "usage", None)
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)
        if in_tok is not None or out_tok is not None:
            return in_tok, out_tok, "provider_reported_or_partial"
    return approx_token_count(prompt), approx_token_count(output_text), "tiktoken_cl100k_fallback"


def extract_gemini_usage(resp: Any, prompt: str, output_text: str) -> Tuple[Optional[int], Optional[int], str]:
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        in_tok = getattr(usage, "prompt_token_count", None)
        out_tok = getattr(usage, "candidates_token_count", None)
        if in_tok is not None or out_tok is not None:
            return in_tok, out_tok, "provider_reported_or_partial"
    return approx_token_count(prompt), approx_token_count(output_text), "tiktoken_cl100k_fallback"


# =========================================================
# 🔷 PROVIDER CLIENTS
# =========================================================
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    print("api_key: ", api_key)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=api_key)


def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
    return genai.Client(api_key=api_key)


# =========================================================
# 🔷 GENERATION FUNCTIONS
# =========================================================
def generate_openai_judge(request_obj: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    body = request_obj["body"]

    start = time.time()

    resp = client.responses.create(
        model=body["model"],
        instructions=body["instructions"],
        input=body["input"],
        max_output_tokens=body["max_output_tokens"],
        text=body["text"]
    )

    latency = time.time() - start

    # Extract text
    output_text = getattr(resp, "output_text", None)
    if not output_text:
        try:
            output_text = resp.output[0].content[0].text
        except Exception:
            output_text = ""

    # Parse JSON
    try:
        parsed = json.loads(output_text)
    except Exception as e:
        parsed = {"parse_error": str(e), "raw_output": output_text}

    # Token usage (if available)
    try:
        usage = resp.usage
        in_tok = usage.input_tokens
        out_tok = usage.output_tokens
        token_method = "provider_reported"
    except Exception:
        in_tok, out_tok, token_method = None, None, None

    total_tok = (in_tok + out_tok) if in_tok and out_tok else None

    # Finish reason
    finish_reason = None
    try:
        finish_reason = resp.output[0].status
    except Exception:
        pass

    return {
        "custom_id": request_obj["custom_id"],
        "provider": "openai",
        "model": body["model"],

        "response": parsed,
        "response_raw": make_serializable(resp),

        "provider_response_id": getattr(resp, "id", None),

        "latency": latency,
        "finish_reason": finish_reason,

        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": total_tok,
        "token_count_method": token_method,

        "error": None
    }

def generate_anthropic_judge(request_obj: Dict[str, Any], client: anthropic.Anthropic) -> Dict[str, Any]:
    start = time.time()

    resp = client.messages.create(
        model=request_obj["model"],
        system=request_obj["system"],
        max_tokens=request_obj["max_tokens"],
        temperature=request_obj["temperature"],
        # top_p=request_obj.get("top_p", 1),
        messages=request_obj["messages"]
    )

    latency = time.time() - start

    # Extract text
    chunks = []
    for block in getattr(resp, "content", []):
        txt = getattr(block, "text", None)
        if txt:
            chunks.append(txt)
    output_text = "".join(chunks)

    # Parse JSON
    raw = output_text
    parsed = extract_json(raw)

    if parsed is None:
        judge_response = {
            "parse_error": "json_extraction_failed",
            "raw_output": raw
        }
    else:
        judge_response = parsed

    # Token usage
    try:
        usage = resp.usage
        in_tok = usage.input_tokens
        out_tok = usage.output_tokens
        token_method = "provider_reported"
    except Exception:
        in_tok, out_tok, token_method = None, None, None

    total_tok = (in_tok + out_tok) if in_tok and out_tok else None

    try:
        print("cache_hit", getattr(usage, "cache_read_input_tokens", 0) > 0)
        print("cache_write", getattr(usage, "cache_creation_input_tokens", 0) > 0)    
    except Exception:        
        print("cache not found")
    
    return {
        "custom_id": request_obj["custom_id"],
        "provider": "anthropic",
        "model": request_obj["model"],

        "response": judge_response,
        "response_raw": make_serializable(resp),

        "provider_response_id": getattr(resp, "id", None),

        "latency": latency,
        "finish_reason": getattr(resp, "stop_reason", None),

        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": total_tok,
        "token_count_method": token_method,

        "error": None
    }


def generate_gemini_judge(request_obj: Dict[str, Any], client: genai.Client) -> Dict[str, Any]:
    from google.genai import types as genai_types

    prompt = request_obj["prompt"]
    system_instruction = request_obj["system_instruction"]

    temperature = request_obj.get("temperature", 0.0)
    max_tokens = request_obj.get("max_tokens", 512)
    top_p = request_obj.get("top_p", 1.0)

    start = time.time()

    resp = client.models.generate_content(
        model=request_obj["model"],
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
        ),
    )

    latency = time.time() - start

    output_text = getattr(resp, "text", "") or ""

    # Parse JSON
    try:
        parsed = json.loads(output_text)
    except Exception as e:
        parsed = {"parse_error": str(e), "raw_output": output_text}

    # Token usage
    try:
        in_tok, out_tok, token_method = extract_gemini_usage(resp, prompt, output_text)
    except Exception:
        in_tok, out_tok, token_method = None, None, None

    total_tok = (in_tok + out_tok) if in_tok and out_tok else None

    # Finish reason
    finish_reason = None
    try:
        cands = getattr(resp, "candidates", None)
        if cands:
            finish_reason = getattr(cands[0], "finish_reason", None)
    except Exception:
        pass

    return {
        "custom_id": request_obj["custom_id"],
        "provider": "gemini",
        "model": request_obj["model"],

        "response": parsed,
        "response_raw": make_serializable(resp),

        "provider_response_id": None,

        "latency": latency,
        "finish_reason": finish_reason,

        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": total_tok,
        "token_count_method": token_method,

        "error": None
    }

def route_generate(request_obj: Dict[str, Any], clients: Dict[str, Any]) -> Dict[str, Any]:
    provider = request_obj["provider"]

    try:
        if provider == "openai":
            return generate_openai_judge(request_obj, clients["openai"])
        elif provider == "anthropic":
            return generate_anthropic_judge(request_obj, clients["anthropic"])
        elif provider == "gemini":
            return generate_gemini_judge(request_obj, clients["gemini"])  # unchanged
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        return {
            "custom_id": request_obj.get("custom_id"),
            "response": None,
            "error": str(e)
        }
# =========================================================
# 🔷 BATCH RUNNER
# =========================================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def append_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================
# 🔥 CORE PROCESSING FUNCTION
# =========================================================
def process_requests(input_file, output_file, clients, flush_every=25):
    requests = load_jsonl(input_file)

    buffer = []
    total = len(requests)

    print(f"🚀 Processing {total} requests...")

    for i, req in enumerate(requests):

        provider = req.get("provider")

        try:
            # 🔥 ROUTING
            if provider == "openai":
                judge_result = generate_openai_judge(req, clients["openai"])

            elif provider == "anthropic":
                judge_result = generate_anthropic_judge(req, clients["anthropic"])

            elif provider == "gemini":
                judge_result = generate_gemini_judge(req, clients["gemini"])

            else:
                raise ValueError(f"Unknown provider: {provider}")

        except Exception as e:
            judge_result = {
                "custom_id": req.get("custom_id"),
                "provider": provider,
                "model": req.get("model"),
                "response": None,
                "error": str(e)
            }

        # ✅ Attach judge result
        req["judge"] = judge_result

        buffer.append(req)

        # 🔥 Flush every N
        if len(buffer) >= flush_every:
            append_jsonl(buffer, output_file)
            print(f"💾 Saved {i+1}/{total}")
            buffer = []
            print("buffer: ", buffer)

        # 🔸 small sleep (rate limit safety)
        if i % 10 == 0:
            time.sleep(0.2)

    # 🔥 Final flush
    if buffer:
        append_jsonl(buffer, output_file)
        print(f"✅ Final save: {total}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-dir", required=True, help="Directory containing provider/model request JSONL files")
    parser.add_argument("--output-dir", required=True, help="Directory to write provider outputs")
    parser.add_argument("--provider", required=True, help="Provider to use")
    parser.add_argument("--judge-model", required=True, help="Judge model to use")
    args = parser.parse_args()

    request_dir = Path(args.request_dir) / args.provider / model_tag(args.judge_model)
    output_dir = Path(args.output_dir) / args.provider / model_tag(args.judge_model)
    ensure_dir(output_dir)

    request_files = sorted(request_dir.glob("*.jsonl"))
    print("request_files.len: ", len(request_files))

    if not request_files:
        raise FileNotFoundError(f"No request files found in {request_dir}")

    provider = None
    if len(request_files) > 0:
        # infer provider from first line of first file
        first_row = load_jsonl(request_files[0])[0]
        provider = first_row["provider"]

    clients: Dict[str, Any] = {}
    if provider == "openai":
        print("if provider == openai:")
        clients["openai"] = get_openai_client()
    elif provider == "anthropic":
        print("if provider == anthropic:")
        clients["anthropic"] = get_anthropic_client()
    elif provider == "gemini":
        print("if provider == gemini:")
        clients["gemini"] = get_gemini_client()
    else:
        raise ValueError(f"Could not infer supported provider from request files in {request_dir}")

    for request_file in request_files:
        print("processing file: ", request_file)
        out_file = output_dir / request_file.name
        process_requests(request_file, out_file, clients)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import json, os, argparse
# from pathlib import Path
# from openai import OpenAI
# import anthropic
# from google import genai

# def load_jsonl(p):
#     return [json.loads(x) for x in open(p, encoding="utf-8")]

# def save_jsonl(rows, p):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     with open(p, "a", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

# # ---------------- OPENAI (BATCH) ----------------
# def run_openai(file, out):
#     client = OpenAI()
#     batch_file = client.files.create(file=open(file, "rb"), purpose="batch")

#     batch = client.batches.create(
#         input_file_id=batch_file.id,
#         endpoint="/v1/responses",
#         completion_window="24h",
#     )

#     print("Batch:", batch.id)

# # ---------------- CLAUDE ----------------
# def run_claude(file, out):
#     client = anthropic.Anthropic()

#     rows = load_jsonl(file)
#     out_rows = []

#     for r in rows:
#         try:
#             resp = client.messages.create(
#                 model=r["model"],
#                 system=r["system"],
#                 max_tokens=r["max_tokens"],
#                 temperature=r["temperature"],
#                 messages=r["messages"]
#             )

#             text = "".join([c.text for c in resp.content if hasattr(c, "text")])

#             out_rows.append({
#                 "custom_id": r["custom_id"],
#                 "response": text,
#                 "error": None
#             })

#         except Exception as e:
#             out_rows.append({
#                 "custom_id": r["custom_id"],
#                 "response": None,
#                 "error": str(e)
#             })

#     save_jsonl(out_rows, out)

# # ---------------- GEMINI ----------------
# def run_gemini(file, out):
#     client = genai.Client()

#     rows = load_jsonl(file)
#     out_rows = []

#     for r in rows:
#         try:
#             resp = client.models.generate_content(
#                 model=r["model"],
#                 contents=r["prompt"],
#             )

#             out_rows.append({
#                 "custom_id": r["custom_id"],
#                 "response": resp.text,
#                 "error": None
#             })

#         except Exception as e:
#             out_rows.append({
#                 "custom_id": r["custom_id"],
#                 "response": None,
#                 "error": str(e)
#             })

#     save_jsonl(out_rows, out)

# # ---------------- MAIN ----------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--file", required=True)
#     parser.add_argument("--provider", required=True)
#     parser.add_argument("--out", required=True)
#     args = parser.parse_args()


#     if args.provider == "openai":
#         run_openai(args.file, args.out)
#     elif args.provider == "anthropic":
#         run_claude(args.file, args.out)
#     elif args.provider == "gemini":
#         run_gemini(args.file, args.out)

# if __name__ == "__main__":
#     main()



