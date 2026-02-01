#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-Judge with key-point aware evaluation.

Summary:
- Uses an LLM (Ollama/OpenAI/vLLM) to judge generated answers and evaluate
  coverage/accuracy of provided key points.
- Pass/fail mirrors the LLM verdict (CORRECT passes). If LLM fails, a
  deterministic fallback labels verdict using key-point matches only.

Inputs:
- --input_json: A JSON file (or glob/dir of JSONs) where each object has `qid`,
  `question`, `generated_answer`, and `key_points`.
- Optional: --corpus to enable precise 424B2 exclusion via doc titles/metadata
- LLM backend args: --judge_backend, --openai_api_key, --openai_model,
  --vllm_model_path, --vllm_gpu_ids, --vllm_tensor_parallel_size,
  --ollama_port, --ollama_host

Outputs:
- results.json: per-QA evaluation (LLM verdict, key-point coverage, pass/fail)
- details.csv: row-by-row breakdown for spreadsheet review
- summary.json: overall + by QA type
"""

import argparse
import csv
import json
import math
import os
import re
import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
from tqdm import tqdm

NUM_RE = re.compile(r"(\d[\d,]*\.?\d*)")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Silence noisy per-request HTTP logs from the OpenAI SDK's httpx transport.
# (e.g., "INFO:httpx:HTTP Request: POST ... 200 OK")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Optional imports are done lazily when the corresponding backend is used ---

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_answers(answers_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load answers from a single JSON file, a directory of JSON files, or a glob pattern.

    Expect rows shaped like {qid, question, generated_answer, doc_ids_used}.
    Returns a dict keyed by qid. Later files can overwrite earlier qids.
    """
    paths: List[str]
    if any(ch in answers_path for ch in ("*", "?", "[", "]")):
        paths = sorted(glob.glob(answers_path))
    elif os.path.isdir(answers_path):
        paths = sorted(glob.glob(os.path.join(answers_path, "*.json")))
    else:
        paths = [answers_path]

    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        try:
            data = load_json(p)
            if not isinstance(data, list):
                logger.warning("answers file is not a list: %s", p)
                continue
            for row in data:
                qid = str(row.get("qid", ""))
                if not qid:
                    continue
                out[qid] = row
        except Exception as e:
            logger.error("Failed to load %s: %s", p, e)
    return out

def unwrap_qa_list(qa_data) -> List[Dict[str, Any]]:
    if isinstance(qa_data, list):
        return qa_data
    if isinstance(qa_data, dict):
        for k in ("data","items","qa","examples"):
            if k in qa_data and isinstance(qa_data[k], list):
                return qa_data[k]
        # fallback: merge lists in dict values
        merged=[]
        for v in qa_data.values():
            if isinstance(v, list):
                merged += v
        if merged:
            return merged
    return []

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    # normalize commas in numbers: 10,000 -> 10000
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    # normalize percents: "20 percent" -> "20%"
    s = re.sub(r"(\d+(?:\.\d+)?)\s*percent", r"\1%", s)
    return s.strip()

def extract_keypoints(ex: Dict[str, Any]) -> List[str]:
    """
    Try multiple shapes:
      - ex["key_points"] = List[str]
      - ex["key_points"] = List[{"text": "..."}] or {"point": "..."} or {"kp": "..."}
      - ex["key_points"]["items"] = [...]
    """
    kps = ex.get("key_points")
    if not kps:
        return []
    if isinstance(kps, list):
        out=[]
        for item in kps:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("text","point","kp","value","content"):
                    if key in item and isinstance(item[key], str):
                        out.append(item[key])
                        break
        return [kp for kp in out if kp and kp.strip()]
    if isinstance(kps, dict):
        arr = None
        for key in ("items","points","list"):
            if key in kps and isinstance(kps[key], list):
                arr = kps[key]; break
        if arr is None:
            return []
        out=[]
        for item in arr:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("text","point","kp","value","content"):
                    if key in item and isinstance(item[key], str):
                        out.append(item[key])
                        break
        return [kp for kp in out if kp and kp.strip()]
    return []

def qa_type_from_qid(qid: str) -> str:
    if not qid:
        return "unknown"
    if qid.startswith("qa_1_"):
        return "chunk_based_qa"
    if qid.startswith("qa_2_"):
        return "tracking_qa"
    if qid.startswith("qa_3_"):
        return "company_comparison_qa"
    return "unknown"

def point_matches_answer(point: str, answer: str) -> bool:
    """
    Simple heuristic: case-insensitive substring after normalization.
    You can make this stricter/looser as needed.
    """
    pt = normalize_text(point)
    ans = normalize_text(answer)
    if not pt or not ans:
        return False
    # direct substring
    if pt in ans:
        return True
    # try a looser numeric check: if point has a % number, ensure that % number is in answer
    pcts = PCT_RE.findall(pt)
    if pcts:
        for p in pcts:
            if f"{p}%" in ans:
                return True
    # try numbers as tokens must appear
    nums = NUM_RE.findall(pt)
    if nums:
        ok = True
        for n in nums:
            n_norm = n.replace(",", "")
            if n_norm and (n_norm not in ans):
                ok = False
                break
        if ok:
            return True
    return False

def should_exclude_424b2(row: Dict[str, Any], corpus: Dict[str, Dict[str, Any]]|None) -> bool:
    """
    If corpus is provided, check any doc_ids_used title/metadata for 424B2.
    If not provided, fallback to look for '424b2' in question.
    """
    doc_ids = row.get("doc_ids_used") or []
    if corpus:
        for did in doc_ids:
            d = corpus.get(did)
            if not d:
                continue
            title = (d.get("title") or "").lower()
            if "424b2" in title:
                return True
            md = d.get("metadata") or {}
            if str(md.get("document_type","")).lower() == "424b2":
                return True
        return False
    # fallback heuristic
    q = (row.get("question") or "").lower()
    return "424b2" in q

def load_corpus_if_needed(path: str|None) -> Dict[str, Dict[str, Any]]|None:
    if not path:
        return None
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d=json.loads(line)
            _id=str(d.get("_id",""))
            if not _id: continue
            corpus[_id]={"title":d.get("title",""),"metadata":d.get("metadata",{})}
    return corpus

# ------------------------
# LLM-as-Judge integration
# ------------------------

def create_kp_judge_prompt(question: str, gold_answer: str, generated_answer: str, key_points: List[str]) -> str:
    """Build a prompt that asks the LLM to judge the answer and evaluate key points.

    We retain the ANALYSIS / DIMENSIONAL SCORES / VERDICT sections to stay close to
    the format in qa_judge_llm.py, and add a KEY POINTS EVALUATION section.
    """
    # Render key points as a numbered list for unambiguous referencing
    kp_lines = []
    for idx, kp in enumerate(key_points, start=1):
        kp_lines.append(f"{idx}. {kp}")
    kp_block = "\n".join(kp_lines) if kp_lines else "(none)"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert evaluator for financial Q&A tasks with retrieved evidence.

CORE DEFINITIONS:
- Use ONLY the provided GOLD ANSWER / KEY POINTS as reference.
- The generated answer may contain additional correct information beyond what's in the gold answer/key points.
- **CRITICAL CONSTRAINT: Without access to the full context/retrieved evidence, you cannot determine if additional information is unsubstantiated.**
- **Therefore: Only penalize for DIRECT CONTRADICTIONS with gold answer/key points.**
- **DO NOT penalize for information merely absent from gold answer/key points.**

VERDICT LABELS (use exactly one):

1. CORRECT:
   - All key points are correctly covered (explicitly or through reasonable paraphrase)
   - No factual errors or contradictions with gold answer/key points
   - Answer is relevant and complete
   - **May include additional correct information not in gold answer/key points**

2. PARTIAL:
   - Some key points are correctly covered, but important ones are missing
   - Missing key points are essential for answering the question
   - No factual errors or contradictions with gold answer/key points
   - Answer is relevant but incomplete
   - **May include additional correct information not in gold answer/key points**

3. INCORRECT:
   - **Contains DIRECT CONTRADICTIONS with gold answer/key points**
   - **Has clear factual errors that conflict with provided reference**
   - Answer is relevant but wrong
   - **DO NOT mark as incorrect for missing gold answer/key points info if answer has alternative correct info**

4. FAILURE:
    - Refuses to answer the question
    - Irrelevant/unrelated to the question
    - Empty/blank/no answer

5. ERROR:
    - API call failed
    - Other unexpected errors

KEY POINTS EVALUATION RULES:
- Check each key point against the generated answer.
- For each key point, label as one of:
  PRESENT (correctly mentioned), PARTIAL (partially addressed),
  MISSING (not addressed), INCORRECT (addressed but factually wrong).
- Consider numeric/date/percent equivalence and allow reasonable paraphrases.

ERROR TYPE TAXONOMY (choose NONE if VERDICT=CORRECT):

B) Generation-related
  B1. Hallucination: answer not entailed by retrieved evidence
    - **Hallucination = Information that CONTRADICTS gold answer/key points**
    - **NOT hallucination = Information absent from but not contradicting gold answer/key points**
  B2. Contradicts Evidence: explicitly conflicts with retrieved evidence
  B3. Excessive Inference: generalizes beyond a reasonable range based on the evidence
  B4. Evidence Fusion Failure: fails to correctly synthesize multiple evidence pieces (complementary or conflicting)

C) Finance-specific numeric & semantic errors
  C1. Numerical Precision: rounding/tolerance mistakes; % vs bps confusion
  C2. Units and scales: millions vs billions; ratio vs absolute confusion; currency/unit mismatch
  C3. Time mismatch: wrong period (e.g., annual vs quarterly, wrong FY/Q)
  C4. Computation Logic: uses correct data but computes incorrectly (formula/arithmetic error)

D) Query and context errors
  D1. Query misunderstanding: misidentifies intent, key entity, or asked metric
  D2. Context window abuse: loses key info due to length limits or fails to prioritize relevant parts

ERROR TAGGING RULES:
- Output 1 PRIMARY error group (B/C/D) and 1 PRIMARY subtype (B1..D2) when VERDICT != CORRECT.
- Optionally output up to 2 SECONDARY subtypes if multiple issues contribute.
- Prefer the MOST CAUSAL error: e.g., if evidence is present but model ignores it -> B/C; if question misunderstood -> D.

RESPONSE FORMAT (strict):
ALL SECTIONS BELOW ARE MANDATORY. Do not omit any section. Use exactly the headings and labels as shown. Do not add extra text outside this format.

ANALYSIS: [Concise analysis of answer quality, groundedness, and any numeric/unit/period issues]

KEY POINTS:
1. [PRESENT|PARTIAL|MISSING|INCORRECT] - brief justification
2. [PRESENT|PARTIAL|MISSING|INCORRECT] - brief justification
... (one line per key point)

KEY POINTS SUMMARY: matched=<int>; partial=<int>; missing=<int>; incorrect=<int> 
DIMENSIONAL SCORES: 
1. Information Coverage: [1-5]
- Includes all query-critical facts/constraints needed to answer.
- Avoids spending space on irrelevant details that don’t support the answer.
2. Reasoning Chain: [1-5]
- Provides a logical sequence linking evidence → intermediate conclusions → final answer.
- Not just paraphrasing; shows why the conclusion follows.
3. Factual Consistency: [1-5]
- Every stated claim is supported by the given evidence/context.
- No contradictions with evidence; no unsupported additions.
4. Clarity of Expression: [1-5]
- Main answer is easy to find; structure is organized (e.g., bullet points, clear sentences).
- Minimal redundancy; no “burying the lead” with unnecessary text.
5. Analytical Depth: [1-5]
- Selects and prioritizes relevant evidence rather than summarizing everything.
- Synthesizes/comparisons/inferences are reasonable and grounded in evidence.
- Produces a decisive, query-directed outcome (e.g., classification, comparison, recommendation).

ERROR TYPE:
PRIMARY_GROUP: [GENERATION_RELATED|FINANCE_NUMERIC_SEMANTIC|QUERY_CONTEXT|NONE]
PRIMARY_SUBTYPE: [B2|B3|B4|C1|C2|C3|C4|D1|D2|NONE]
SECONDARY_SUBTYPES: [<subtype>|<subtype>|NONE]
EVIDENCE_IDS_USED: [comma-separated ids from the provided evidence; or NONE]

VERDICT: [CORRECT|INCORRECT|PARTIAL|FAILURE]
(All sections are mandatory; the VERDICT line must contain only one of the listed labels.)
<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

GENERATED ANSWER:
{generated_answer}

KEY POINTS TO CHECK:
{kp_block}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt


def _call_ollama(prompt: str, host: str, port: int) -> str:
    try:
        import requests  # local import to avoid hard dependency
        url = f"http://{host}:{port}/api/generate"
        payload = {"model": "deepseek-r1:14b", "prompt": prompt, "stream": False}
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code == 200:
            obj = resp.json()
            return obj.get("response", "")
        logger.error("Ollama error %s: %s", resp.status_code, resp.text)
        return "ANALYSIS: Ollama API call failed\nVERDICT: ERROR"
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        return "ANALYSIS: Ollama API call failed\nVERDICT: ERROR"


class VLLMClient:
    def __init__(self, model_path: str, gpu_ids: str, tensor_parallel_size: int, gpu_mem_util: float):
        try:
            # Note: CUDA_VISIBLE_DEVICES is set in main() before this class is initialized.
            import torch
            import gc
            from vllm import LLM, SamplingParams

            self.clear_gpu_memory()

            if torch.cuda.is_available():
                num_gpus = len(gpu_ids.split(','))
                if tensor_parallel_size > 1 and tensor_parallel_size != num_gpus:
                    logger.warning(
                        "Mismatch between tensor_parallel_size (%d) and number of GPUs (%d). "
                        "Ensure tensor_parallel_size matches the number of GPUs in use.",
                        tensor_parallel_size, num_gpus
                    )
                logger.info("Using physical GPU(s) %s for vLLM with tensor_parallel_size=%d", gpu_ids, tensor_parallel_size)
            else:
                logger.warning("CUDA not available; vLLM may run on CPU")

            def _try_init(util: float):
                return LLM(
                    model=model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=util,
                    trust_remote_code=True,
                    dtype="auto",
                    max_model_len=4096,
                )

            # Try with requested utilization; on failure, retry with lower value(s)
            try:
                self.llm = _try_init(gpu_mem_util)
            except Exception as e1:
                logger.warning("vLLM init failed at gpu_memory_utilization=%.2f: %s", gpu_mem_util, e1)
                fallback_utils = []
                if gpu_mem_util > 0.6:
                    fallback_utils.append(0.6)
                if gpu_mem_util > 0.5:
                    fallback_utils.append(0.5)
                last_err = e1
                self.llm = None
                for fu in fallback_utils:
                    try:
                        logger.info("Retrying vLLM init with gpu_memory_utilization=%.2f", fu)
                        self.llm = _try_init(fu)
                        gpu_mem_util = fu
                        break
                    except Exception as e2:
                        last_err = e2
                        continue
                if self.llm is None:
                    raise last_err

            self.params = SamplingParams(temperature=0, top_p=0.9, max_tokens=300)
            self.torch = torch
            self.gc = gc
        except Exception as e:
            raise RuntimeError(f"Failed to init vLLM: {e}")

    def clear_gpu_memory(self):
        """Clear GPU memory before loading new model"""
        import torch
        import gc
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        gc.collect()
        logger.info("GPU memory cleared before vLLM init")

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts with OOM fallback."""
        if not prompts:
            return []
        try:
            outputs = self.llm.generate(prompts, self.params)
            return [out.outputs[0].text for out in outputs]
        except self.torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM on batch generation, retrying prompts individually.")
            self.clear_gpu_memory()
            
            results = []
            for i, prompt in enumerate(prompts):
                try:
                    single_output = self.llm.generate([prompt], self.params)
                    results.append(single_output[0].outputs[0].text)
                    if (i + 1) % 5 == 0:
                        self.clear_gpu_memory()
                except Exception as e:
                    logger.error(f"Failed to generate for a single prompt after OOM: {e}")
                    results.append("ANALYSIS: vLLM generation failed due to OOM\nVERDICT: ERROR")
            return results
        except Exception as e:
            logger.error(f"An unexpected vLLM error occurred during generation: {e}")
            return ["ANALYSIS: vLLM generation failed\nVERDICT: ERROR"] * len(prompts)


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
            top_p=0.9,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("OpenAI call failed: %s", e)
        return "ANALYSIS: OpenAI API call failed\nVERDICT: ERROR"


def _call_gpt(prompt: str, deployment_name: str, reasoning_effort: str | None = "low") -> str:
    """
    Call GPT via Azure/OpenAI similar to evaluation/gpt4_qa.py:
    - If deployment_name starts with 'gpt-5': use OpenAI client with Azure base_url
    - Else: use AzureOpenAI client with endpoint+api_version
    Removes any <think>...</think> blocks from the output.
    """
    try:
        # Local imports to avoid hard dependency unless this backend is used
        from openai import AzureOpenAI, OpenAI  # type: ignore
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

        if not api_key:
            logger.error("AZURE_OPENAI_API_KEY not set for GPT backend")
            return "ANALYSIS: Missing AZURE_OPENAI_API_KEY\nVERDICT: ERROR"

        is_gpt5_family = deployment_name.startswith("gpt-5")

        if is_gpt5_family:
            endpoint = "https://chronosense.openai.azure.com/openai/v1"
            client = OpenAI(base_url=endpoint, api_key=api_key)
            completion_params = {
                "model": deployment_name,
                "messages": [{"role": "user", "content": prompt}],
                "stop": None,
                "stream": False,
                "max_completion_tokens": 8192,
            }
        else:
            endpoint = os.getenv("ENDPOINT_URL", "https://chronosense.openai.azure.com/")
            api_version = "2025-01-01-preview"
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                max_retries=5,
            )
            completion_params = {
                "model": deployment_name,
                "messages": [{"role": "user", "content": prompt}],
                "stop": None,
                "stream": False,
                "max_tokens": 8192,
                "temperature": 0,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }

        # Prefer Responses API for reasoning-capable models when supported.
        # Fall back to Chat Completions if the endpoint doesn't support it.
        if is_gpt5_family:
            try:
                responses_params: Dict[str, Any] = {
                    "model": deployment_name,
                    "input": prompt,
                    "max_output_tokens": completion_params["max_completion_tokens"],
                }
                if reasoning_effort is not None:
                    responses_params["reasoning"] = {"effort": reasoning_effort}

                resp = client.responses.create(**responses_params)
                text = resp.output_text
            except Exception:
                completion = client.chat.completions.create(**completion_params)
                text = completion.choices[0].message.content or ""
        else:
            completion = client.chat.completions.create(**completion_params)
            text = completion.choices[0].message.content or ""

        if text:
            text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE)
        return text
    except Exception as e:
        logger.error("GPT (Azure/OpenAI) call failed: %s", e)
        return "ANALYSIS: GPT API call failed\nVERDICT: ERROR"


def parse_kp_judge_response(text: str) -> Tuple[str, Dict[str, int], Dict[str, int], str, Dict[str, Any], List[str]]:
    """Parse LLM response.

    Returns:
    - analysis: str
    - scores: dict[str,int] for five dimensions
    - kp_counts: dict with keys 'matched','partial','missing','incorrect'
    - verdict: str
    - error_info: dict with keys error_primary_group, error_primary_subtype, error_secondary_subtypes
    - evidence_ids_used: list[str]
    """
    analysis_match = re.search(r"ANALYSIS:\s*(.*?)(?=KEY POINTS:|KEY POINTS SUMMARY:|DIMENSIONAL SCORES:|VERDICT:|$)", text, re.DOTALL)
    analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided"

    # Parse KEY POINTS SUMMARY counts (flexible separators/order)
    kp_counts = {"matched": None, "partial": None, "missing": None, "incorrect": None}  # type: ignore[assignment]
    kp_sum_block = re.search(r"KEY POINTS SUMMARY:\s*(.*?)(?=DIMENSIONAL SCORES:|VERDICT:|$)", text, re.IGNORECASE | re.DOTALL)
    if kp_sum_block:
        block = kp_sum_block.group(1)
        def _find(label: str) -> int | None:
            m = re.search(fr"{label}\s*[:=]\s*(\d+)", block, re.IGNORECASE)
            return int(m.group(1)) if m else None
        m = _find("matched")
        p = _find("partial")
        mi = _find("missing")
        inc = _find("incorrect")
        # Only set if at least one is found
        if any(v is not None for v in (m, p, mi, inc)):
            kp_counts = {
                "matched": m if m is not None else 0,
                "partial": p if p is not None else 0,
                "missing": mi if mi is not None else 0,
                "incorrect": inc if inc is not None else 0,
            }
    # If summary not found, try parsing the KEY POINTS lines
    if kp_counts["matched"] is None:
        kp_block_match = re.search(r"KEY POINTS:\s*(.*?)(?=KEY POINTS SUMMARY:|DIMENSIONAL SCORES:|VERDICT:|$)", text, re.IGNORECASE | re.DOTALL)
        if kp_block_match:
            block = kp_block_match.group(1)
            present = len(re.findall(r"\bPRESENT\b", block, re.IGNORECASE))
            partial = len(re.findall(r"\bPARTIAL\b", block, re.IGNORECASE))
            missing = len(re.findall(r"\bMISSING\b", block, re.IGNORECASE))
            incorrect = len(re.findall(r"\bINCORRECT\b", block, re.IGNORECASE))
            total_detected = present + partial + missing + incorrect
            if total_detected > 0:
                kp_counts = {
                    "matched": present,
                    "partial": partial,
                    "missing": missing,
                    "incorrect": incorrect,
                }

    # Parse dimensional scores (fallback to 1s if absent)
    scores: Dict[str, int] = {}
    dims = [
        "Information Coverage",
        "Reasoning Chain",
        "Factual Consistency",
        "Clarity of Expression",
        "Analytical Depth",
    ]
    dim_block = re.search(r"DIMENSIONAL SCORES:(.*?)(?=VERDICT:|$)", text, re.DOTALL)
    if dim_block:
        s = dim_block.group(1)
        for d in dims:
            m = re.search(fr"{re.escape(d)}:\s*([1-5])", s)
            if m:
                scores[d] = int(m.group(1))
    if not scores:
        scores = {d: 1 for d in dims}

    # Parse verdict
    verdict_match = re.search(r"VERDICT:\s*(CORRECT|INCORRECT|PARTIAL|FAILURE|ERROR)", text, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "UNCLEAR"

    # Parse ERROR TYPE block
    group_match = re.search(r"PRIMARY_GROUP:\s*\[?([A-Z_]+|NONE)\]?", text, re.IGNORECASE)
    error_primary_group = (group_match.group(1).upper() if group_match else "NONE")
    subtype_match = re.search(r"PRIMARY_SUBTYPE:\s*\[?([A-Z]\d+|NONE)\]?", text, re.IGNORECASE)
    error_primary_subtype = (subtype_match.group(1).upper() if subtype_match else "NONE")
    sec_match = re.search(r"SECONDARY_SUBTYPES:\s*\[?(.*?)\]?(?=\n|$)", text, re.IGNORECASE | re.DOTALL)
    error_secondary_subtypes: List[str] = []
    if sec_match:
        raw = sec_match.group(1)
        parts = [p.strip().upper() for p in re.split(r"[,\s]+", raw) if p.strip()]
        error_secondary_subtypes = [p for p in parts if p not in ("NONE",)]

    # Parse evidence ids used
    ev_match = re.search(r"EVIDENCE_IDS_USED:\s*\[?(.*?)\]?(?=\n|$)", text, re.IGNORECASE | re.DOTALL)
    evidence_ids_used: List[str] = []
    if ev_match:
        raw = ev_match.group(1).strip()
        if raw.upper() != "NONE" and raw != "":
            parts = [p.strip().strip("'").strip('"') for p in raw.split(",")]
            evidence_ids_used = [p for p in parts if p]

    error_info = {
        "error_primary_group": error_primary_group,
        "error_primary_subtype": error_primary_subtype,
        "error_secondary_subtypes": error_secondary_subtypes,
    }

    return analysis, scores, kp_counts, verdict, error_info, evidence_ids_used

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to input JSON file(s) (file, dir, or glob). Each entry must have 'qid', 'question', 'generated_answer', and 'key_points'.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_424b2", action="store_true", help="Exclude QAs tied to 424B2 forms")
    ap.add_argument("--corpus", default=None, help="Optional corpus jsonl for precise 424B2 exclusion")
    # LLM backends
    ap.add_argument("--judge_backend", choices=["ollama","openai","vllm","gpt"], default="ollama")
    ap.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--openai_model", default="gpt-4.1-nano")
    ap.add_argument(
        "--reasoning_effort",
        default="low",
        choices=["low", "medium", "high", "none"],
        help="Reasoning effort for gpt-5* deployments when using --judge_backend gpt (use 'none' to disable).",
    )
    ap.add_argument("--vllm_model_path", default=None)
    ap.add_argument("--vllm_gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use for vLLM.")
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM if using multiple GPUs.")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.8, help="Fraction of GPU memory to use (0-1)")
    ap.add_argument("--ollama_port", type=int, default=11434)
    ap.add_argument("--ollama_host", default="localhost")
    ap.add_argument("--max_examples", type=int, default=None, help="If set, process only the first N QAs considered after applying filters.")
    ap.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="1-indexed start position among QAs after applying filters (useful for resuming).",
    )
    ap.add_argument("--ollama_gpu_id", type=str, default=None, help="GPU ID to use for Ollama backend (sets CUDA_VISIBLE_DEVICES).")
    args = ap.parse_args()

    if args.start_index < 1:
        raise ValueError("--start_index must be >= 1 (1-indexed)")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    qa_data_map = load_answers(args.input_json)
    corpus = load_corpus_if_needed(args.corpus)

    # Initialize backend if needed
    if args.judge_backend == "ollama" and args.ollama_gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.ollama_gpu_id
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.ollama_gpu_id} for Ollama backend")

    vllm_client = None
    llm_responses: Dict[str, str] = {}

    # Resume support: if start_index != 1 and results.json exists, do not discard it.
    # We'll also avoid double-counting by skipping qids already present in existing results.json.
    res_path = Path(args.out_dir) / "results.json"
    resume_append = (args.start_index != 1 and res_path.exists())
    previous_results: List[Dict[str, Any]] = []
    existing_qids: set[str] = set()
    if resume_append:
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                previous_results = loaded
                for r in previous_results:
                    if isinstance(r, dict):
                        qid0 = r.get("qid")
                        if isinstance(qid0, str) and qid0:
                            existing_qids.add(qid0)
            else:
                logger.warning("Existing results.json is not a list; will overwrite: %s", res_path)
                resume_append = False
                previous_results = []
        except Exception as e:
            logger.warning("Failed to load existing results.json; will overwrite (%s): %s", res_path, e)
            resume_append = False
            previous_results = []
            existing_qids = set()

    if args.judge_backend == "vllm":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.vllm_gpu_ids
        if not args.vllm_model_path:
            raise ValueError("--vllm_model_path is required for vllm backend")
        vllm_client = VLLMClient(
            args.vllm_model_path,
            args.vllm_gpu_ids,
            args.vllm_tensor_parallel_size,
            args.vllm_gpu_memory_utilization
        )

        prompts_to_run: List[Tuple[str, str]] = []
        seen_after_filters = 0
        for qid, ex in qa_data_map.items():
            if args.exclude_424b2 and should_exclude_424b2(ex, corpus):
                continue
            seen_after_filters += 1
            if seen_after_filters < args.start_index:
                continue
            if qid in existing_qids:
                continue
            if not extract_keypoints(ex):
                continue
            gen = ex.get("generated_answer") or ex.get("final_answer") or ""
            question = ex.get("question", "")
            gold_answer = ex.get("answer") or ""
            prompt = create_kp_judge_prompt(question, gold_answer, gen, extract_keypoints(ex))
            prompts_to_run.append((qid, prompt))
            if args.max_examples is not None and len(prompts_to_run) >= args.max_examples:
                break

        batch_size = 16
        for i in tqdm(range(0, len(prompts_to_run), batch_size), desc="Judging QA batches (vLLM)"):
            batch = prompts_to_run[i:i + batch_size]
            qids_in_batch = [item[0] for item in batch]
            prompts_in_batch = [item[1] for item in batch]

            if not prompts_in_batch:
                continue

            generated_texts = vllm_client.generate_batch(prompts_in_batch)
            for qid, text in zip(qids_in_batch, generated_texts):
                llm_responses[qid] = text

    # Prepare streaming results.json to write each case as it's processed.
    # If resuming (start_index != 1) and results.json exists, append instead of overwriting.
    if resume_append:
        # Open existing file and remove the closing ']' so we can append new objects.
        _results_stream = open(res_path, "r+", encoding="utf-8")
        content = _results_stream.read()
        stripped = content.rstrip()
        if not stripped.endswith("]"):
            logger.warning("Existing results.json does not end with ']'; will overwrite: %s", res_path)
            _results_stream.close()
            resume_append = False
        else:
            last_bracket_pos = content.rfind("]")
            _results_stream.seek(last_bracket_pos)
            _results_stream.truncate()
            before_close = stripped[:-1].rstrip()
            _first_result_written = not before_close.endswith("[")
    if not resume_append:
        _results_stream = open(res_path, "w", encoding="utf-8")
        _results_stream.write("[\n")
        _first_result_written = False

    results = []
    total_considered = 0
    seen_after_filters = 0

    for qid, ex in tqdm(qa_data_map.items(), desc="Processing QAs"):
        if args.max_examples is not None and total_considered >= args.max_examples:
            break
        kps = extract_keypoints(ex)
        qa_type = qa_type_from_qid(qid)

        if args.exclude_424b2 and should_exclude_424b2(ex, corpus):
            # skip this QA entirely
            continue

        seen_after_filters += 1
        if seen_after_filters < args.start_index:
            continue
        if qid in existing_qids:
            continue

        total_considered += 1
        # Prefer generated_answer; fallback to final_answer or answer to support multiple generators
        gen = ex.get("generated_answer") or ex.get("final_answer") or ex.get("answer") or ""
        question = ex.get("question", "")
        gold_answer = ex.get("answer") or ""

        if not kps:
            result_row = {
                "qid": qid,
                "qa_type": qa_type,
                "n_keypoints": 0,
                "kp_matched": 0,
                "kp_partial": 0,
                "kp_missing": 0,
                "kp_incorrect": 0,
                "kp_coverage_ratio": None,
                "judge_verdict": None,
                "judge_analysis": "no_key_points_in_source",
                "passed": None,
            }
            results.append(result_row)
            # Stream write this result
            if _first_result_written:
                _results_stream.write(",\n")
            json.dump(result_row, _results_stream, ensure_ascii=False, indent=2)
            _results_stream.flush()
            _first_result_written = True
            continue

        # Build prompt and query LLM
        llm_text = ""
        if args.judge_backend == "vllm":
            llm_text = llm_responses.get(qid, "")
        else:
            prompt = create_kp_judge_prompt(question, gold_answer, gen, kps)
            if args.judge_backend == "ollama":
                llm_text = _call_ollama(prompt, args.ollama_host, args.ollama_port)
            elif args.judge_backend == "openai":
                if not args.openai_api_key:
                    logger.error("OpenAI API key not provided; falling back to rule-based")
                else:
                    llm_text = _call_openai(prompt, args.openai_api_key, args.openai_model)
            elif args.judge_backend == "gpt":
                # Use Azure/OpenAI GPT backend mirroring evaluation/gpt4_qa.py
                # Reuse openai_model as the deployment name (e.g., 'gpt-4.1' or 'gpt-5')
                deployment_name = args.openai_model or "gpt-4.1"
                # Some defaults from gpt4_qa.py expect 'gpt-4.1'/'gpt-5'; if user kept the default 'gpt-4.1-nano',
                # still attempt the call; backend will error gracefully if unsupported.
                effort = None if args.reasoning_effort == "none" else args.reasoning_effort
                llm_text = _call_gpt(prompt, deployment_name, reasoning_effort=effort)

        judge_analysis = None
        judge_scores: Dict[str, int] = {}
        kp_counts = {"matched": None, "partial": None, "missing": None, "incorrect": None}
        judge_verdict = None

        parsed_error_info: Dict[str, Any] = {}
        parsed_evidence_ids: List[str] = []
        if llm_text:
            try:
                analysis, scores, kp_counts_parsed, verdict, error_info, evidence_ids_used = parse_kp_judge_response(llm_text)
                judge_analysis = analysis
                judge_scores = scores
                kp_counts = kp_counts_parsed
                judge_verdict = verdict
                parsed_error_info = error_info or {}
                parsed_evidence_ids = evidence_ids_used or []
            except Exception as e:
                logger.error("Failed to parse LLM response, falling back: %s", e)

        # If VERDICT is missing/unclear but we have KP counts, infer a verdict heuristically.
        if judge_verdict in (None, "UNCLEAR", "ERROR"):
            try:
                if kp_counts["matched"] is not None:
                    matched_count = int(kp_counts.get("matched") or 0)
                    partial_count = int(kp_counts.get("partial") or 0)
                    incorrect_count = int(kp_counts.get("incorrect") or 0)
                    total_kps = len(kps)
                    if total_kps > 0 and matched_count == total_kps:
                        judge_verdict = "CORRECT"
                    elif matched_count > 0 or partial_count > 0:
                        judge_verdict = "PARTIAL"
                    elif incorrect_count > 0 or (total_kps > 0 and matched_count == 0 and partial_count == 0):
                        judge_verdict = "INCORRECT"
            except Exception as e:
                logger.warning("Failed to infer verdict from KP counts: %s", e)

        # If LLM failed or no counts, fallback to deterministic matching
        if kp_counts["matched"] is None:
            matched = sum(1 for kp in kps if point_matches_answer(kp, gen))
            kp_counts = {
                "matched": matched,
                "partial": 0,
                "missing": max(0, len(kps) - matched),
                "incorrect": 0,
            }
            if not judge_verdict:
                judge_verdict = "CORRECT" if matched == len(kps) else ("PARTIAL" if matched > 0 else "INCORRECT")
            if not judge_analysis:
                judge_analysis = "Rule-based fallback applied."

        kp_coverage_ratio = kp_counts["matched"] / max(1, len(kps))
        # Pass mirrors LLM verdict; fallback verdict used when LLM fails
        # Final safeguard: if verdict is still missing/unclear/error, infer from KP counts
        if judge_verdict in (None, "UNCLEAR", "ERROR"):
            matched_count = int(kp_counts.get("matched") or 0)
            partial_count = int(kp_counts.get("partial") or 0)
            incorrect_count = int(kp_counts.get("incorrect") or 0)
            total_kps = len(kps)
            if total_kps > 0 and matched_count == total_kps:
                judge_verdict = "CORRECT"
            elif matched_count > 0 or partial_count > 0:
                judge_verdict = "PARTIAL"
            elif incorrect_count > 0 or (total_kps > 0 and matched_count == 0 and partial_count == 0):
                judge_verdict = "INCORRECT"

        passed = (judge_verdict == "CORRECT")

        result_row = {
            "qid": qid,
            "qa_type": qa_type,
            "n_keypoints": len(kps),
            "kp_matched": kp_counts["matched"],
            "kp_partial": kp_counts["partial"],
            "kp_missing": kp_counts["missing"],
            "kp_incorrect": kp_counts["incorrect"],
            "kp_coverage_ratio": round(kp_coverage_ratio, 4),
            "judge_verdict": judge_verdict,
            "judge_analysis": judge_analysis,
            "judge_scores": judge_scores,
            "error_primary_group": parsed_error_info.get("error_primary_group"),
            "error_primary_subtype": parsed_error_info.get("error_primary_subtype"),
            "error_secondary_subtypes": parsed_error_info.get("error_secondary_subtypes"),
            "evidence_ids_used": parsed_evidence_ids,
            "passed": bool(passed),
        }
        results.append(result_row)
        # Stream write this result
        if _first_result_written:
            _results_stream.write(",\n")
        json.dump(result_row, _results_stream, ensure_ascii=False, indent=2)
        _results_stream.flush()
        _first_result_written = True

    # close streaming results array
    _results_stream.write("\n]\n")
    _results_stream.close()

    # write CSV
    csv_path = Path(args.out_dir) / "details.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "qid","qa_type","n_keypoints",
            "kp_matched","kp_partial","kp_missing","kp_incorrect",
            "kp_coverage_ratio","judge_verdict","passed"
        ])
        all_rows_for_csv = (previous_results + results) if resume_append else results
        for r in all_rows_for_csv:
            w.writerow([
                r["qid"], r["qa_type"], r["n_keypoints"],
                r.get("kp_matched"), r.get("kp_partial"), r.get("kp_missing"), r.get("kp_incorrect"),
                r.get("kp_coverage_ratio"), r.get("judge_verdict"), r["passed"]
            ])

    # summary (include previous cases if we resumed/appended)
    all_results_for_summary = (previous_results + results) if resume_append else results
    overall = {
        "evaluated_qas": len(all_results_for_summary),
        # total_considered counts only newly processed items in this run; for resumed runs we want the combined count.
        "total_considered_after_filters": len(all_results_for_summary) if resume_append else total_considered,
        "verdict_counts": {
            "CORRECT": sum(1 for r in all_results_for_summary if r.get("judge_verdict") == "CORRECT"),
            "PARTIAL": sum(1 for r in all_results_for_summary if r.get("judge_verdict") == "PARTIAL"),
            "INCORRECT": sum(1 for r in all_results_for_summary if r.get("judge_verdict") == "INCORRECT"),
            "FAILURE": sum(1 for r in all_results_for_summary if r.get("judge_verdict") == "FAILURE"),
            "ERROR/UNCLEAR": sum(1 for r in all_results_for_summary if r.get("judge_verdict") in ("ERROR","UNCLEAR",None)),
        }
    }
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in all_results_for_summary:
        t = r["qa_type"]
        by_type.setdefault(t, {"count":0,"passed":0,"avg_ratio_sum":0.0,"with_ratio":0})
        by_type[t]["count"] += 1
        if r["passed"] is True:
            by_type[t]["passed"] += 1
        if r.get("kp_coverage_ratio") is not None:
            by_type[t]["avg_ratio_sum"] += r["kp_coverage_ratio"]
            by_type[t]["with_ratio"] += 1
    for t, s in by_type.items():
        s["pass_rate"] = round(s["passed"]/max(1,s["count"]), 4)
        s["avg_match_ratio"] = round(s["avg_ratio_sum"]/max(1,s["with_ratio"]), 4)

    summary = {"overall": overall, "by_type": by_type}
    sum_path = Path(args.out_dir) / "summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "results": str(res_path),
        "details_csv": str(csv_path),
        "summary": str(sum_path),
        **summary
    }, indent=2))

if __name__ == "__main__":
    main()
