import json
import html
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

from normalize import format_candidates, normalize_meaning_zh_soft


# ===== 路径配置（ZK） =====
APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent
PROJECT_ROOT = APP_DIR.parent
ZK_DIR = APP_DIR / "zk"
ZK_INPUT_DIR = ZK_DIR / "input"
ZK_OUTPUT_DIR = ZK_DIR / "output"
ZK_BUILD_DIR = ZK_DIR / "build"
ZK_CIRCUITS_DIR = ZK_DIR / "circuits"
ZK_COMMON_DIR = PROJECT_ROOT / "zk_common"
PTAU_DIR = ZK_COMMON_DIR / "ptau"

TRUTHFINDER_PATH = APP_DIR / "TruthFinder.py"
NORMALIZE_PATH = APP_DIR / "normalize.py"
CIRCUIT_SPEC_PATH = APP_DIR / "circuit_spec.json"
SCHEMA_PATH = ZK_DIR / "truthfinder_runtime_input_schema.json"
PTAU_FINAL_PATH = PTAU_DIR / "pot24_final.ptau"
CIRCOM_INPUT_PREPARE_PATH = ZK_DIR / "prepare_circom_input.py"
CIRCUIT_REF_PATH = ZK_DIR / "TruthFinder_circuit_ref.py"
CIRCUIT_PATH = ZK_CIRCUITS_DIR / "truthfinder.circom"

WASM_PATH = ZK_BUILD_DIR / "truthfinder_js" / "truthfinder.wasm"
WITNESS_JS_PATH = ZK_BUILD_DIR / "truthfinder_js" / "generate_witness.js"
FINAL_ZKEY_PATH = ZK_BUILD_DIR / "truthfinder_final.zkey"
VERIFICATION_KEY_PATH = ZK_BUILD_DIR / "verification_key.json"

RUNTIME_INPUT_PATH = ZK_INPUT_DIR / "truthfinder_runtime_input.json"
DENSE_INPUT_PATH = ZK_INPUT_DIR / "truthfinder_dense_input.json"
CIRCOM_INPUT_PATH = ZK_INPUT_DIR / "truthfinder_circom_input.json"
WITNESS_INPUT_PATH = ZK_INPUT_DIR / "truthfinder_witness_input.json"
WITNESS_PATH = ZK_BUILD_DIR / "witness.wtns"
PROOF_PATH = ZK_OUTPUT_DIR / "proof.json"
PUBLIC_PATH = ZK_OUTPUT_DIR / "public.json"
REFERENCE_OUTPUT_PATH = ZK_OUTPUT_DIR / "reference_output.json"

SNARKJS_CLI_PATH = PROJECT_ROOT / "node_modules" / "snarkjs" / "build" / "cli.cjs"

ZK_INPUT_DIR.mkdir(parents=True, exist_ok=True)
ZK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZK_BUILD_DIR.mkdir(parents=True, exist_ok=True)

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if str(ZK_DIR) not in sys.path:
    sys.path.insert(0, str(ZK_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TruthFinder import (  # noqa: E402
    TruthFinderConfig,
    pick_truth_per_keyword,
    rank_models_by_trust,
    truthfinder_run,
)
from expander import expand_runtime_input  # noqa: E402
from prepare_circom_input import (  # noqa: E402
    build_witness_input_from_circom_input,
    prepare_circom_input_from_dense,
)
from zk_input_builder import build_truthfinder_runtime_input_from_state  # noqa: E402
from TruthFinder_circuit_ref import run_truthfinder_circuit_ref_from_file  # noqa: E402


# ===== ollama部署的4个模型 =====
MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q5_0",
    "gemma2:9b-instruct-q4_K_M",
    "koesn/mistral-7b-instruct:Q4_0",
]

OLLAMA_URL = "http://localhost:11434/api/chat"
ZK_DEMO_K_MAX = 10
ZK_DEMO_N_MAX = 8
ZK_DEMO_ITER_N = 15
ZK_DEMO_TOPN_PER_MODEL = 2


def run_command(cmd, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    """统一运行外部命令，返回 (success, message)。"""
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        out = (completed.stdout or "").strip()
        err = (completed.stderr or "").strip()
        message = out if out else ""
        if err:
            message = (message + "\n" + err).strip()

        if completed.returncode != 0:
            return False, f"命令失败(returncode={completed.returncode}): {' '.join(map(str, cmd))}\n{message}"
        return True, message or f"命令执行成功: {' '.join(map(str, cmd))}"
    except Exception as ex:
        return False, f"执行命令异常: {' '.join(map(str, cmd))}\n{type(ex).__name__}: {ex}"

def run_command_details(cmd, cwd: Optional[Path] = None) -> Dict[str, Any]:
    cmd_list = [str(part) for part in cmd]
    try:
        completed = subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        out = (completed.stdout or "").strip()
        err = (completed.stderr or "").strip()
        message = out if out else ""
        if err:
            message = (message + "\n" + err).strip()
        return {
            "success": completed.returncode == 0,
            "command": " ".join(cmd_list),
            "cwd": str(cwd or PROJECT_ROOT),
            "returncode": completed.returncode,
            "stdout": out,
            "stderr": err,
            "message": message or f"命令执行成功: {' '.join(cmd_list)}",
        }
    except Exception as ex:
        return {
            "success": False,
            "command": " ".join(cmd_list),
            "cwd": str(cwd or PROJECT_ROOT),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(ex).__name__}: {ex}",
            "message": f"执行命令异常: {' '.join(cmd_list)}\n{type(ex).__name__}: {ex}",
        }


def call_ollama(model: str, prompt: str, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严谨、可靠的英语翻译与词汇释义助手。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


# 停用词表（用于自动抽取英文关键词时过滤掉无意义高频词）
_DASH_TRANSLATION = str.maketrans({
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2212": "-",
    "\ufe58": "-",
    "\ufe63": "-",
    "\uff0d": "-",
    "‐": "-",
    "-": "-",
    "‒": "-",
    "–": "-",
    "−": "-",
    "﹣": "-",
    "－": "-",
    "’": "'",
    "‘": "'",
    "＇": "'",
    "ʼ": "'",
})

TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*")

STOPWORDS = set(
    """
a an the and or but if then is are was were be been being
to of in on at by for from with as can both all among between into
this that these those it its they them we our you your i he she my their
who whom whose which how what when where why
have has had having do does did doing done only
about above after against along around before behind below beside beyond not
each any all some many most
""".split()
)

LOW_VALUE_WORDS = set(
    """
given using based include including should would could may might
make made get got take taken
create created provide provided
question questions answer answers
level topic focus
long term self
""".split()
)

DOMAIN_TERMS = {
    "adversarial",
    "prompt",
    "prompts",
    "harmlessness",
    "safety",
    "generation",
    "generations",
    "model",
    "medical",
    "validation",
    "validated",
    "feature-based",
    "real-world",
}


def score_keyword_candidate(word: str, freq: int, first_pos: int) -> float:
    score = freq * 1.0
    word_len = len(word)

    if word_len >= 10:
        score += 1.5
    elif word_len >= 8:
        score += 1.0
    elif word_len >= 6:
        score += 0.5

    if "-" in word:
        score += 3.0

    if word.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ive", "ous", "al")):
        score += 0.8

    if word in DOMAIN_TERMS:
        score += 1.5

    if word in LOW_VALUE_WORDS:
        score -= 2.0

    return score


def normalize_english_for_keyword_extraction(text: str) -> str:
    s = (text or "").translate(_DASH_TRANSLATION)
    s = re.sub(r"(?<=[A-Za-z0-9])\s*-\s*(?=[A-Za-z0-9])", "-", s)
    return s


def _rank_keyword_candidates(english_text: str) -> list[dict]:
    normalized_text = normalize_english_for_keyword_extraction(english_text)
    words = TOKEN_RE.findall(normalized_text)
    if not words:
        return []

    freq: Dict[str, int] = {}
    first_pos: Dict[str, int] = {}

    for pos, word in enumerate(words):
        low = word.lower()
        if low in STOPWORDS or len(low) < 3:
            continue
        freq[low] = freq.get(low, 0) + 1
        if low not in first_pos:
            first_pos[low] = pos

    ranked = []
    for word, count in freq.items():
        ranked.append(
            {
                "word": word,
                "freq": count,
                "first_pos": first_pos[word],
                "score": score_keyword_candidate(word, count, first_pos[word]),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["first_pos"]))
    return ranked


def extract_keywords(english_text: str, k: int = 6):
    """
    从英文原文里抽取 k 个关键词
    规则：支持连字符词，去停用词，按关键词分数排序
    """
    ranked = _rank_keyword_candidates(english_text)
    return [item["word"] for item in ranked[:k]]


def build_prompt_keyword_meanings(keywords: list) -> str:
    """构建提示词：要求模型按给定关键词列表逐个输出中文释义（JSON格式）。"""
    kw_json = json.dumps(keywords, ensure_ascii=False)
    return f"""
你是一个严谨的英语学习与翻译助手。请只输出 JSON，不要输出任何多余文字。

我给你一个关键词数组 keywords（必须保持原样、顺序一致、数量一致），你必须逐个给出中文释义：
- 输出中的 keywords 数组长度必须与输入一致
- 输出中的每个 word 必须与输入 keywords 对应位置完全相同（不要改写、不要合并、不要拆分）
- 不允许新增或删除任何关键词
- meaning_zh 不允许为空字符串

输入关键词数组：
{kw_json}

输出 JSON 格式必须严格如下（必须严格按输入顺序逐个返回）：
{{
  "keywords": [
    {{"word": "与输入keywords[0]完全相同", "meaning_zh": "中文释义"}},
    {{"word": "与输入keywords[1]完全相同", "meaning_zh": "中文释义"}}
  ]
}}
""".strip()


def try_parse_json(text: str):
    """
    尝试从模型输出中解析 JSON。
    容错策略：
    1) 直接 json.loads
    2) 提取 ```json ``` 代码块中的内容
    3) 截取第一个 "{" 到 最后一个 "}" 之间的内容
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```json\s*(\{.*?})\s*```", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(text[l : r + 1])
        except Exception:
            return None
    return None


def get_keyword_meanings(model: str, keywords: list, timeout: int = 180) -> dict:
    """调用模型，获取关键词列表对应的中文释义，返回 {word: meaning_zh} 字典。"""
    prompt = build_prompt_keyword_meanings(keywords)
    out = call_ollama(model, prompt, timeout=timeout)
    parsed = try_parse_json(out)
    if not parsed:
        return {}
    kw_map = {}
    model_kw = parsed.get("keywords", [])
    if isinstance(model_kw, list):
        for item in model_kw:
            if isinstance(item, dict):
                w = str(item.get("word", "")).strip()
                z = str(item.get("meaning_zh", "")).strip()
                if w:
                    kw_map[w] = z
    return kw_map


def fill_missing_meanings(model: str, words: list, timeout: int = 180) -> dict:
    """对于给定的未获取释义的英文单词列表，调用模型补全中文释义，返回 {word: meaning_zh}。"""
    words_json = json.dumps(words, ensure_ascii=False)
    prompt = f"""
你是英语词汇翻译助手。请只输出 JSON，不要输出任何多余文字。

给定英文单词列表 words，请逐个给出准确中文释义（每个都必须填写，不允许空字符串）。

输入 words:
{words_json}

输出格式严格如下：
{{
  "meanings": [
    {{"word": "words[0]原样", "meaning_zh": "中文释义"}},
    {{"word": "words[1]原样", "meaning_zh": "中文释义"}}
  ]
}}
""".strip()
    out = call_ollama(model, prompt, timeout=timeout)
    parsed = try_parse_json(out)
    if not parsed:
        return {}
    meanings = {}
    arr = parsed.get("meanings", [])
    if isinstance(arr, list):
        for item in arr:
            if isinstance(item, dict):
                w = str(item.get("word", "")).strip()
                z = str(item.get("meaning_zh", "")).strip()
                if w:
                    meanings[w] = z
    return meanings


def generate_translation_only(model: str, english_text: str, timeout: int = 180) -> str:
    """调用模型，仅生成整段英文文本的中文翻译结果。"""
    prompt = f"""
你是严谨的英译中助手。请只输出中文翻译结果，不要输出任何解释。

英文原文：
\"\"\"{english_text}\"\"\"
""".strip()
    out = call_ollama(model, prompt, timeout=timeout)
    return (out or "").strip()


def collect_current_zk_state() -> Dict[str, Any]:
    """从当前页面已生成状态中收集 ZK 输入构建所需数据。"""
    english_text = (st.session_state.get("last_english") or "").strip()
    keywords = st.session_state.get("fixed_keywords") or []
    results = st.session_state.get("results") or {}

    if not english_text:
        raise ValueError("当前英文文本为空，请先输入并运行主流程。")
    if not keywords:
        raise ValueError("当前关键词为空，请先完成关键词提取与模型调用。")
    if not results:
        raise ValueError("当前模型结果为空，请先点击“调用模型”。")

    normalized_by_model: Dict[str, Dict[str, list]] = {}
    for model_name in MODELS:
        model_payload = results.get(model_name, {}) or {}
        rows = model_payload.get("keywords", []) or []
        kw2meaning = {str(r.get("keyword", "")).strip(): str(r.get("meaning_zh", "")).strip() for r in rows}
        normalized_by_model[model_name] = {}
        for kw in keywords:
            cands = normalize_meaning_zh_soft(
                kw2meaning.get(kw, ""),
                top_n=ZK_DEMO_TOPN_PER_MODEL,
            )
            normalized_by_model[model_name][kw] = [c for c in cands if c and c.strip()]

    payload_cfg = ((st.session_state.get("truthfinder_payload") or {}).get("cfg_dict") or {})
    cfg_dict = dict(payload_cfg) if payload_cfg else {
        "t0": 0.75,
        "gamma": 0.35,
        "beta": 0.35,
        "alpha_imp": 0.25,
        "alpha_conflict": 0.15,
        "topn_candidates": ZK_DEMO_TOPN_PER_MODEL,
        "delta": 1e-4,
        "max_iter": ZK_DEMO_ITER_N,
        "cand_decay": 0.30,
        "min_tau_scale": 0.20,
    }
    cfg_dict["max_iter"] = ZK_DEMO_ITER_N
    cfg_dict["topn_candidates"] = ZK_DEMO_TOPN_PER_MODEL

    return {
        "input_text": english_text,
        "keywords": list(keywords),
        "model_ids": list(MODELS),
        "results": results,
        "normalized_by_model": normalized_by_model,
        "cfg_dict": cfg_dict,
        "sentence_id": "s0",
        "session_id": "streamlit-current",
        "truthfinder_payload": st.session_state.get("truthfinder_payload"),
    }


def build_all_zk_inputs(state: Dict[str, Any]) -> Dict[str, Path]:
    """基于当前 state 生成 runtime/dense/circom/witness 输入。"""
    cfg = TruthFinderConfig(**state["cfg_dict"])

    runtime_input = build_truthfinder_runtime_input_from_state(
        input_text=state["input_text"],
        sentence_id=state["sentence_id"],
        session_id=state["session_id"],
        keywords=state["keywords"],
        results=state["results"],
        cfg=cfg,
        schema_path=SCHEMA_PATH,
        normalized_by_model=state["normalized_by_model"],
        model_ids=state["model_ids"],
        truthfinder_path=TRUTHFINDER_PATH,
        normalize_path=NORMALIZE_PATH,
        app_path=APP_FILE,
    )
    RUNTIME_INPUT_PATH.write_text(json.dumps(runtime_input, ensure_ascii=False, indent=2), encoding="utf-8")

    dense_input = expand_runtime_input(runtime_input)
    DENSE_INPUT_PATH.write_text(json.dumps(dense_input, ensure_ascii=False, indent=2), encoding="utf-8")

    circom_input = prepare_circom_input_from_dense(dense_input)
    CIRCOM_INPUT_PATH.write_text(json.dumps(circom_input, ensure_ascii=False, indent=2), encoding="utf-8")

    witness_input = build_witness_input_from_circom_input(circom_input)
    WITNESS_INPUT_PATH.write_text(json.dumps(witness_input, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "runtime_input_path": RUNTIME_INPUT_PATH,
        "dense_input_path": DENSE_INPUT_PATH,
        "circom_input_path": CIRCOM_INPUT_PATH,
        "witness_input_path": WITNESS_INPUT_PATH,
    }


def generate_witness_file(witness_input_path: Path) -> Dict[str, Any]:
    """调用 generate_witness.js 生成 witness.wtns。"""
    required = [WITNESS_JS_PATH, WASM_PATH, witness_input_path]
    for p in required:
        if not p.exists():
            return {"success": False, "message": f"缺少文件: {p}", "witness_path": str(WITNESS_PATH)}

    ok, msg = run_command(
        ["node", str(WITNESS_JS_PATH), str(WASM_PATH), str(witness_input_path), str(WITNESS_PATH)],
        cwd=PROJECT_ROOT,
    )
    return {"success": ok, "message": msg, "witness_path": str(WITNESS_PATH)}


def generate_proof_from_witness(witness_path: Path) -> Dict[str, Any]:
    """调用 snarkjs groth16 prove 生成 proof/public。"""
    required = [SNARKJS_CLI_PATH, FINAL_ZKEY_PATH, witness_path]
    for p in required:
        if not p.exists():
            return {
                "success": False,
                "message": f"缺少文件: {p}",
                "proof_path": str(PROOF_PATH),
                "public_path": str(PUBLIC_PATH),
            }

    ok, msg = run_command(
        [
            "node",
            str(SNARKJS_CLI_PATH),
            "groth16",
            "prove",
            str(FINAL_ZKEY_PATH),
            str(witness_path),
            str(PROOF_PATH),
            str(PUBLIC_PATH),
        ],
        cwd=PROJECT_ROOT,
    )
    return {"success": ok, "message": msg, "proof_path": str(PROOF_PATH), "public_path": str(PUBLIC_PATH)}


def verify_generated_proof(proof_path: Path, public_path: Path) -> Dict[str, Any]:
    """调用 snarkjs groth16 verify 验证 proof。"""
    required = [SNARKJS_CLI_PATH, VERIFICATION_KEY_PATH, proof_path, public_path]
    for p in required:
        if not p.exists():
            return {"success": False, "message": f"缺少文件: {p}"}

    ok, msg = run_command(
        ["node", str(SNARKJS_CLI_PATH), "groth16", "verify", str(VERIFICATION_KEY_PATH), str(public_path), str(proof_path)],
        cwd=PROJECT_ROOT,
    )
    return {"success": ok, "message": msg}


def load_public_summary(public_path: Path) -> Dict[str, Any]:
    """解析 public.json 为结构化摘要。"""
    raw = json.loads(public_path.read_text(encoding="utf-8"))
    expected_len = 2 + ZK_DEMO_K_MAX
    if not isinstance(raw, list) or len(raw) < expected_len:
        raise ValueError(
            f"public.json 格式异常：期望长度>={expected_len}，实际={len(raw) if isinstance(raw, list) else type(raw)}"
        )

    vals = [int(str(x)) for x in raw]
    return {
        "best_model_idx": vals[0],
        "best_model_score_q16": vals[1],
        "winning_fact_idx_by_object": vals[2 : 2 + ZK_DEMO_K_MAX],
        "public_raw": raw,
    }


def _tail_text(msg: str, max_chars: int = 2000) -> str:
    msg = msg or ""
    if len(msg) <= max_chars:
        return msg
    return msg[-max_chars:]


def reset_zk_state(remove_artifacts: bool = False) -> None:
    """统一重置 zk 相关会话状态。"""
    st.session_state["zk_proof_generated"] = False
    st.session_state["zk_verified"] = False
    st.session_state["zk_public_summary"] = None
    st.session_state["zk_proof_message"] = ""
    st.session_state["zk_verify_message"] = ""
    st.session_state["zk_proof_path"] = ""
    st.session_state["zk_public_path"] = ""
    st.session_state["zk_stage_status"] = "未生成"
    st.session_state["zk_verify_status"] = "未验证"
    st.session_state["zk_last_runtime_sec"] = 0.0
    st.session_state["zk_last_logs"] = {}

    if remove_artifacts:
        for path in [
            RUNTIME_INPUT_PATH,
            DENSE_INPUT_PATH,
            CIRCOM_INPUT_PATH,
            WITNESS_INPUT_PATH,
            WITNESS_PATH,
            PROOF_PATH,
            PUBLIC_PATH,
        ]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

def collect_current_zk_state() -> Dict[str, Any]:
    english_text = (st.session_state.get("last_english") or "").strip()
    keywords = st.session_state.get("fixed_keywords") or []
    results = st.session_state.get("results") or {}

    if not english_text:
        raise ValueError("当前英文文本为空，请先输入英文文本并完成多模型分析。")
    if not keywords:
        raise ValueError("当前关键词为空，请先提取并确认关键词。")
    if not results:
        raise ValueError("当前模型结果为空，请先点击“开始多模型分析”。")

    normalized_by_model: Dict[str, Dict[str, list]] = {}
    for model_name in MODELS:
        model_payload = results.get(model_name, {}) or {}
        rows = model_payload.get("keywords", []) or []
        kw2meaning = {str(r.get("keyword", "")).strip(): str(r.get("meaning_zh", "")).strip() for r in rows}
        normalized_by_model[model_name] = {}
        for kw in keywords:
            cands = normalize_meaning_zh_soft(
                kw2meaning.get(kw, ""),
                top_n=ZK_DEMO_TOPN_PER_MODEL,
            )
            normalized_by_model[model_name][kw] = [c for c in cands if c and c.strip()]

    payload_cfg = ((st.session_state.get("truthfinder_payload") or {}).get("cfg_dict") or {})
    cfg_dict = dict(payload_cfg) if payload_cfg else {
        "t0": 0.75,
        "gamma": 0.35,
        "beta": 0.35,
        "alpha_imp": 0.25,
        "alpha_conflict": 0.15,
        "topn_candidates": ZK_DEMO_TOPN_PER_MODEL,
        "delta": 1e-4,
        "max_iter": ZK_DEMO_ITER_N,
        "cand_decay": 0.30,
        "min_tau_scale": 0.20,
    }
    cfg_dict["max_iter"] = ZK_DEMO_ITER_N
    cfg_dict["topn_candidates"] = ZK_DEMO_TOPN_PER_MODEL

    return {
        "input_text": english_text,
        "keywords": list(keywords),
        "model_ids": list(MODELS),
        "results": results,
        "normalized_by_model": normalized_by_model,
        "cfg_dict": cfg_dict,
        "sentence_id": "s0",
        "session_id": "streamlit-current",
        "truthfinder_payload": st.session_state.get("truthfinder_payload"),
    }


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail_text(msg: str, max_chars: int = 2000) -> str:
    msg = msg or ""
    if len(msg) <= max_chars:
        return msg
    return msg[-max_chars:]


def _short_hash(value: str, keep: int = 16) -> str:
    value = str(value or "")
    if len(value) <= keep:
        return value
    return f"{value[:keep]}..."


def _artifact_status(path: Path, present_label: str = "已生成", missing_label: str = "缺失") -> str:
    return present_label if path.exists() else missing_label


def _q16_to_float_text(value: Any) -> str:
    try:
        return f"{int(value) / 65536:.4f}"
    except Exception:
        return str(value)


def _status_tone(status: str) -> str:
    if status in {"已生成", "验证通过", "通过"}:
        return "success"
    if status in {"运行中", "生成中", "验证中"}:
        return "info"
    if status in {"失败", "验证失败"}:
        return "error"
    if status in {"需检查", "警告"}:
        return "warning"
    return "pending"


def _build_step_error(step: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"step": step, "message": message, "details": details or {}}


def build_all_zk_inputs(state: Dict[str, Any]) -> Dict[str, Path]:
    cfg = TruthFinderConfig(**state["cfg_dict"])

    runtime_input = build_truthfinder_runtime_input_from_state(
        input_text=state["input_text"],
        sentence_id=state["sentence_id"],
        session_id=state["session_id"],
        keywords=state["keywords"],
        results=state["results"],
        cfg=cfg,
        schema_path=SCHEMA_PATH,
        normalized_by_model=state["normalized_by_model"],
        model_ids=state["model_ids"],
        truthfinder_path=TRUTHFINDER_PATH,
        normalize_path=NORMALIZE_PATH,
        app_path=APP_FILE,
    )
    RUNTIME_INPUT_PATH.write_text(json.dumps(runtime_input, ensure_ascii=False, indent=2), encoding="utf-8")

    dense_input = expand_runtime_input(runtime_input)
    DENSE_INPUT_PATH.write_text(json.dumps(dense_input, ensure_ascii=False, indent=2), encoding="utf-8")

    circom_input = prepare_circom_input_from_dense(dense_input)
    CIRCOM_INPUT_PATH.write_text(json.dumps(circom_input, ensure_ascii=False, indent=2), encoding="utf-8")

    witness_input = build_witness_input_from_circom_input(circom_input)
    if "support_flat" in witness_input:
        raise ValueError("新版 witness input 不应包含 support_flat。")
    if "cand_decay" in witness_input:
        raise ValueError("新版 witness input 不应包含 cand_decay。")
    WITNESS_INPUT_PATH.write_text(json.dumps(witness_input, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "runtime_input_path": RUNTIME_INPUT_PATH,
        "dense_input_path": DENSE_INPUT_PATH,
        "circom_input_path": CIRCOM_INPUT_PATH,
        "witness_input_path": WITNESS_INPUT_PATH,
    }


def generate_witness_file(witness_input_path: Path) -> Dict[str, Any]:
    required = [WITNESS_JS_PATH, WASM_PATH, witness_input_path]
    for p in required:
        if not p.exists():
            return {
                "success": False,
                "message": f"缺少文件: {p}",
                "command": "",
                "returncode": None,
                "stderr": "",
                "stdout": "",
                "witness_path": str(WITNESS_PATH),
            }

    result = run_command_details(
        ["node", str(WITNESS_JS_PATH), str(WASM_PATH), str(witness_input_path), str(WITNESS_PATH)],
        cwd=PROJECT_ROOT,
    )
    result["witness_path"] = str(WITNESS_PATH)
    return result


def generate_proof_from_witness(witness_path: Path) -> Dict[str, Any]:
    required = [SNARKJS_CLI_PATH, FINAL_ZKEY_PATH, witness_path]
    for p in required:
        if not p.exists():
            return {
                "success": False,
                "message": f"缺少文件: {p}",
                "command": "",
                "returncode": None,
                "stderr": "",
                "stdout": "",
                "proof_path": str(PROOF_PATH),
                "public_path": str(PUBLIC_PATH),
            }

    result = run_command_details(
        [
            "node",
            str(SNARKJS_CLI_PATH),
            "groth16",
            "prove",
            str(FINAL_ZKEY_PATH),
            str(witness_path),
            str(PROOF_PATH),
            str(PUBLIC_PATH),
        ],
        cwd=PROJECT_ROOT,
    )
    result["proof_path"] = str(PROOF_PATH)
    result["public_path"] = str(PUBLIC_PATH)
    return result


def verify_generated_proof(proof_path: Path, public_path: Path) -> Dict[str, Any]:
    required = [SNARKJS_CLI_PATH, VERIFICATION_KEY_PATH, proof_path, public_path]
    for p in required:
        if not p.exists():
            return {
                "success": False,
                "message": f"缺少文件: {p}",
                "command": "",
                "returncode": None,
                "stderr": "",
                "stdout": "",
            }

    return run_command_details(
        ["node", str(SNARKJS_CLI_PATH), "groth16", "verify", str(VERIFICATION_KEY_PATH), str(public_path), str(proof_path)],
        cwd=PROJECT_ROOT,
    )


def generate_reference_output(circom_input_path: Path) -> Dict[str, Any]:
    if not circom_input_path.exists():
        return {"success": False, "message": f"缺少文件: {circom_input_path}", "reference_path": str(REFERENCE_OUTPUT_PATH)}
    try:
        result = run_truthfinder_circuit_ref_from_file(circom_input_path)
        REFERENCE_OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "success": True,
            "message": f"已生成 reference output: {REFERENCE_OUTPUT_PATH}",
            "reference_path": str(REFERENCE_OUTPUT_PATH),
            "reference_output": result,
        }
    except Exception as ex:
        return {
            "success": False,
            "message": f"{type(ex).__name__}: {ex}",
            "reference_path": str(REFERENCE_OUTPUT_PATH),
        }


def load_public_summary(public_path: Path) -> Dict[str, Any]:
    raw = load_json_file(public_path)
    expected_len = 2 + ZK_DEMO_K_MAX
    if not isinstance(raw, list) or len(raw) < expected_len:
        raise ValueError(
            f"public.json 格式异常：期望长度>={expected_len}，实际={len(raw) if isinstance(raw, list) else type(raw)}"
        )

    vals = [int(str(x)) for x in raw]
    return {
        "best_model_idx": vals[0],
        "best_model_score_q16": vals[1],
        "winning_fact_idx_by_object": vals[2 : 2 + ZK_DEMO_K_MAX],
        "public_raw": raw,
    }


def build_zk_compare_rows() -> List[Dict[str, Any]]:
    dense_input = load_json_file(DENSE_INPUT_PATH)
    objects_dense = dense_input.get("objects_dense", []) or []
    tf_payload = st.session_state.get("truthfinder_payload") or {}
    truth_rows = tf_payload.get("truth_rows", []) or []
    summary = st.session_state.get("zk_public_summary") or {}
    winners = summary.get("winning_fact_idx_by_object", []) or []

    rows: List[Dict[str, Any]] = []
    for obj in objects_dense:
        if not obj.get("is_effective"):
            continue
        idx = int(obj.get("o", 0))
        facts = obj.get("facts_padded", []) or []
        fact_count = int(obj.get("fact_count", 0))
        winning_idx = winners[idx] if idx < len(winners) else None
        zk_candidate = "候选文本映射待解析"
        if isinstance(winning_idx, int) and 0 <= winning_idx < fact_count and winning_idx < len(facts):
            zk_candidate = facts[winning_idx] or "候选文本映射待解析"

        truth_row = truth_rows[idx] if idx < len(truth_rows) else {}
        frontend_truth = [str(item).strip() for item in (truth_row.get("truth", []) or []) if str(item).strip()]
        frontend_display = " / ".join(frontend_truth) if frontend_truth else "(空)"
        consistent = "通过" if zk_candidate in frontend_truth else "需检查"

        rows.append(
            {
                "序号": idx + 1,
                "关键词": str(obj.get("keyword", "")),
                "前端可信候选": frontend_display,
                "ZK 证明候选": zk_candidate,
                "一致性": consistent,
            }
        )
    return rows


def reset_zk_state(remove_artifacts: bool = False) -> None:
    st.session_state["zk_proof_generated"] = False
    st.session_state["zk_verified"] = False
    st.session_state["zk_public_summary"] = None
    st.session_state["zk_reference_output"] = None
    st.session_state["zk_compare_rows"] = []
    st.session_state["zk_proof_message"] = ""
    st.session_state["zk_verify_message"] = ""
    st.session_state["zk_proof_path"] = ""
    st.session_state["zk_public_path"] = ""
    st.session_state["zk_reference_path"] = ""
    st.session_state["zk_stage_status"] = "未生成"
    st.session_state["zk_verify_status"] = "未验证"
    st.session_state["zk_last_runtime_sec"] = 0.0
    st.session_state["zk_last_logs"] = {}
    st.session_state["zk_error"] = None
    st.session_state["zk_status_cards"] = {
        "inputs": "未生成",
        "witness": "未生成",
        "proof": "未生成",
        "verify": "未验证",
    }

    if remove_artifacts:
        for path in [
            RUNTIME_INPUT_PATH,
            DENSE_INPUT_PATH,
            CIRCOM_INPUT_PATH,
            WITNESS_INPUT_PATH,
            WITNESS_PATH,
            PROOF_PATH,
            PUBLIC_PATH,
            REFERENCE_OUTPUT_PATH,
        ]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass


MODEL_LABELS = {
    "qwen2.5:7b-instruct-q4_K_M": "Qwen2.5",
    "mistral:7b-instruct-v0.3-q5_0": "Mistral",
    "gemma2:9b-instruct-q4_K_M": "Gemma2",
    "koesn/mistral-7b-instruct:Q4_0": "Koesn/Mistral",
}


def model_ui_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>基于 TruthFinder 的多模型翻译可信聚合系统</h1>
            <p>输入英文文本 → 确认关键词 → 运行多模型分析 → 查看可信译法与证明结果</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_step_header(step_no: int, title: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="step-title">Step {step_no}</div>
            <h3 style="margin: 0.25rem 0 0.1rem 0; color: #0F172A;">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def extract_keyword_candidates_for_ui(english_text: str) -> list[str]:
    ranked = _rank_keyword_candidates(english_text)
    return [item["word"] for item in ranked[:50]]


def render_light_table(rows: list[dict], columns: list[str]) -> None:
    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")

    st.markdown(
        f"""
        <div class="light-table-wrap">
            <table class="light-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_candidate_replace_panel(row_idx: int) -> None:
    candidate_pool = [word for word in (st.session_state.get("keyword_candidate_pool") or []) if str(word).strip()]
    current_kw = st.session_state.get("selected_keywords", [])[row_idx]
    st.markdown('<div class="candidate-menu">', unsafe_allow_html=True)
    st.markdown('<div class="candidate-menu-title">可替换候选词</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="candidate-menu-caption">当前关键词：{html.escape(current_kw)}</div>',
        unsafe_allow_html=True,
    )

    if not candidate_pool:
        st.markdown('<div class="candidate-empty">暂无可替换候选词。</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="candidate-menu-list">', unsafe_allow_html=True)
        for candidate_idx, candidate in enumerate(candidate_pool):
            if st.button(
                candidate,
                key=f"replace_kw_{row_idx}_{candidate_idx}",
                type="secondary",
                use_container_width=True,
            ):
                updated_keywords = list(st.session_state.get("selected_keywords", []))
                updated_keywords[row_idx] = candidate
                st.session_state["selected_keywords"] = updated_keywords
                st.session_state["custom_keywords"] = " ".join(updated_keywords)
                st.session_state["truthfinder_payload"] = None
                st.session_state["results"] = None
                st.session_state["times"] = None
                st.session_state["fixed_keywords"] = None
                st.session_state["show_norm"] = False
                reset_zk_state(remove_artifacts=True)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_translation_summary(results: Dict[str, Any], times: Dict[str, float]) -> None:
    with st.expander("查看四模型整句翻译汇总", expanded=False):
        for model_name in MODELS:
            payload = results.get(model_name, {}) or {}
            if payload.get("error"):
                status = "调用异常"
            elif payload.get("ok", False):
                status = "已完成"
            else:
                status = "结果不完整"

            with st.expander(f"{model_ui_name(model_name)} | {status} | {times.get(model_name, 0.0):.2f} 秒", expanded=False):
                translation = (payload.get("translation_zh") or "").strip()
                if translation:
                    st.write(translation)
                else:
                    st.warning("该模型未返回整句译文。")


def _badge_tone(status: str) -> str:
    if status in {"已完成", "已生成", "验证通过"}:
        return "is-success"
    if status in {"失败", "验证失败"}:
        return "is-error"
    if status in {"生成中", "验证中"}:
        return "is-warning"
    return "is-pending"


def _badge_html(label: str, status: str) -> str:
    return f'<span class="status-badge {_badge_tone(status)}">{label}：{status}</span>'


def render_step_badges() -> None:
    input_status = "已完成" if (st.session_state.get("last_english") or "").strip() else "待输入"
    model_status = "已完成" if st.session_state.get("results") else "待运行"
    norm_status = "已完成" if st.session_state.get("show_norm") else "待运行"
    tf_status = "已完成" if st.session_state.get("truthfinder_payload") else "待运行"

    if st.session_state.get("zk_verified"):
        zk_status = "验证通过"
    elif st.session_state.get("zk_verify_status") == "验证失败":
        zk_status = "验证失败"
    elif st.session_state.get("zk_proof_generated"):
        zk_status = "已生成"
    else:
        zk_status = st.session_state.get("zk_stage_status", "未生成")

    st.markdown(
        f"""
        <div class="section-card">
            <div class="small-caption">流程状态</div>
            <div style="margin-top: 0.5rem;">
                {_badge_html("输入文本", input_status)}
                {_badge_html("模型调用", model_status)}
                {_badge_html("归一化", norm_status)}
                {_badge_html("TruthFinder", tf_status)}
                {_badge_html("ZK 证明", zk_status)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_overview(results: Dict[str, Any], times: Dict[str, float], fixed_keywords: list) -> None:
    rows = []
    for model_name in MODELS:
        payload = results.get(model_name, {}) or {}
        if payload.get("error"):
            status = "调用异常"
        elif payload.get("ok", False):
            status = "已完成"
        else:
            status = "结果不完整"

        rows.append(
            {
                "模型": model_ui_name(model_name),
                "状态": status,
                "耗时（秒）": round(times.get(model_name, 0.0), 2),
                "关键词数量": len(payload.get("keywords", []) or fixed_keywords or []),
                "整句翻译是否为空": "是" if not (payload.get("translation_zh", "") or "").strip() else "否",
            }
        )

    render_light_table(rows, ["模型", "状态", "耗时（秒）", "关键词数量", "整句翻译是否为空"])


def render_model_detail_tabs(results: Dict[str, Any], times: Dict[str, float]) -> None:
    tabs = st.tabs([model_ui_name(model_name) for model_name in MODELS])
    for tab, model_name in zip(tabs, MODELS):
        payload = results.get(model_name, {}) or {}
        with tab:
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="small-caption">模型标识</div>
                        <div>{model_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with metric_cols[1]:
                status = "调用异常" if payload.get("error") else ("已完成" if payload.get("ok", False) else "结果不完整")
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="small-caption">运行状态</div>
                        <div>{status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with metric_cols[2]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="small-caption">耗时</div>
                        <div>{times.get(model_name, 0.0):.2f} 秒</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if payload.get("error"):
                st.warning(f"模型调用异常：{payload['error']}")
            elif not payload.get("ok", True):
                st.warning("该模型输出存在空缺释义，系统已尝试自动补全。")

            st.markdown("#### 关键词释义")
            render_light_table(
                [
                    {"序号": i + 1, "关键词": row["keyword"], "中文释义": row["meaning_zh"]}
                    for i, row in enumerate(payload.get("keywords", []))
                ],
                ["序号", "关键词", "中文释义"],
            )

            st.markdown("#### 全文译文")
            translation = (payload.get("translation_zh") or "").strip()
            if translation:
                st.write(translation)
            else:
                st.warning("该模型未返回整句译文。")


def render_truthfinder_results(tf_payload: Dict[str, Any]) -> None:
    rank = tf_payload["rank"]
    truth_rows = tf_payload["truth_rows"]
    cand_map = tf_payload["cand_map"]
    sentence_id = tf_payload["sentence_id"]

    def _join_truth(truth_list: list) -> str:
        return " / ".join(truth_list) if truth_list else "(空)"

    st.markdown("#### 模型来源可信度估计")
    st.caption("该分数表示模型在关键词候选译法层面的相对可信度，不直接代表整句翻译质量。")
    render_light_table(
        [{"模型": model_ui_name(model_name), "相对可信度": round(score, 4)} for model_name, score in rank],
        ["模型", "相对可信度"],
    )

    st.markdown("#### 关键词级可信译法发现结果")
    render_light_table(
        [
            {
                "序号": i + 1,
                "关键词": row["keyword"],
                "可信译法候选": _join_truth(row["truth"]),
                "置信度": " / ".join([f"{conf:.4f}" for conf in row["conf"]]),
            }
            for i, row in enumerate(truth_rows)
        ],
        ["序号", "关键词", "可信译法候选", "置信度"],
    )

    with st.expander("查看候选细节", expanded=False):
        render_light_table(
            [
                {
                    "关键词": row["keyword"],
                    "候选集合（调试）": " / ".join(cand_map.get((sentence_id, row["keyword"]), [])),
                }
                for row in truth_rows
            ],
            ["关键词", "候选集合（调试）"],
        )


def render_zk_panel() -> None:
    st.markdown("#### Groth16 零知识证明验证")
    st.caption("零知识证明用于验证关键词级 TruthFinder 聚合计算是否按照预设电路执行。")

    status_cols = st.columns(3)
    with status_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-caption">证明状态</div>
                <div>{st.session_state.get("zk_stage_status", "未生成")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-caption">验证状态</div>
                <div>{st.session_state.get("zk_verify_status", "未验证")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_cols[2]:
        st.markdown(
            """
            <div class="metric-card">
                <div class="small-caption">证明系统</div>
                <div>Groth16 / BN128</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-card">
            <div class="muted-text">
                当前关键词级聚合结果可以进一步生成 Groth16 零知识证明，并验证其是否来自预定义电路计算过程。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("生成 Groth16 证明", use_container_width=True, key="btn_zk_prove", type="primary"):
            if not st.session_state.get("truthfinder_payload"):
                st.warning("请先完成 TruthFinder 聚合计算。")
            else:
                reset_zk_state(remove_artifacts=False)
                st.session_state["zk_stage_status"] = "生成中"

                start_ts = time.time()
                with st.status("正在生成 Groth16 证明…", expanded=True) as zk_status:
                    try:
                        st.write("1/4 正在收集当前 zk state…")
                        state = collect_current_zk_state()

                        st.write("2/4 正在生成 zk 输入链路（runtime/dense/circom/witness_input）…")
                        paths = build_all_zk_inputs(state)
                        st.session_state["zk_last_logs"]["inputs"] = {k: str(v) for k, v in paths.items()}

                        st.write("3/4 正在生成 witness…")
                        witness_ret = generate_witness_file(paths["witness_input_path"])
                        st.session_state["zk_last_logs"]["witness"] = witness_ret["message"]
                        if not witness_ret["success"]:
                            raise RuntimeError(f"witness 生成失败: {witness_ret['message']}")

                        st.write("4/4 正在生成 proof/public…")
                        prove_ret = generate_proof_from_witness(Path(witness_ret["witness_path"]))
                        st.session_state["zk_last_logs"]["prove"] = prove_ret["message"]
                        if not prove_ret["success"]:
                            raise RuntimeError(f"proof 生成失败: {prove_ret['message']}")

                        summary = load_public_summary(Path(prove_ret["public_path"]))
                        elapsed = time.time() - start_ts

                        st.session_state["zk_proof_generated"] = True
                        st.session_state["zk_stage_status"] = "已生成"
                        st.session_state["zk_proof_message"] = "证明生成完成"
                        st.session_state["zk_public_summary"] = summary
                        st.session_state["zk_proof_path"] = prove_ret["proof_path"]
                        st.session_state["zk_public_path"] = prove_ret["public_path"]
                        st.session_state["zk_last_runtime_sec"] = elapsed

                        zk_status.update(label="证明生成完成", state="complete")
                    except Exception as ex:
                        elapsed = time.time() - start_ts
                        st.session_state["zk_stage_status"] = "失败"
                        st.session_state["zk_proof_message"] = f"{type(ex).__name__}: {ex}"
                        st.session_state["zk_last_runtime_sec"] = elapsed
                        zk_status.update(label="证明生成失败", state="error")

    with btn_col2:
        if st.button("验证证明", use_container_width=True, key="btn_zk_verify", type="primary"):
            proof_path = Path(st.session_state.get("zk_proof_path", "") or "")
            public_path = Path(st.session_state.get("zk_public_path", "") or "")
            if not proof_path.exists() or not public_path.exists():
                st.warning("请先生成证明。")
            else:
                st.session_state["zk_verify_status"] = "验证中"
                with st.status("正在验证 proof…", expanded=True) as verify_status:
                    ret = verify_generated_proof(proof_path, public_path)
                    st.session_state["zk_last_logs"]["verify"] = ret["message"]
                    st.session_state["zk_verify_message"] = ret["message"]
                    if ret["success"]:
                        st.session_state["zk_verified"] = True
                        st.session_state["zk_verify_status"] = "验证通过"
                        verify_status.update(label="验证通过", state="complete")
                    else:
                        st.session_state["zk_verified"] = False
                        st.session_state["zk_verify_status"] = "验证失败"
                        verify_status.update(label="验证失败", state="error")

    if st.session_state.get("zk_proof_message"):
        if st.session_state.get("zk_proof_generated"):
            st.success(st.session_state["zk_proof_message"])
        else:
            st.error(st.session_state["zk_proof_message"])

    if st.session_state.get("zk_verify_message"):
        if st.session_state.get("zk_verified"):
            st.success("验证通过")
        else:
            st.error("验证失败")

    summary = st.session_state.get("zk_public_summary")
    if summary:
        st.markdown("#### public.json 摘要")
        st.write(
            {
                "best_model_idx": summary.get("best_model_idx"),
                "best_model_score_q16": summary.get("best_model_score_q16"),
                "winning_fact_idx_by_object": summary.get("winning_fact_idx_by_object"),
            }
        )

    with st.expander("查看技术细节", expanded=False):
        st.markdown(f"- proof 路径: `{st.session_state.get('zk_proof_path', '')}`")
        st.markdown(f"- public 路径: `{st.session_state.get('zk_public_path', '')}`")
        st.markdown(f"- verification key: `{VERIFICATION_KEY_PATH}`")
        st.markdown(f"- snarkjs cli: `{SNARKJS_CLI_PATH}`")
        st.markdown(f"- wasm: `{WASM_PATH}`")
        st.markdown(f"- 运行耗时: {st.session_state.get('zk_last_runtime_sec', 0.0):.3f} 秒")

        logs = st.session_state.get("zk_last_logs", {}) or {}
        if logs.get("inputs"):
            st.markdown("**输入文件路径：**")
            st.code(json.dumps(logs["inputs"], ensure_ascii=False, indent=2))

        if logs:
            st.markdown("**命令输出日志：**")
            for name in ["witness", "prove", "verify"]:
                if name in logs:
                    st.markdown(f"- {name}")
                    st.code(_tail_text(str(logs[name]), max_chars=1500))

        if summary and "public_raw" in summary:
            st.markdown("**public.json 原文：**")
            st.code(json.dumps(summary["public_raw"], ensure_ascii=False, indent=2))


# ===== Streamlit 页面 =====
st.set_page_config(page_title="基于 TruthFinder 的多模型翻译可信聚合系统", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: #F5F7FB;
        color: #334155;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    section.main > div {
        padding-top: 0rem !important;
    }
    .block-container {
        max-width: 1560px;
        padding-top: 1rem !important;
        padding-bottom: 4rem;
    }
    .hero-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 22px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
        text-align: center;
    }
    .hero-card h1 {
        margin: 0;
        color: #1E293B;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .hero-card p {
        margin: 0.55rem 0 0 0;
        color: #334155;
        line-height: 1.45;
        font-size: 0.94rem;
    }
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        margin: 0.6rem 0 1rem 0;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        min-height: 92px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
    }
    .zk-status-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        min-height: 96px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        border-left-width: 4px;
    }
    .zk-status-card.tone-success {
        border-left-color: #16A34A;
    }
    .zk-status-card.tone-info {
        border-left-color: #2563EB;
    }
    .zk-status-card.tone-error {
        border-left-color: #DC2626;
    }
    .zk-status-card.tone-warning {
        border-left-color: #F59E0B;
    }
    .zk-status-card.tone-pending {
        border-left-color: #9CA3AF;
    }
    .zk-status-label {
        color: #6B7280;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .zk-status-value {
        color: #1F2937;
        font-size: 1.08rem;
        font-weight: 700;
        margin-top: 0.45rem;
    }
    .muted-text {
        color: #64748B;
        line-height: 1.65;
    }
    .status-badge {
        display: inline-block;
        margin: 0.2rem 0.45rem 0.2rem 0;
        padding: 0.34rem 0.75rem;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 600;
        border: 1px solid #E2E8F0;
    }
    .status-badge.is-info {
        background: rgba(37, 99, 235, 0.08);
        color: #2563EB;
        border-color: rgba(37, 99, 235, 0.18);
    }
    .status-badge.is-success {
        background: rgba(34, 197, 94, 0.12);
        color: #15803D;
        border-color: rgba(34, 197, 94, 0.35);
    }
    .status-badge.is-warning {
        background: rgba(245, 158, 11, 0.12);
        color: #B45309;
        border-color: rgba(245, 158, 11, 0.35);
    }
    .status-badge.is-error {
        background: rgba(239, 68, 68, 0.12);
        color: #B91C1C;
        border-color: rgba(239, 68, 68, 0.35);
    }
    .status-badge.is-pending {
        background: rgba(148, 163, 184, 0.10);
        color: #64748B;
        border-color: rgba(148, 163, 184, 0.26);
    }
    .step-title {
        color: #2563EB;
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .keyword-row {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.45rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .keyword-table-head {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.55rem 0.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.45rem;
    }
    .candidate-table-head {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.45rem 0.7rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.35rem;
    }
    .candidate-row-cell {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.42rem 0.7rem;
        color: #1E293B;
        margin-bottom: 0.3rem;
        min-height: 34px;
        display: flex;
        align-items: center;
    }
    .keyword-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        margin: 0;
        max-height: 260px;
        overflow-y: auto;
    }
    .candidate-menu {
        background: #FFFFFF;
        color: #1E293B;
        width: 100%;
    }
    .candidate-menu-title {
        color: #1E293B;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .candidate-menu-caption {
        color: #64748B;
        font-size: 0.78rem;
        margin-bottom: 0.7rem;
    }
    .candidate-menu-list {
        max-height: 220px;
        overflow-y: auto;
        padding-right: 0.15rem;
    }
    .candidate-empty {
        color: #64748B;
        font-size: 0.88rem;
        padding: 0.15rem 0 0.1rem 0;
    }
    .small-caption {
        color: #64748B;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .light-table-wrap {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }
    .light-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
        color: #1E293B;
    }
    .light-table thead tr {
        background: #F1F5F9;
    }
    .light-table th,
    .light-table td {
        padding: 0.78rem 0.9rem;
        border-bottom: 1px solid #E2E8F0;
        text-align: left;
        vertical-align: top;
        font-size: 0.95rem;
        color: #1E293B;
    }
    .light-table tbody tr:hover {
        background: #F8FAFC;
    }
    .light-table tbody tr:last-child td {
        border-bottom: none;
    }
    .inline-badge {
        display: inline-block;
        min-width: 64px;
        padding: 0.22rem 0.58rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        text-align: center;
    }
    .inline-badge.ok {
        background: rgba(22, 163, 74, 0.12);
        color: #15803D;
    }
    .inline-badge.warn {
        background: rgba(220, 38, 38, 0.10);
        color: #B91C1C;
    }
    .zk-file-preview {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        max-height: 260px;
        overflow: auto;
        margin-top: 0.5rem;
    }
    .zk-file-preview pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: #0F172A !important;
        background: transparent !important;
        font-size: 0.86rem;
        line-height: 1.55;
        font-family: Consolas, "Courier New", monospace;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #334155;
    }
    div.stButton > button[kind="primary"] {
        background: #2563EB;
        color: white;
        border: 1px solid #2563EB;
        border-radius: 14px;
        height: 3rem;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.12);
    }
    div.stButton > button[kind="primary"]:hover {
        border-color: #1D4ED8;
        color: white;
    }
    div.stButton > button[kind="secondary"] {
        background: #FFFFFF;
        color: #1E293B;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        height: 2.15rem;
        font-weight: 600;
        box-shadow: none;
        width: auto;
        padding: 0 0.85rem;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #EFF6FF;
        border-color: #2563EB;
        color: #1D4ED8;
    }
    div[data-testid="stDownloadButton"] > button {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        min-height: 2.35rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: #EFF6FF !important;
        color: #1D4ED8 !important;
        border-color: #2563EB !important;
    }
    div[data-testid="stDownloadButton"] > button:active {
        background: #DBEAFE !important;
        color: #1D4ED8 !important;
        border-color: #2563EB !important;
    }
    div[data-testid="stDownloadButton"] > button:disabled {
        background: #F8FAFC !important;
        color: #94A3B8 !important;
        border-color: #E2E8F0 !important;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 0.35rem;
    }
    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        color: #1E293B !important;
    }
    div[data-testid="stExpander"] details {
        background: #FFFFFF !important;
        color: #1E293B !important;
    }
    div[data-testid="stExpander"] summary {
        background: #FFFFFF !important;
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] * {
        color: #1E293B !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    div[data-testid="stPopover"] {
        background: transparent !important;
        color: #1E293B !important;
    }
    div[data-testid="stPopover"] * {
        color: #1E293B !important;
    }
    div[data-testid="stPopover"] button {
        min-width: 110px !important;
        min-height: 2.35rem !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: none !important;
    }
    div[data-testid="stPopover"] button:hover {
        background: #EFF6FF !important;
        color: #1D4ED8 !important;
        border-color: #2563EB !important;
    }
    div[data-testid="stPopoverBody"] {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.16) !important;
        width: 340px !important;
        max-width: 340px !important;
        padding: 0.25rem !important;
    }
    div[data-testid="stPopoverBody"] * {
        color: #1E293B !important;
    }
    div[data-baseweb="popover"] {
        background: transparent !important;
    }
    div[data-baseweb="popover"] > div {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.16) !important;
    }
    div[data-baseweb="layer"] {
        background: transparent !important;
    }
    div[data-baseweb="layer"] > div {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.16) !important;
    }
    div[role="dialog"] {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.16) !important;
    }
    div[data-testid="stPopoverBody"] div.stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        min-height: 38px !important;
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.55rem 0.75rem !important;
        margin: 0 0 0.38rem 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="stPopoverBody"] div.stButton > button[kind="secondary"]:hover {
        background: #EFF6FF !important;
        color: #1D4ED8 !important;
        border-color: #2563EB !important;
    }
    div[data-testid="stPopoverBody"] input,
    div[data-testid="stPopoverBody"] textarea {
        background: #FFFFFF !important;
        color: #0F172A !important;
        caret-color: #2563EB !important;
        border: 1px solid #CBD5E1 !important;
    }
    div[data-testid="stPopoverBody"] table,
    div[data-testid="stPopoverBody"] th,
    div[data-testid="stPopoverBody"] td {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border-color: #E2E8F0 !important;
    }
    div[data-testid="stPopoverBody"] th {
        background: #F1F5F9 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stPopoverBody"] .stMarkdown {
        background: #FFFFFF !important;
        color: #1E293B !important;
    }
    textarea,
    input {
        background: #FFFFFF !important;
        color: #111827 !important;
        caret-color: #111827 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }
    textarea:focus,
    input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
        outline: none !important;
    }
    textarea::placeholder,
    input::placeholder {
        color: #94A3B8 !important;
        opacity: 1 !important;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: #111827 !important;
        caret-color: #111827 !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 14px;
    }
    div[data-testid="stDataFrame"] *,
    div[data-testid="stTable"] *,
    div[data-testid="stExpander"] * {
        color: #334155 !important;
    }
    div[data-testid="stDataFrame"] [role="grid"],
    div[data-testid="stDataFrame"] canvas {
        background: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 初始化会话状态
if "results" not in st.session_state:
    st.session_state["results"] = None
if "times" not in st.session_state:
    st.session_state["times"] = None
if "fixed_keywords" not in st.session_state:
    st.session_state["fixed_keywords"] = None
if "last_english" not in st.session_state:
    st.session_state["last_english"] = ""
if "last_k" not in st.session_state:
    st.session_state["last_k"] = 6
if "custom_keywords" not in st.session_state:
    st.session_state["custom_keywords"] = ""
if "keyword_editor_rows" not in st.session_state:
    st.session_state["keyword_editor_rows"] = []
if "keywords_extracted" not in st.session_state:
    st.session_state["keywords_extracted"] = False
if "selected_keywords" not in st.session_state:
    st.session_state["selected_keywords"] = []
if "keyword_candidate_pool" not in st.session_state:
    st.session_state["keyword_candidate_pool"] = []
if "active_keyword_edit_idx" not in st.session_state:
    st.session_state["active_keyword_edit_idx"] = None
if "show_norm" not in st.session_state:
    st.session_state["show_norm"] = False
if "truthfinder_payload" not in st.session_state:
    st.session_state["truthfinder_payload"] = None

# zk 状态
if "zk_proof_generated" not in st.session_state:
    st.session_state["zk_proof_generated"] = False
if "zk_verified" not in st.session_state:
    st.session_state["zk_verified"] = False
if "zk_public_summary" not in st.session_state:
    st.session_state["zk_public_summary"] = None
if "zk_proof_message" not in st.session_state:
    st.session_state["zk_proof_message"] = ""
if "zk_verify_message" not in st.session_state:
    st.session_state["zk_verify_message"] = ""
if "zk_proof_path" not in st.session_state:
    st.session_state["zk_proof_path"] = ""
if "zk_public_path" not in st.session_state:
    st.session_state["zk_public_path"] = ""
if "zk_stage_status" not in st.session_state:
    st.session_state["zk_stage_status"] = "未生成"
if "zk_verify_status" not in st.session_state:
    st.session_state["zk_verify_status"] = "未验证"
if "zk_last_runtime_sec" not in st.session_state:
    st.session_state["zk_last_runtime_sec"] = 0.0
if "zk_last_logs" not in st.session_state:
    st.session_state["zk_last_logs"] = {}
if "zk_reference_output" not in st.session_state:
    st.session_state["zk_reference_output"] = None
if "zk_compare_rows" not in st.session_state:
    st.session_state["zk_compare_rows"] = []
if "zk_reference_path" not in st.session_state:
    st.session_state["zk_reference_path"] = ""
if "zk_error" not in st.session_state:
    st.session_state["zk_error"] = None
if "zk_status_cards" not in st.session_state:
    st.session_state["zk_status_cards"] = {
        "inputs": "未生成",
        "witness": "未生成",
        "proof": "未生成",
        "verify": "未验证",
    }
if st.session_state.get("zk_stage_status") not in {"未生成", "已生成", "失败"}:
    st.session_state["zk_stage_status"] = "未生成"
if st.session_state.get("zk_verify_status") not in {"未验证", "验证通过", "验证失败"}:
    st.session_state["zk_verify_status"] = "未验证"

def render_truthfinder_results(tf_payload: Dict[str, Any]) -> None:
    rank = tf_payload["rank"]
    truth_rows = tf_payload["truth_rows"]
    cand_map = tf_payload["cand_map"]
    sentence_id = tf_payload["sentence_id"]

    def _join_truth(truth_list: list) -> str:
        return " / ".join(truth_list) if truth_list else "(空)"

    st.markdown("#### 模型相对可信度")
    st.caption("模型相对可信度用于解释模型在关键词候选选择上的一致性表现，不直接代表整句翻译质量。")
    render_light_table(
        [{"模型": model_ui_name(model_name), "相对可信度": round(score, 4)} for model_name, score in rank],
        ["模型", "相对可信度"],
    )

    st.markdown("#### 关键词级可信候选")
    render_light_table(
        [
            {
                "序号": i + 1,
                "关键词": row["keyword"],
                "可信候选": _join_truth(row["truth"]),
                "候选事实置信度": " / ".join([f"{conf:.4f}" for conf in row["conf"]]),
            }
            for i, row in enumerate(truth_rows)
        ],
        ["序号", "关键词", "可信候选", "候选事实置信度"],
    )

    with st.expander("查看候选细节", expanded=False):
        render_light_table(
            [
                {
                    "关键词": row["keyword"],
                    "候选集合（调试）": " / ".join(cand_map.get((sentence_id, row["keyword"]), [])),
                }
                for row in truth_rows
            ],
            ["关键词", "候选集合（调试）"],
        )


def render_zk_status_card(label: str, status: str) -> None:
    tone = _status_tone(status)
    st.markdown(
        f"""
        <div class="zk-status-card tone-{tone}">
            <div class="zk-status-label">{html.escape(label)}</div>
            <div class="zk-status-value">{html.escape(status)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_zk_compare_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.info("当前还没有可对照的 ZK 输出结果。")
        return

    header = "".join(f"<th>{col}</th>" for col in ["序号", "关键词", "前端可信候选", "ZK 证明候选", "一致性"])
    body_rows = []
    for row in rows:
        badge_class = "ok" if row["一致性"] == "通过" else "warn"
        body_rows.append(
            """
            <tr>
                <td>{idx}</td>
                <td>{kw}</td>
                <td>{frontend}</td>
                <td>{zk}</td>
                <td><span class="inline-badge {badge}">{status}</span></td>
            </tr>
            """.format(
                idx=html.escape(str(row["序号"])),
                kw=html.escape(str(row["关键词"])),
                frontend=html.escape(str(row["前端可信候选"])),
                zk=html.escape(str(row["ZK 证明候选"])),
                badge=badge_class,
                status=html.escape(str(row["一致性"])),
            )
        )

    table_html = f"""
    <div class="light-table-wrap">
        <table class="light-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """
    st.markdown(
        table_html,
        unsafe_allow_html=True,
    )


def render_zk_file_viewer() -> None:
    files = [
        ("proof.json", "Groth16 证明文件", PROOF_PATH, "已生成"),
        ("public.json", "公开输出", PUBLIC_PATH, "已生成"),
        ("verification_key.json", "验证密钥", VERIFICATION_KEY_PATH, "已存在"),
    ]

    st.markdown("#### 查看证明文件")

    for filename, desc, path, present_label in files:
        exists = path.exists()
        status = present_label if exists else "缺失"

        col1, col2, col3, col4 = st.columns([1.4, 2.0, 0.8, 1.0])
        with col1:
            st.markdown(f"**{filename}**")
        with col2:
            st.write(desc)
        with col3:
            st.write(status)
        with col4:
            if exists:
                st.download_button(
                    label="下载",
                    data=path.read_bytes(),
                    file_name=filename,
                    mime="application/json",
                    key=f"download_zk_{filename.replace('.', '_')}",
                )

        if exists:
            with st.expander(f"预览 {filename}", expanded=False):
                try:
                    text = path.read_text(encoding="utf-8")
                    max_chars = 3000
                    preview = text[:max_chars]
                    if len(text) > max_chars:
                        preview += "\n\n... 文件较长，仅预览前 3000 字符。请下载查看完整文件。"
                    st.markdown(
                        f"""
                        <div class="zk-file-preview">
                            <pre>{html.escape(preview)}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as ex:
                    st.warning(f"文件预览失败：{type(ex).__name__}: {ex}")


def render_zk_panel() -> None:
    st.markdown("#### 零知识证明验证")
    st.markdown(
        """
        <div class="section-card">
            <div class="muted-text">
                零知识证明用于验证本次关键词级可信聚合计算是否按照预设 Groth16 电路执行。验证通过表示：系统展示的聚合结果由当前输入、模型选择、候选关系和固定 TruthFinder 计算流程共同得到，计算过程未被随意篡改。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("注意：该证明验证的是计算过程，不直接证明大语言模型原始回答一定正确。")
    st.caption(
        f"当前演示电路支持最多 {ZK_DEMO_K_MAX} 个关键词、每个关键词最多 {ZK_DEMO_N_MAX} 个候选 fact，并固定执行 {ZK_DEMO_ITER_N} 轮 Q16 TruthFinder 聚合。"
    )

    status_cards = st.session_state.get("zk_status_cards", {}) or {}
    status_cols = st.columns(4)
    with status_cols[0]:
        render_zk_status_card("输入绑定", status_cards.get("inputs", "未生成"))
    with status_cols[1]:
        render_zk_status_card("Witness", status_cards.get("witness", "未生成"))
    with status_cols[2]:
        render_zk_status_card("Groth16 证明", status_cards.get("proof", "未生成"))
    with status_cols[3]:
        render_zk_status_card("验证结果", status_cards.get("verify", "未验证"))

    should_rerun = False
    if st.button("生成并验证 ZK 证明", key="btn_zk_all_in_one", type="primary"):
        zk_keywords = st.session_state.get("fixed_keywords") or []
        if len(zk_keywords) > ZK_DEMO_K_MAX:
            st.warning("当前 ZK 电路最多支持 10 个关键词，请减少关键词数量后再生成证明。")
        elif not st.session_state.get("truthfinder_payload"):
            st.warning("请先完成 TruthFinder 可信聚合结果计算。")
        else:
            reset_zk_state(remove_artifacts=True)
            start_ts = time.time()
            current_step = "输入绑定"
            with st.status("正在执行 ZK 证明链路...", expanded=True) as zk_status:
                try:
                    st.write("正在生成输入...")
                    current_step = "输入绑定"
                    st.session_state["zk_status_cards"]["inputs"] = "运行中"
                    state = collect_current_zk_state()
                    paths = build_all_zk_inputs(state)
                    st.session_state["zk_status_cards"]["inputs"] = "已生成"
                    st.session_state["zk_last_logs"]["inputs"] = {k: str(v) for k, v in paths.items()}

                    st.write("正在生成 witness...")
                    current_step = "Witness"
                    st.session_state["zk_status_cards"]["witness"] = "运行中"
                    witness_ret = generate_witness_file(paths["witness_input_path"])
                    st.session_state["zk_last_logs"]["witness"] = witness_ret
                    if not witness_ret["success"]:
                        st.session_state["zk_status_cards"]["witness"] = "失败"
                        raise RuntimeError("witness 生成失败")
                    st.session_state["zk_status_cards"]["witness"] = "已生成"

                    st.write("正在生成 Groth16 proof...")
                    current_step = "Groth16 证明"
                    st.session_state["zk_status_cards"]["proof"] = "运行中"
                    prove_ret = generate_proof_from_witness(Path(witness_ret["witness_path"]))
                    st.session_state["zk_last_logs"]["prove"] = prove_ret
                    if not prove_ret["success"]:
                        st.session_state["zk_status_cards"]["proof"] = "失败"
                        raise RuntimeError("Groth16 proof 生成失败")
                    st.session_state["zk_status_cards"]["proof"] = "已生成"

                    st.write("正在验证 proof...")
                    current_step = "Groth16 验证"
                    st.session_state["zk_status_cards"]["verify"] = "运行中"
                    verify_ret = verify_generated_proof(Path(prove_ret["proof_path"]), Path(prove_ret["public_path"]))
                    st.session_state["zk_last_logs"]["verify"] = verify_ret
                    st.session_state["zk_verify_message"] = verify_ret["message"]
                    if not verify_ret["success"]:
                        st.session_state["zk_status_cards"]["verify"] = "验证失败"
                        raise RuntimeError("Groth16 proof 验证失败")
                    st.session_state["zk_status_cards"] = {
                        "inputs": "已生成",
                        "witness": "已生成",
                        "proof": "已生成",
                        "verify": "验证通过",
                    }
                    st.session_state["zk_proof_generated"] = True
                    st.session_state["zk_verified"] = True
                    st.session_state["zk_stage_status"] = "已生成"
                    st.session_state["zk_verify_status"] = "验证通过"
                    st.session_state["zk_error"] = None
                    st.session_state["zk_proof_message"] = "ZK 证明与验证已完成"
                    st.session_state["zk_proof_path"] = prove_ret["proof_path"]
                    st.session_state["zk_public_path"] = prove_ret["public_path"]
                    st.session_state["zk_public_summary"] = None
                    st.session_state["zk_compare_rows"] = []
                    st.session_state["zk_reference_output"] = None
                    st.session_state["zk_reference_path"] = ""
                    st.session_state["zk_last_runtime_sec"] = time.time() - start_ts
                    should_rerun = True
                    zk_status.update(label="验证完成", state="complete")
                except Exception as ex:
                    elapsed = time.time() - start_ts
                    if current_step == "输入绑定":
                        st.session_state["zk_status_cards"] = {
                            "inputs": "失败",
                            "witness": "未生成",
                            "proof": "未生成",
                            "verify": "未验证",
                        }
                    elif current_step == "Witness":
                        st.session_state["zk_status_cards"] = {
                            "inputs": "已生成",
                            "witness": "失败",
                            "proof": "未生成",
                            "verify": "未验证",
                        }
                    elif current_step == "Groth16 证明":
                        st.session_state["zk_status_cards"] = {
                            "inputs": "已生成",
                            "witness": "已生成",
                            "proof": "失败",
                            "verify": "未验证",
                        }
                    else:
                        st.session_state["zk_status_cards"] = {
                            "inputs": "已生成",
                            "witness": "已生成",
                            "proof": "已生成",
                            "verify": "验证失败",
                        }
                    st.session_state["zk_stage_status"] = "失败"
                    st.session_state["zk_verify_status"] = (
                        "验证失败"
                        if current_step == "Groth16 验证"
                        else st.session_state.get("zk_verify_status", "未验证")
                    )
                    st.session_state["zk_last_runtime_sec"] = elapsed
                    current_cards = st.session_state["zk_status_cards"]
                    st.session_state["zk_proof_generated"] = current_cards.get("proof") == "已生成"
                    st.session_state["zk_verified"] = current_cards.get("verify") == "验证通过"
                    st.session_state["zk_proof_message"] = f"{type(ex).__name__}: {ex}"

                    logs = st.session_state.get("zk_last_logs", {})
                    detail = None
                    for name in ["verify", "prove", "witness"]:
                        if isinstance(logs.get(name), dict):
                            detail = logs[name]
                            break
                    st.session_state["zk_error"] = _build_step_error(
                        step=current_step,
                        message=f"{type(ex).__name__}: {ex}",
                        details=detail,
                    )
                    should_rerun = True
                    zk_status.update(label="验证失败", state="error")
    if should_rerun:
        st.rerun()

    zk_error = st.session_state.get("zk_error")
    if zk_error:
        st.error("ZK 证明生成/验证失败，请检查输入文件、证明密钥或运行环境。")
        st.caption(f"失败步骤：{zk_error.get('step')}；错误摘要：{zk_error.get('message')}")
    elif st.session_state.get("zk_verified"):
        st.success("ZK 证明验证通过：本次可信聚合结果已通过 Groth16 证明验证，系统按照固定 Q16 TruthFinder 电路完成了 15 轮聚合计算。")
        st.caption("该验证保证计算流程一致性，不代表模型原始翻译一定绝对正确。")
        render_zk_file_viewer()


render_header()
render_step_header(1, "输入英文文本")

english = st.text_area("英文原文", value="", height=180, placeholder="请粘贴或输入待分析的英文文本…")
k = st.number_input("关键词数量", min_value=3, max_value=10, value=6, step=1)

current_english = english.strip()
current_k = int(k)

if current_english != st.session_state["last_english"] or st.session_state["last_k"] != current_k:
    st.session_state["last_english"] = current_english
    st.session_state["last_k"] = current_k
    st.session_state["keyword_editor_rows"] = []
    st.session_state["keywords_extracted"] = False
    st.session_state["selected_keywords"] = []
    st.session_state["keyword_candidate_pool"] = []
    st.session_state["active_keyword_edit_idx"] = None
    st.session_state["custom_keywords"] = ""
    st.session_state["results"] = None
    st.session_state["times"] = None
    st.session_state["fixed_keywords"] = None
    st.session_state["truthfinder_payload"] = None
    st.session_state["show_norm"] = False
    reset_zk_state(remove_artifacts=True)

render_step_badges()

if current_english:
    action_cols = st.columns([1, 4])
    with action_cols[0]:
        extract_btn = st.button("提取关键词", use_container_width=True, type="primary")

    if extract_btn:
        extracted_keywords = extract_keywords(current_english, k=current_k)
        st.session_state["keywords_extracted"] = True
        st.session_state["selected_keywords"] = list(extracted_keywords)
        st.session_state["keyword_candidate_pool"] = extract_keyword_candidates_for_ui(current_english)
        st.session_state["active_keyword_edit_idx"] = None
        st.session_state["custom_keywords"] = " ".join(extracted_keywords) if extracted_keywords else ""
        st.session_state["results"] = None
        st.session_state["times"] = None
        st.session_state["fixed_keywords"] = None
        st.session_state["truthfinder_payload"] = None
        st.session_state["show_norm"] = False
        reset_zk_state(remove_artifacts=True)
        if not extracted_keywords:
            st.warning("未提取到关键词，请调整文本内容后重试。")

    if st.session_state["results"] is None:
        render_step_header(2, "确认关键词")
        if st.session_state["keywords_extracted"] and st.session_state["selected_keywords"]:
            selected_keywords = st.session_state.get("selected_keywords", [])
            head_cols = st.columns([0.8, 4.6, 2.1])
            with head_cols[0]:
                st.markdown('<div class="keyword-table-head">序号</div>', unsafe_allow_html=True)
            with head_cols[1]:
                st.markdown('<div class="keyword-table-head">当前关键词</div>', unsafe_allow_html=True)
            with head_cols[2]:
                st.markdown('<div class="keyword-table-head">操作</div>', unsafe_allow_html=True)
            for idx, keyword in enumerate(selected_keywords):
                row_cols = st.columns([0.6, 4.9, 2.1])
                with row_cols[0]:
                    st.markdown(f'<div class="keyword-row"><strong>{idx + 1}</strong></div>', unsafe_allow_html=True)
                with row_cols[1]:
                    st.markdown(f'<div class="keyword-row">{html.escape(keyword)}</div>', unsafe_allow_html=True)
                with row_cols[2]:
                    with st.popover("更改", use_container_width=True, width="medium"):
                        render_candidate_replace_panel(idx)
        else:
            st.info("请先点击“提取关键词”，系统将默认使用自动抽取的前 k 个关键词。")

        call_btn = st.button("开始多模型分析", use_container_width=True, type="primary")
        if call_btn:
            if not current_english:
                st.warning("请先输入英文文本。")
            else:
                final_keywords = [str(word).strip() for word in st.session_state.get("selected_keywords", []) if str(word).strip()]
                if not final_keywords:
                    st.warning("请先提取关键词并确认至少一个关键词。")
                else:
                    st.info("正在调用模型并生成结果，请稍等…")
                    results = {}
                    times = {}
                    for model_name in MODELS:
                        start = time.time()
                        try:
                            kw_map = get_keyword_meanings(model_name, final_keywords)
                            missing_words = [w for w in final_keywords if not kw_map.get(w, "").strip()]
                            if missing_words:
                                fill_map = fill_missing_meanings(model_name, missing_words)
                                for w in missing_words:
                                    if not kw_map.get(w, "").strip():
                                        kw_map[w] = fill_map.get(w, "")

                            aligned = [{"keyword": w, "meaning_zh": kw_map.get(w, "")} for w in final_keywords]
                            translation = generate_translation_only(model_name, current_english)
                            has_empty = any(not row["meaning_zh"].strip() for row in aligned)
                            results[model_name] = {
                                "ok": not has_empty,
                                "keywords": aligned,
                                "translation_zh": translation,
                            }
                        except Exception as e:
                            aligned = [{"keyword": w, "meaning_zh": ""} for w in final_keywords]
                            results[model_name] = {
                                "ok": False,
                                "keywords": aligned,
                                "translation_zh": "",
                                "error": f"{type(e).__name__}: {e}",
                            }
                        times[model_name] = time.time() - start

                    st.session_state["results"] = results
                    st.session_state["times"] = times
                    st.session_state["fixed_keywords"] = final_keywords
                    st.session_state["custom_keywords"] = " ".join(final_keywords)
                    st.session_state["truthfinder_payload"] = None
                    reset_zk_state(remove_artifacts=True)

    if st.session_state["results"]:
        results = st.session_state["results"]
        times = st.session_state["times"]
        fixed_keywords = st.session_state["fixed_keywords"]

        render_step_header(3, "四模型原始输出")
        with st.expander("查看关键词列表", expanded=False):
            render_light_table([{"序号": i + 1, "关键词": w} for i, w in enumerate(fixed_keywords)], ["序号", "关键词"])

        st.markdown("#### 模型调用概览")
        render_model_overview(results, times, fixed_keywords)
        render_model_detail_tabs(results, times)
        render_translation_summary(results, times)

        render_step_header(4, "关键词级可信译法发现")
        if st.button("生成关键词候选译法", use_container_width=True, key="btn_norm", type="primary"):
            st.session_state["show_norm"] = True

        if st.session_state["show_norm"]:
            normalized_rows = []
            for i, w in enumerate(fixed_keywords):
                row = {"关键词": w}
                for model_name in MODELS:
                    meaning = results[model_name]["keywords"][i]["meaning_zh"]
                    cands = normalize_meaning_zh_soft(meaning, top_n=3)
                    row[model_ui_name(model_name)] = format_candidates(cands, sep=" / ")
                normalized_rows.append(row)

            with st.expander("查看关键词候选译法归一化结果", expanded=not bool(st.session_state.get("truthfinder_payload"))):
                render_light_table(normalized_rows, ["关键词"] + [model_ui_name(model_name) for model_name in MODELS])

            tf_btn = st.button("运行 TruthFinder 可信聚合", use_container_width=True, key="btn_tf", type="primary")

            if tf_btn:
                if not st.session_state.get("results") or not st.session_state.get("fixed_keywords"):
                    st.warning("请先点击“开始多模型分析”生成结果，再运行 TruthFinder。")
                else:
                    sentence_id = "s0"
                    cfg = TruthFinderConfig(
                        t0=0.75,
                        gamma=0.35,
                        beta=0.35,
                        alpha_imp=0.25,
                        alpha_conflict=0.15,
                        topn_candidates=3,
                        delta=1e-4,
                        max_iter=25,
                    )

                    t_score, s_score, cand_map = truthfinder_run(
                        models=MODELS,
                        sentence_id=sentence_id,
                        keywords=fixed_keywords,
                        results=results,
                        cfg=cfg,
                    )

                    rank = rank_models_by_trust(t_score)
                    truth_rows = pick_truth_per_keyword(
                        sentence_id=sentence_id,
                        keywords=fixed_keywords,
                        s_score=s_score,
                        top_k=2,
                        margin=0.03,
                    )

                    best_model = rank[0][0] if rank else MODELS[0]
                    st.session_state["truthfinder_payload"] = {
                        "sentence_id": sentence_id,
                        "rank": rank,
                        "truth_rows": truth_rows,
                        "cand_map": cand_map,
                        "best_model": best_model,
                        "cfg_dict": {
                            "t0": 0.75,
                            "gamma": 0.35,
                            "beta": 0.35,
                            "alpha_imp": 0.25,
                            "alpha_conflict": 0.15,
                            "topn_candidates": 3,
                            "delta": 1e-4,
                            "max_iter": 25,
                            "cand_decay": 0.30,
                            "min_tau_scale": 0.20,
                        },
                    }

            tf_payload = st.session_state.get("truthfinder_payload")
            if tf_payload:
                render_truthfinder_results(tf_payload)

                render_step_header(5, "整句译文参考")
                best_model = tf_payload["best_model"]
                st.markdown(f"**整句译文参考来源：** {model_ui_name(best_model)}")
                st.caption(f"模型标识：{best_model}")
                st.write(results.get(best_model, {}).get("translation_zh", "") or "(空)")

                render_step_header(6, "零知识证明验证")
                render_zk_panel()
else:
    st.info("请输入英文文本。")

