import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

from normalize import format_candidates, normalize_meaning_zh_soft


# ===== 路径配置（ZK） =====
APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent
PROJECT_ROOT = APP_DIR.parent
ZK_DIR = PROJECT_ROOT / "zk"
ZK_BUILD_DIR = ZK_DIR / "build"
ZK_OUTPUT_DIR = ZK_DIR / "output"

WASM_PATH = ZK_BUILD_DIR / "truthfinder_js" / "truthfinder.wasm"
WITNESS_JS_PATH = ZK_BUILD_DIR / "truthfinder_js" / "generate_witness.js"
FINAL_ZKEY_PATH = ZK_BUILD_DIR / "truthfinder_final.zkey"
VERIFICATION_KEY_PATH = ZK_BUILD_DIR / "verification_key.json"

RUNTIME_INPUT_PATH = ZK_OUTPUT_DIR / "truthfinder_runtime_input.json"
DENSE_INPUT_PATH = ZK_OUTPUT_DIR / "truthfinder_dense_input.json"
CIRCOM_INPUT_PATH = ZK_OUTPUT_DIR / "truthfinder_circom_input.json"
WITNESS_INPUT_PATH = ZK_OUTPUT_DIR / "truthfinder_witness_input.json"
WITNESS_PATH = ZK_OUTPUT_DIR / "witness.wtns"
PROOF_PATH = ZK_OUTPUT_DIR / "proof.json"
PUBLIC_PATH = ZK_OUTPUT_DIR / "public.json"

SCHEMA_PATH = ZK_DIR / "truthfinder_runtime_input_schema.json"
SNARKJS_CLI_PATH = PROJECT_ROOT / "node_modules" / "snarkjs" / "build" / "cli.cjs"

ZK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
from runtime_input_builder import build_truthfinder_runtime_input_from_state  # noqa: E402


# ===== ollama部署的4个模型 =====
MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q5_0",
    "gemma2:9b-instruct-q4_K_M",
    "koesn/mistral-7b-instruct:Q4_0",
]

OLLAMA_URL = "http://localhost:11434/api/chat"


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
STOPWORDS = set(
    """
a an the in on at for to can both all of and or but so with without among
between into from by as is are was were be been being
this that these those it its we you they i he she them our your their my myself
your yourself his her myself oneself have has had having been has been only
how what when where why who which about above after against along
among around at before behind below beside between beyond do does did doing done not
""".split()
)


def extract_keywords(english_text: str, k: int = 6):
    """
    从英文原文里抽取 k 个关键词（保持英文原样显示）
    规则：只要字母单词，去停用词，按出现频率排序
    """
    original_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", english_text)
    lower_words = [w.lower() for w in original_words]
    filtered = [(orig, low) for orig, low in zip(original_words, lower_words) if low not in STOPWORDS and len(low) >= 3]
    if not filtered:
        return []

    freq = {}
    first_seen = {}
    for orig, low in filtered:
        freq[low] = freq.get(low, 0) + 1
        if low not in first_seen:
            first_seen[low] = orig

    sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    keywords = [first_seen[low] for low, _ in sorted_words[:k]]

    if len(keywords) < k:
        seen = {w.lower() for w in keywords}
        for orig, low in filtered:
            if low not in seen:
                keywords.append(orig)
                seen.add(low)
            if len(keywords) >= k:
                break
    return keywords[:k]


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
            cands = normalize_meaning_zh_soft(kw2meaning.get(kw, ""), top_n=3)
            normalized_by_model[model_name][kw] = [c for c in cands if c and c.strip()]

    payload_cfg = ((st.session_state.get("truthfinder_payload") or {}).get("cfg_dict") or {})
    cfg_dict = payload_cfg if payload_cfg else {
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
    }

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
        truthfinder_path=APP_DIR / "TruthFinder.py",
        normalize_path=APP_DIR / "normalize.py",
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
    if not isinstance(raw, list) or len(raw) < 17:
        raise ValueError(f"public.json 格式异常：期望长度>=17，实际={len(raw) if isinstance(raw, list) else type(raw)}")

    vals = [int(str(x)) for x in raw]
    return {
        "best_model_idx": vals[0],
        "best_model_score_q16": vals[1],
        "winning_fact_idx_by_object": vals[2:17],
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


# ===== Streamlit 页面 =====
st.set_page_config(page_title="LLM Translator Comparator", layout="wide")
st.title("LLM Translator Comparator")

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

english = st.text_area("请输入英文文本：", value="", height=160, placeholder="请粘贴或输入英文文本…")
k = st.slider("关键词数量", min_value=3, max_value=15, value=6)

if english.strip():
    if english.strip() != st.session_state["last_english"] or st.session_state["last_k"] != k:
        st.session_state["last_english"] = english.strip()
        st.session_state["last_k"] = k
        extracted_keywords = extract_keywords(english.strip(), k=k)
        st.session_state["custom_keywords"] = " ".join(extracted_keywords) if extracted_keywords else ""
        st.session_state["results"] = None
        st.session_state["times"] = None
        st.session_state["fixed_keywords"] = None
        st.session_state["truthfinder_payload"] = None
        st.session_state["show_norm"] = False
        reset_zk_state(remove_artifacts=True)

    if st.session_state["results"] is None:
        current_keywords_list = [w.strip() for w in re.split(r"[\s,，、]+", st.session_state["custom_keywords"]) if w.strip()]
        st.markdown("关键词如下：")
        st.dataframe([{"序号": i + 1, "关键词": w} for i, w in enumerate(current_keywords_list)], use_container_width=True, hide_index=True)

        st.text_input("自定义关键词（多个关键词用空格或逗号分隔）", key="custom_keywords")
        call_btn = st.button("调用模型", use_container_width=True)
        if call_btn:
            if not english.strip():
                st.warning("请先输入英文文本。")
            else:
                final_keywords = [w.strip() for w in re.split(r"[\s,，、]+", st.session_state["custom_keywords"]) if w.strip()]
                if not final_keywords:
                    st.warning("关键词列表不能为空，请输入至少一个关键词。")
                else:
                    st.info("正在调用模型以生成结果，请稍等…")
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
                            translation = generate_translation_only(model_name, english.strip())
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
                    st.session_state["truthfinder_payload"] = None
                    reset_zk_state(remove_artifacts=True)

    if st.session_state["results"]:
        results = st.session_state["results"]
        times = st.session_state["times"]
        fixed_keywords = st.session_state["fixed_keywords"]

        st.success("完成")
        st.markdown("关键词如下：")
        st.dataframe([{"序号": i + 1, "关键词": w} for i, w in enumerate(fixed_keywords)], use_container_width=True, hide_index=True)

        st.subheader("输出展示（关键词释义 + 整段翻译）")
        cols = st.columns(2)
        for idx, model_name in enumerate(MODELS):
            with cols[idx % 2]:
                st.markdown(f"##  {model_name}")
                st.caption(f"耗时：{times.get(model_name, 0.0):.2f} 秒")
                if not results[model_name].get("ok", True):
                    if "error" in results[model_name]:
                        st.warning(f"该模型调用异常：{results[model_name]['error']}")
                    else:
                        st.warning("该模型输出不稳定：部分释义仍为空（已尽力自动补全）")

                st.markdown("### 关键词释义")
                table_data = [
                    {"序号": i + 1, "关键词": row["keyword"], "中文释义": row["meaning_zh"]}
                    for i, row in enumerate(results[model_name]["keywords"])
                ]
                st.dataframe(table_data, use_container_width=True, hide_index=True)

                st.markdown("### 全文翻译")
                st.write(results[model_name]["translation_zh"] if results[model_name]["translation_zh"] else "(空)")
                st.divider()

        st.subheader("结果汇总")
        st.dataframe(
            [{"模型": model_name, "整句翻译": results[model_name]["translation_zh"] or "(空)"} for model_name in MODELS],
            use_container_width=True,
            hide_index=True,
        )

        if st.button("归一化关键词翻译", use_container_width=True, key="btn_norm"):
            st.session_state["show_norm"] = True

        if st.session_state["show_norm"]:
            normalized_rows = []
            for i, w in enumerate(fixed_keywords):
                row = {"关键词": w}
                for model_name in MODELS:
                    meaning = results[model_name]["keywords"][i]["meaning_zh"]
                    cands = normalize_meaning_zh_soft(meaning, top_n=3)
                    row[model_name] = format_candidates(cands, sep="｜")
                normalized_rows.append(row)

            with st.expander("归一化关键词释义对比表格", expanded=True):
                st.dataframe(normalized_rows, use_container_width=True, hide_index=True)

                tf_btn = st.button("运行真值发现算法（TruthFinder）", use_container_width=True, key="btn_tf")

                if tf_btn:
                    if not st.session_state.get("results") or not st.session_state.get("fixed_keywords"):
                        st.warning("请先点击“调用模型”生成结果，再运行真值发现。")
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
                    rank = tf_payload["rank"]
                    truth_rows = tf_payload["truth_rows"]
                    cand_map = tf_payload["cand_map"]
                    sentence_id = tf_payload["sentence_id"]

                    def _join_truth(truth_list):
                        return "｜".join(truth_list) if truth_list else "(空)"

                    st.subheader("TruthFinder：模型可信度（t）")
                    st.dataframe(
                        [{"模型": m, "可信度t": round(v, 4)} for m, v in rank],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("TruthFinder：关键词的真值候选（fact）")
                    st.dataframe(
                        [
                            {
                                "序号": i + 1,
                                "关键词": r["keyword"],
                                "真值候选": _join_truth(r["truth"]),
                                "置信度": "｜".join([f"{c:.4f}" for c in r["conf"]]),
                                "所有候选（调试）": "｜".join(cand_map.get((sentence_id, r["keyword"]), [])),
                            }
                            for i, r in enumerate(truth_rows)
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    best_model = tf_payload["best_model"]
                    st.subheader("TruthFinder：推荐翻译结果（按可信度最高）")
                    st.markdown(f"**MODEL：{best_model}**")
                    st.write(results.get(best_model, {}).get("translation_zh", "") or "(空)")

            tf_payload = st.session_state.get("truthfinder_payload")
            if tf_payload:
                st.subheader("零知识证明验证")
                st.caption("当前推荐结果可以进一步生成零知识证明，并验证该结果来自预定义计算过程。")

                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    st.markdown(f"**证明状态：** {st.session_state.get('zk_stage_status', '未生成')}")
                with status_col2:
                    st.markdown(f"**验证状态：** {st.session_state.get('zk_verify_status', '未验证')}")

                btn_col1, btn_col2 = st.columns(2)

                with btn_col1:
                    if st.button("生成证明", use_container_width=True, key="btn_zk_prove"):
                        if not st.session_state.get("truthfinder_payload"):
                            st.warning("请先完成翻译推荐计算。")
                        else:
                            reset_zk_state(remove_artifacts=False)
                            st.session_state["zk_stage_status"] = "生成中"

                            start_ts = time.time()
                            with st.status("正在生成零知识证明…", expanded=True) as zk_status:
                                try:
                                    st.write("1/4 正在收集当前 zk state…")
                                    state = collect_current_zk_state()

                                    st.write("2/4 正在生成 zk 输入链（runtime/dense/circom/witness_input）…")
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
                    if st.button("验证证明", use_container_width=True, key="btn_zk_verify"):
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
                    st.code(_tail_text(st.session_state["zk_verify_message"]))

                summary = st.session_state.get("zk_public_summary")
                if summary:
                    st.markdown("### public.json 摘要")
                    st.write({
                        "best_model_idx": summary.get("best_model_idx"),
                        "best_model_score_q16": summary.get("best_model_score_q16"),
                        "winning_fact_idx_by_object": summary.get("winning_fact_idx_by_object"),
                    })

                with st.expander("查看技术细节", expanded=False):
                    st.markdown(f"- proof 路径: `{st.session_state.get('zk_proof_path', '')}`")
                    st.markdown(f"- public 路径: `{st.session_state.get('zk_public_path', '')}`")
                    st.markdown(f"- verification key: `{VERIFICATION_KEY_PATH}`")
                    st.markdown(f"- snarkjs cli: `{SNARKJS_CLI_PATH}`")
                    st.markdown(f"- wasm: `{WASM_PATH}`")
                    st.markdown(f"- 运行耗时: `{st.session_state.get('zk_last_runtime_sec', 0.0):.3f} 秒`")

                    logs = st.session_state.get("zk_last_logs", {}) or {}
                    if logs:
                        st.markdown("**命令输出摘要：**")
                        for name in ["witness", "prove", "verify"]:
                            if name in logs:
                                st.markdown(f"- {name}")
                                st.code(_tail_text(str(logs[name]), max_chars=1500))

                    if summary and "public_raw" in summary:
                        st.markdown("**public.json 原文（用于审计）**")
                        st.code(json.dumps(summary["public_raw"], ensure_ascii=False, indent=2))

else:
    st.info("请先输入英文文本。")
