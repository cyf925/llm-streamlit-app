import streamlit as st
import requests
import time
import json
import re

# ===== ollama部署的4个模型 =====
MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q5_0",
    "gemma2:9b-instruct-q4_K_M",
    "koesn/mistral-7b-instruct:Q4_0",
]

OLLAMA_URL = "http://localhost:11434/api/chat"

def call_ollama(model: str, prompt: str, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严谨、可靠的英语翻译与词汇释义助手。"},
            {"role": "user",  "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 512,
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message", {}) or {}).get("content", "").strip()

# 停用词表（用于自动抽取英文关键词时过滤掉无意义高频词）
STOPWORDS = set("""
a an the in on at for to can both all of and or but so with without among 
between into from by as is are was were be been being
this that these those it its we you they i he she them our your their my myself 
your yourself his her its myself oneself have has had having been has been only 
how what when where why who which about above after against along 
among around at before behind below beside between beyond
""".split())

def extract_keywords(english_text: str, k: int = 6):
    """
    从英文原文里抽取 k 个关键词（保持英文原样显示）
    规则：只要字母单词，去停用词，按出现频率排序
    """
    original_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", english_text)
    lower_words = [w.lower() for w in original_words]
    # 过滤停用词和长度小于3的词
    filtered = [(orig, low) for orig, low in zip(original_words, lower_words)
                if low not in STOPWORDS and len(low) >= 3]
    if not filtered:
        return []
    # 统计频率
    freq = {}
    first_seen = {}
    for orig, low in filtered:
        freq[low] = freq.get(low, 0) + 1
        if low not in first_seen:
            first_seen[low] = orig
    # 按频率降序、字母升序排序，选取前 k 个
    sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    keywords = [first_seen[low] for low, _ in sorted_words[:k]]
    # 若不足 k 个关键词则按出现顺序补齐
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
    """
    构建提示词：要求模型按给定关键词列表逐个输出中文释义（JSON格式）。
    """
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
    # 1) 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) 提取代码块
    match = re.search(r"```json\s*(\{.*?})\s*```", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # 3) 截取最外层的 JSON 内容
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(text[l:r+1])
        except Exception:
            return None
    return None

def get_keyword_meanings(model: str, keywords: list, timeout: int = 180) -> dict:
    """
    调用模型，获取关键词列表对应的中文释义，返回 {word: meaning_zh} 字典
    """
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
    """
    对于给定的未获取释义的英文单词列表，调用模型补全中文释义，返回 {word: meaning_zh}
    """
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
    """
    调用模型，仅生成整段英文文本的中文翻译结果
    """
    prompt = f"""
你是严谨的英译中助手。请只输出中文翻译结果，不要输出任何解释。

英文原文：
\"\"\"{english_text}\"\"\"
""".strip()
    out = call_ollama(model, prompt, timeout=timeout)
    return (out or "").strip()

# ===== 归一化关键词释义处理（soft：保留候选列表，1~3个） =====
from typing import List

ZH_ALIAS_SOFT = {
    "站点": "网站",
    "网页": "网站",
    "網站": "网站",
}
_SPLIT_RE = re.compile(r"[，,;；/｜|、\n]+|(?:\s+)|(?:或者)|(?:或是)|(?:\b或\b)")
_PARENS_RE = re.compile(r"[\(\（].*?[\)\）]")
_PREFIX_PATTERNS = [
    r"^指的是[:：]?", r"^指为[:：]?", r"^表示[:：]?", r"^意为[:：]?",
    r"^意思是[:：]?", r"^一种[:：]?", r"^用于[:：]?", r"^用来[:：]?",
    r"^用於[:：]?", r"^即[:：]?", r"^也称[:：]?", r"^又称[:：]?", r"^亦称[:：]?"
]
_PREFIX_RE = re.compile("|".join(_PREFIX_PATTERNS))
_KEEP_CORE_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")

def _clean_piece(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = _PARENS_RE.sub("", s).strip()
    s = _PREFIX_RE.sub("", s).strip()
    s = s.strip(" \"'“”‘’。.!！?？:：")
    if not s:
        return ""
    s = _KEEP_CORE_RE.sub("", s).strip()
    s = ZH_ALIAS_SOFT.get(s, s)
    return s

def normalize_meaning_zh_soft(raw: str, top_n: int = 3) -> List[str]:
    """
    软归一化：返回候选中文释义列表，保留差异，不强行合并多义。
    """
    if not raw or not str(raw).strip():
        return []
    raw2 = _PARENS_RE.sub("", str(raw).strip())
    parts = [p.strip() for p in _SPLIT_RE.split(raw2) if p and p.strip()]
    if not parts:
        parts = [raw2]
    cleaned = []
    seen = set()
    for p in parts:
        c = _clean_piece(p)
        if c and c not in seen:
            seen.add(c)
            cleaned.append(c)
    if not cleaned:
        return []
    # 排序：优先包含汉字的，次优先长度较短的（更像词条）
    cleaned.sort(key=lambda x: (1 if re.search(r"[\u4e00-\u9fff]", x) else 0, -len(x)), reverse=True)
    return cleaned[:top_n]

def display_norm_candidates(cands: List[str]) -> str:
    """前端展示：将候选列表拼接成一个字符串"""
    if not cands:
        return "(空)"
    return "｜".join(cands)

# ===== Streamlit 页面 =====
st.set_page_config(page_title="LLM Translator Comparator", layout="wide")
st.title("LLM Translator Comparator")
st.caption("输入英文文本，系统调用 4 个本地模型分别生成该文本的中文翻译")

# 初始化会话状态用于存储结果
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'times' not in st.session_state:
    st.session_state['times'] = None
if 'fixed_keywords' not in st.session_state:
    st.session_state['fixed_keywords'] = None

# 其他会话状态：用于跟踪输入变化和自定义关键词
if 'last_english' not in st.session_state:
    st.session_state['last_english'] = ""
if 'last_k' not in st.session_state:
    st.session_state['last_k'] = 6
if 'custom_keywords' not in st.session_state:
    st.session_state['custom_keywords'] = ""

english = st.text_area("请输入英文文本：", value="", height=160, placeholder="请粘贴或输入英文文本…")
k = st.slider("关键词数量", min_value=3, max_value=15, value=6)

# 当用户输入英文文本后，立即提取关键词并显示
if english.strip():
    if st.session_state['results'] is None:
        # 提取关键词（当英文文本或关键词数量改变时）
        if english.strip() != st.session_state['last_english'] or st.session_state['last_k'] != k:
            st.session_state['last_english'] = english.strip()
            st.session_state['last_k'] = k
            extracted_keywords = extract_keywords(english.strip(), k=k)
            st.session_state['custom_keywords'] = " ".join(extracted_keywords) if extracted_keywords else ""
            # 清除之前的结果，避免每次交互重复运行模型
            st.session_state['results'] = None
            st.session_state['times'] = None
            st.session_state['fixed_keywords'] = None
        # 显示当前关键词列表（表格）
        current_keywords_list = [w.strip() for w in re.split(r'[\s,，、]+', st.session_state['custom_keywords']) if w.strip()]
        st.markdown("关键词如下：")
        st.dataframe([{"序号": i+1, "关键词": w} for i, w in enumerate(current_keywords_list)], use_container_width=True, hide_index=True)
        # 允许用户手动编辑关键词列表
        st.text_input("自定义关键词（多个关键词用空格或逗号分隔）", key="custom_keywords")
        # 按下“调用模型”按钮后再触发模型调用
        call_btn = st.button("调用模型", use_container_width=True)
        if call_btn:
            if not english.strip():
                st.warning("请先输入英文文本。")
            else:
                # 获取最终关键词列表（用户可能进行了编辑）
                final_keywords = [w.strip() for w in re.split(r'[\s,，、]+', st.session_state['custom_keywords']) if w.strip()]
                if not final_keywords:
                    st.warning("关键词列表不能为空，请输入至少一个关键词。")
                else:
                    st.info("正在调用模型以生成结果，请稍等…")
                    results = {}
                    times = {}
                    # 调用每个模型获取关键词释义和整段翻译
                    for model_name in MODELS:
                        start = time.time()
                        try:
                            # 1) 关键词释义（JSON格式）
                            kw_map = get_keyword_meanings(model_name, final_keywords)
                            # 2) 补全缺失的释义
                            missing_words = [w for w in final_keywords if not kw_map.get(w, "").strip()]
                            if missing_words:
                                fill_map = fill_missing_meanings(model_name, missing_words)
                                for w in missing_words:
                                    if not kw_map.get(w, "").strip():
                                        kw_map[w] = fill_map.get(w, "")
                            # 3) 对齐关键词和释义结果顺序
                            aligned = [{"keyword": w, "meaning_zh": kw_map.get(w, "")} for w in final_keywords]
                            # 4) 获取全文中文翻译
                            translation = generate_translation_only(model_name, english.strip())
                            has_empty = any(not row["meaning_zh"].strip() for row in aligned)
                            results[model_name] = {
                                "ok": not has_empty,
                                "keywords": aligned,
                                "translation_zh": translation
                            }
                        except Exception as e:
                            aligned = [{"keyword": w, "meaning_zh": ""} for w in final_keywords]
                            results[model_name] = {
                                "ok": False,
                                "keywords": aligned,
                                "translation_zh": "",
                                "error": f"{type(e).__name__}: {e}"
                            }
                        times[model_name] = time.time() - start
                    # 保存结果和耗时到会话状态
                    st.session_state['results'] = results
                    st.session_state['times'] = times
                    st.session_state['fixed_keywords'] = final_keywords

    # 如已获得模型调用结果，显示关键词列表和各模型输出
    if st.session_state['results']:
        results = st.session_state['results']
        times = st.session_state['times']
        fixed_keywords = st.session_state['fixed_keywords']
        st.success("完成")
        st.markdown("关键词如下：")
        st.dataframe([{"序号": i + 1, "关键词": w} for i, w in enumerate(fixed_keywords)], use_container_width=True, hide_index=True)
        st.subheader("输出展示（关键词释义 + 整段翻译）")
        cols = st.columns(2)
        for idx, model_name in enumerate(MODELS):
            with cols[idx % 2]:
                st.markdown(f"##  {model_name}")
                st.caption(f"耗时：{times.get(model_name, 0.0):.2f} 秒")
                # 稳定性提示
                if not results[model_name].get("ok", True):
                    if "error" in results[model_name]:
                        st.warning(f"该模型调用异常：{results[model_name]['error']}")
                    else:
                        st.warning("该模型输出不稳定：部分释义仍为空（已尽力自动补全）")
                # 关键词释义表格
                st.markdown("### 关键词释义")
                table_data = [
                    {"序号": i + 1, "关键词": row["keyword"], "中文释义": row["meaning_zh"]}
                    for i, row in enumerate(results[model_name]["keywords"])
                ]
                st.dataframe(table_data, use_container_width=True, hide_index=True)
                # 全文翻译文本
                st.markdown("### 全文翻译")
                st.write(results[model_name]["translation_zh"] if results[model_name]["translation_zh"] else "(空)")
                st.divider()
        # 汇总各模型的整句翻译结果
        st.subheader("结果汇总")
        st.dataframe(
            [{"模型": model_name, "整句翻译": results[model_name]["translation_zh"] or "(空)"} for model_name in MODELS],
            use_container_width=True, hide_index=True
        )
        # --- 1) 先初始化 show_norm ---
        if "show_norm" not in st.session_state:
            st.session_state["show_norm"] = False

        # --- 2) 点击一次归一化按钮，就把 show_norm 打开 ---
        if st.button("归一化关键词翻译", use_container_width=True, key="btn_norm"):
            st.session_state["show_norm"] = True

        # --- 3) 只要 show_norm=True，就持续显示归一化表格 + TruthFinder按钮 ---
        if st.session_state["show_norm"]:
            normalized_rows = []
            for i, w in enumerate(fixed_keywords):
                row = {"关键词": w}
                for model_name in MODELS:
                    meaning = results[model_name]["keywords"][i]["meaning_zh"]
                    cands = normalize_meaning_zh_soft(meaning, top_n=3)
                    row[model_name] = display_norm_candidates(cands)
                normalized_rows.append(row)

            with st.expander("归一化关键词释义对比表格", expanded=True):
                st.dataframe(normalized_rows, use_container_width=True, hide_index=True)

                # ===== 真值发现（TruthFinder）按钮：不重复调用模型，只基于已生成的 results =====
                from TruthFinder import TruthFinderConfig, truthfinder_run, pick_truth_per_keyword, rank_models_by_trust

                tf_btn = st.button("运行真值发现算法（TruthFinder）", use_container_width=True, key="btn_tf")

                if tf_btn:
                    if not st.session_state.get("results") or not st.session_state.get("fixed_keywords"):
                        st.warning("请先点击“调用模型”生成结果，再运行真值发现。")
                    else:
                        results = st.session_state["results"]
                        fixed_keywords = st.session_state["fixed_keywords"]

                        sentence_id = "s0"
                        cfg = TruthFinderConfig(
                            t0=0.75,
                            gamma=0.35,
                            beta=0.35,
                            alpha_imp=0.25,
                            alpha_conflict=0.15,
                            topn_candidates=3,
                            delta=1e-4,
                            max_iter=25
                        )

                        t_score, s_score, cand_map = truthfinder_run(
                            models=MODELS,
                            sentence_id=sentence_id,
                            keywords=fixed_keywords,
                            results=results,
                            cfg=cfg
                        )

                        st.subheader("TruthFinder：模型可信度（t）")
                        rank = rank_models_by_trust(t_score)
                        st.dataframe(
                            [{"模型": m, "可信度t": round(v, 4)} for m, v in rank],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.subheader("TruthFinder：关键词的真值候选（fact）")
                        truth_rows = pick_truth_per_keyword(
                            sentence_id=sentence_id,
                            keywords=fixed_keywords,
                            s_score=s_score,
                            top_k=2,
                            margin=0.03
                        )


                        def _join_truth(truth_list):
                            return "｜".join(truth_list) if truth_list else "(空)"


                        st.dataframe(
                            [
                                {
                                    "序号": i + 1,
                                    "关键词": r["keyword"],
                                    "真值候选": _join_truth(r["truth"]),
                                    "置信度": "｜".join([f"{c:.4f}" for c in r["conf"]]),
                                    "所有候选（调试）": "｜".join(cand_map.get((sentence_id, r["keyword"]), []))
                                }
                                for i, r in enumerate(truth_rows)
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                        best_model = rank[0][0] if rank else MODELS[0]
                        st.subheader("TruthFinder：推荐翻译结果（按模型可信度最高）")
                        st.markdown(f"**推荐模型：{best_model}**")
                        st.write(results.get(best_model, {}).get("translation_zh", "") or "(空)")
