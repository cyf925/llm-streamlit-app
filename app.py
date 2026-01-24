import streamlit as st
import requests
import time
import json
import re

# ===== ollama部署的4个模型 =====
MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q4_K_M",
    "gemma2:9b-instruct-q4_K_M",
    "koesn/mistral-7b-instruct:Q4_0",
]

OLLAMA_URL = "http://localhost:11434/api/chat"

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
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # /api/chat 返回字段是 message.content
    return (data.get("message", {}) or {}).get("content", "").strip()



# ✅ 停用词表（用于自动抽取英文关键词时过滤掉无意义高频词）
STOPWORDS = set("""
a an the in on at for to can both all of and or but so with without among between into from by as is are was were be been being
this that these those it its we you they i he she them our your their
""".split())


def extract_keywords(english_text: str, k: int = 6):
    """
    ✅ 从英文原文里抽取 k 个关键词（保持英文原样显示）
    规则：只要字母单词，去停用词，按出现频率排序
    """
    # 保留原文形式（original_words），统计用小写（lower_words）
    original_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", english_text)
    lower_words = [w.lower() for w in original_words]

    # 过滤
    filtered = [(orig, low) for orig, low in zip(original_words, lower_words)
                if low not in STOPWORDS and len(low) >= 3]

    if not filtered:
        return []

    # 统计频率
    freq = {}
    first_seen = {}  # 记录每个词第一次出现时的原始大小写
    for orig, low in filtered:
        freq[low] = freq.get(low, 0) + 1
        if low not in first_seen:
            first_seen[low] = orig

    # 按 (频率降序, 字母升序) 排序
    sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    # 输出用原样（first_seen）
    keywords = [first_seen[low] for low, _ in sorted_words[:k]]

    # 不足 k 个时补齐（按出现顺序）
    if len(keywords) < k:
        seen = set([w.lower() for w in keywords])
        for orig, low in filtered:
            if low not in seen:
                keywords.append(orig)
                seen.add(low)
            if len(keywords) >= k:
                break

    return keywords[:k]


def build_prompt_keyword_meanings(keywords: list) -> str:
    """
    ✅ 强约束：模型必须按给定 keywords 列表逐个输出释义（顺序一致、数量一致、原词不变）
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
    2) 识别 ```json ... ``` 代码块
    3) 截取第一个 { 到最后一个 } 之间
    """
    text = (text or "").strip()
    if not text:
        return None

    # 1) 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 解析 ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3) 截取 { ... }
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(text[l:r + 1])
        except Exception:
            return None

    return None


def get_keyword_meanings(model: str, keywords: list, timeout: int = 180) -> dict:
    """
    让某个模型对给定关键词列表逐个给中文释义，返回 dict: {word: meaning_zh}
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


# 补全机制：补全关键词中文释义
def fill_missing_meanings(model: str, words: list, timeout: int = 180) -> dict:
    """
    给定缺失释义的英文词表，返回 {word: meaning_zh}
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


# 全文翻译，模型单独翻译
def generate_translation_only(model: str, english_text: str, timeout: int = 180) -> str:
    prompt = f"""
你是严谨的英译中助手。请只输出中文翻译结果，不要输出任何解释。

英文原文：
\"\"\"{english_text}\"\"\"
""".strip()
    out = call_ollama(model, prompt, timeout=timeout)
    return (out or "").strip()


# ===== Streamlit 页面 =====
st.set_page_config(page_title="LLM Translator Comparator", layout="wide")
st.title("LLM Translator Comparator")
st.caption("输入英文文本，系统调用 4 个本地模型分别生成该文本的中文翻译")

english = st.text_area(
    "请输入英文文本：",
    value="",
    height=160,
    placeholder="请粘贴或输入英文文本…"
)

col1, col2 = st.columns([1, 1])
with col1:
    k = st.slider("关键词数量", min_value=3, max_value=15, value=6)
with col2:
    run_btn = st.button("🚀 Click to translate", use_container_width=True)

if run_btn:
    if not english.strip():
        st.warning("请先输入英文文本。")
    else:
        st.info("正在调用模型以生成结果，请稍等…")

        results = {}
        times = {}

        # 统一关键词：所有模型使用同一批关键词（数量一致 + 英文原词）
        fixed_keywords = extract_keywords(english.strip(), k=k)

        st.markdown("关键词如下：")
        st.dataframe(
            [{"序号": i + 1, "关键词": w} for i, w in enumerate(fixed_keywords)],
            use_container_width=True,
            hide_index=True
        )

        # 核心变化：每个模型独立生成自己的关键词释义 + 自己补全缺失
        for model_name in MODELS:
            start = time.time()
            try:
                # 1) 这个模型自己生成关键词释义
                kw_map = get_keyword_meanings(model_name, fixed_keywords)

                # 2) 如果缺失，就让它自己补全缺失的词
                missing_words = [w for w in fixed_keywords if not kw_map.get(w, "").strip()]
                if missing_words:
                    fill_map = fill_missing_meanings(model_name, missing_words)
                    for w in missing_words:
                        if not kw_map.get(w, "").strip():
                            kw_map[w] = fill_map.get(w, "")

                # 3) 对齐显示（确保行数一定是 k 行）
                aligned = [{"keyword": w, "meaning_zh": kw_map.get(w, "")} for w in fixed_keywords]

                # 4) 全文翻译也由该模型自己生成（保持差异）
                translation = generate_translation_only(model_name, english.strip())

                # 判定是否稳定（只要还有空释义，就提示不稳定）
                has_empty = any(not row["meaning_zh"].strip() for row in aligned)
                results[model_name] = {
                    "ok": not has_empty,
                    "keywords": aligned,
                    "translation_zh": translation,
                }

            except Exception as e:
                aligned = [{"keyword": w, "meaning_zh": ""} for w in fixed_keywords]
                results[model_name] = {
                    "ok": False,
                    "keywords": aligned,
                    "translation_zh": "",
                    "error": f"{type(e).__name__}: {e}"
                }

            times[model_name] = time.time() - start

        st.success("完成")

        st.subheader("输出展示（关键词释义 + 整段翻译）")

        cols = st.columns(2)
        for idx, model_name in enumerate(MODELS):
            with cols[idx % 2]:
                st.markdown(f"##  {model_name}")
                st.caption(f"耗时：{times[model_name]:.2f} 秒")

                # 提示稳定性
                if not results[model_name].get("ok", True):
                    if "error" in results[model_name]:
                        st.warning(f"该模型调用异常：{results[model_name]['error']}")
                    else:
                        st.warning("该模型输出不稳定：部分释义仍为空（已尽力自动补全）")

                # 关键词表格：列名是「关键词」，显示英文
                st.markdown("### 关键词释义")
                table_data = [
                    {"序号": i + 1, "关键词": row["keyword"], "中文释义": row["meaning_zh"]}
                    for i, row in enumerate(results[model_name]["keywords"])
                ]
                st.dataframe(table_data, use_container_width=True, hide_index=True)

                # 翻译
                st.markdown("### 全文翻译")
                st.write(results[model_name]["translation_zh"] if results[model_name]["translation_zh"] else "(空)")

                st.divider()

        # 汇总表：四模型整句翻译
        st.subheader("结果汇总")
        st.dataframe(
            [
                {"模型": model_name, "整句翻译": results[model_name]["translation_zh"] or "(空)"}
                for model_name in MODELS
            ],
            use_container_width=True,
            hide_index=True
        )
