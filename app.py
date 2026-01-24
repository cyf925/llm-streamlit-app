import streamlit as st
import requests
import time
import json
import re

# ===== 你的 4 个模型（必须和 ollama list 一致）=====
MODELS = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "phi3:mini",
    "gemma2:2b",
]

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(model: str, prompt: str, timeout: int = 300) -> str:
    """调用 Ollama /api/generate，返回模型回答文本"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 512,  # 防止输出太短/被截断
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


# ✅ 停用词表（用于自动抽取英文关键词时过滤掉无意义高频词）
STOPWORDS = set("""
a an the in on at for to of and or but so with without among between into from by as is are was were be been being
this that these those it its we you they i he she them our your their
""".split())


def extract_keywords(english_text: str, k: int = 6):
    """
    从英文原文里抽取 k 个关键词（纯英文），保证数量一致
    规则：只要字母单词，去停用词，按出现频率排序
    """
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", english_text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) >= 3]

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # 按 (频率降序, 字母升序) 排序
    sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    keywords = [w for w, _ in sorted_words[:k]]

    # 如果不足 k 个，补一些原文出现的单词
    if len(keywords) < k:
        seen = set(keywords)
        for w in words:
            if w not in seen:
                keywords.append(w)
                seen.add(w)
            if len(keywords) >= k:
                break

    return keywords[:k]


def build_prompt_fixed_keywords(english_text: str, keywords: list) -> str:
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
  ],
  "translation_zh": "整段中文翻译"
}}

英文文本如下：
\"\"\"{english_text}\"\"\"
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


# ✅✅✅ 补全机制 1：只补关键词中文释义（更容易成功）
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


# ✅✅✅ 补全机制 2：只做全文翻译（更稳）
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
st.title("📝 四模型英文翻译对比（关键词释义 + 全文翻译）")
st.caption("输入一段英文，4 个本地模型会分别给出关键词释义和整段翻译，并标注清楚来源模型。")

# ✅ 打开网页默认空白
english = st.text_area(
    "请输入英文文本：",
    value="",
    height=160,
    placeholder="请粘贴或输入英文文本…"
)

col1, col2 = st.columns([1, 1])
with col1:
    k = st.slider("关键词数量", min_value=3, max_value=12, value=6)
with col2:
    run_btn = st.button("🚀 生成释义与翻译", use_container_width=True)

if run_btn:
    if not english.strip():
        st.warning("请先输入英文文本。")
    else:
        st.info("正在调用 4 个模型生成结果，请稍等…")

        results = {}
        times = {}

        # ✅ 固定关键词：所有模型使用同一批关键词（数量一致 + 英文原词）
        fixed_keywords = extract_keywords(english.strip(), k=k)

        st.markdown("✅ 本次统一关键词：")
        st.table([{"序号": i + 1, "关键词": w} for i, w in enumerate(fixed_keywords)])

        main_prompt = build_prompt_fixed_keywords(english.strip(), fixed_keywords)

        for model_name in MODELS:
            start = time.time()
            try:
                out = call_ollama(model_name, main_prompt)
                parsed = try_parse_json(out)

                # 先准备对齐表格（固定 k 行）
                aligned = [{"单词": w, "中文释义": ""} for w in fixed_keywords]
                translation = ""

                if parsed:
                    # ✅ 尝试读 keywords
                    model_kw = parsed.get("keywords", [])
                    kw_map = {}
                    if isinstance(model_kw, list):
                        for item in model_kw:
                            if isinstance(item, dict):
                                w = str(item.get("word", "")).strip()
                                z = str(item.get("meaning_zh", "")).strip()
                                if w:
                                    kw_map[w] = z

                    for row in aligned:
                        row["中文释义"] = kw_map.get(row["单词"], "")

                    # ✅ 尝试读 translation
                    translation = str(parsed.get("translation_zh", "")).strip()

                # ✅✅✅ 关键：内容不完整时，自动补全
                missing_words = [row["单词"] for row in aligned if not row["中文释义"].strip()]
                if missing_words:
                    fill_map = fill_missing_meanings(model_name, missing_words)
                    for row in aligned:
                        if not row["中文释义"].strip():
                            row["中文释义"] = fill_map.get(row["单词"], "")

                if not translation.strip():
                    translation = generate_translation_only(model_name, english.strip())

                # 判定是否稳定（只要还有空释义，就提示不稳定）
                has_empty = any(not row["中文释义"].strip() for row in aligned)
                results[model_name] = {
                    "ok": not has_empty,
                    "keywords": aligned,
                    "translation_zh": translation,
                }

            except Exception as e:
                aligned = [{"单词": w, "中文释义": ""} for w in fixed_keywords]
                results[model_name] = {
                    "ok": False,
                    "keywords": aligned,
                    "translation_zh": "",
                    "error": f"{type(e).__name__}: {e}"
                }

            times[model_name] = time.time() - start

        st.success("完成 ✅")

        st.subheader("📌 四模型输出（关键词释义 + 整段翻译）")

        cols = st.columns(2)
        for idx, model_name in enumerate(MODELS):
            with cols[idx % 2]:
                st.markdown(f"## 🧠 {model_name}")
                st.caption(f"耗时：{times[model_name]:.2f} 秒")

                # 提示稳定性
                if not results[model_name].get("ok", True):
                    if "error" in results[model_name]:
                        st.warning(f"该模型调用异常：{results[model_name]['error']}")
                    else:
                        st.warning("该模型输出不稳定：部分释义仍为空（已尽力自动补全）")

                # ✅ 关键词表格（序号从 1 开始）
                st.markdown("### ✅ 关键词释义")
                table_data = [
                    {"序号": i + 1, "单词": row["单词"], "中文释义": row["中文释义"]}
                    for i, row in enumerate(results[model_name]["keywords"])
                ]
                st.table(table_data)

                # 翻译
                st.markdown("### ✅ 整段中文翻译")
                st.write(results[model_name]["translation_zh"] if results[model_name]["translation_zh"] else "(空)")

                st.divider()

        # 汇总表预览
        st.subheader("🧾 汇总表（预览）")
        PREVIEW_LEN = 120
        st.table([
            {
                "model": model_name,
                "time(s)": f"{times[model_name]:.2f}",
                "translation_preview": (results[model_name]["translation_zh"][:PREVIEW_LEN] + " …")
                if len(results[model_name]["translation_zh"]) > PREVIEW_LEN else results[model_name]["translation_zh"],
            }
            for model_name in MODELS
        ])
