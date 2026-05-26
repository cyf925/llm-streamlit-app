from __future__ import annotations

import html
import json
import re
import time
from typing import Any

import requests
import streamlit as st

from medical_app.normalize_medical import (
    build_medical_fact_table,
    get_medical_objects,
    normalize_all_models_medical_outputs,
)

try:
    from medical_app.medical_truthfinder import (
        build_medical_zk_payload,
        explain_truth_per_medical_object,
        medical_truthfinder_run,
    )
    try:
        from medical_app.medical_truthfinder import rank_models_by_trust
    except ImportError:
        def rank_models_by_trust(t_score: dict[str, float]) -> list[tuple[str, float]]:
            return sorted(t_score.items(), key=lambda x: x[1], reverse=True)
    TRUTHFINDER_READY = True
    TRUTHFINDER_IMPORT_ERROR = ""
except Exception as ex:
    build_medical_zk_payload = None
    explain_truth_per_medical_object = None
    medical_truthfinder_run = None

    def rank_models_by_trust(t_score: dict[str, float]) -> list[tuple[str, float]]:
        return sorted(t_score.items(), key=lambda x: x[1], reverse=True)

    TRUTHFINDER_READY = False
    TRUTHFINDER_IMPORT_ERROR = f"{type(ex).__name__}: {ex}"


st.set_page_config(
    page_title="医疗 TruthFinder 聚合系统",
    page_icon="🩺",
    layout="wide",
)


MODELS = [
    "qwen2.5:7b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q5_0",
    "gemma2:9b-instruct-q4_K_M",
    "koesn/mistral-7b-instruct:Q4_0",
]

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_LABELS = {
    "qwen2.5:7b-instruct-q4_K_M": "Qwen2.5",
    "mistral:7b-instruct-v0.3-q5_0": "Mistral",
    "gemma2:9b-instruct-q4_K_M": "Gemma2",
    "koesn/mistral-7b-instruct:Q4_0": "Koesn/Mistral",
}

MEDICAL_OBJECTS = get_medical_objects()
OBJECT_LABELS = {item["object_id"]: item["label"] for item in MEDICAL_OBJECTS}
OBJECT_MODES = {item["object_id"]: item["mode"] for item in MEDICAL_OBJECTS}

SESSION_DEFAULTS = {
    "medical_user_text": "",
    "medical_results": None,
    "medical_times": None,
    "medical_normalized_all": None,
    "medical_truthfinder_payload": None,
    "medical_final_advice": "",
    "medical_show_structured": False,
    "medical_show_normalized": False,
    "medical_show_truthfinder_input": False,
    "medical_model_running": "",
    "medical_error": "",
    "medical_zk_preview": None,
    "medical_zk_error": "",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.08), transparent 26%),
                linear-gradient(180deg, #F8FCFF 0%, #FFFFFF 18%, #FFFFFF 100%);
            color: #0F172A;
            font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F4FBFF 100%);
            border: 1px solid #D9ECF7;
            border-radius: 24px;
            padding: 1.3rem 1.6rem;
            margin-top: 0.75rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
        }
        .hero-card h1 {
            margin: 0;
            color: #0F172A;
            font-size: 2.05rem;
            font-weight: 750;
        }
        .hero-card p {
            margin: 0.55rem 0 0 0;
            color: #334155;
            line-height: 1.5;
            font-size: 0.96rem;
        }
        .section-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 1rem 1.15rem;
            margin: 0.6rem 0 1rem 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.03);
        }
        .soft-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.03);
        }
        .disclaimer-card {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.12), rgba(255, 255, 255, 0.98));
            border: 1px solid rgba(245, 158, 11, 0.32);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
        }
        .disclaimer-card h4 {
            margin: 0 0 0.45rem 0;
            color: #92400E;
        }
        .disclaimer-card p {
            margin: 0;
            color: #7C2D12;
            line-height: 1.55;
        }
        .step-title {
            color: #0284C7;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .small-caption {
            color: #64748B;
            font-size: 0.82rem;
        }
        .muted-text {
            color: #475569;
            line-height: 1.55;
        }
        .status-badge {
            display: inline-block;
            margin: 0.2rem 0.45rem 0.2rem 0;
            padding: 0.34rem 0.75rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 650;
            border: 1px solid #E2E8F0;
        }
        .status-badge.is-info {
            background: rgba(14, 165, 233, 0.08);
            color: #0369A1;
            border-color: rgba(14, 165, 233, 0.18);
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
        .light-table-wrap {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.02);
        }
        .light-table {
            width: 100%;
            border-collapse: collapse;
            background: #FFFFFF;
            color: #0F172A;
        }
        .light-table thead tr {
            background: #EFF8FF;
        }
        .light-table th,
        .light-table td {
            padding: 0.78rem 0.9rem;
            border-bottom: 1px solid #E2E8F0;
            text-align: left;
            vertical-align: top;
            font-size: 0.94rem;
            color: #0F172A;
            word-break: break-word;
            white-space: pre-wrap;
        }
        .light-table tbody tr:hover {
            background: #F8FAFC;
        }
        .light-table tbody tr:last-child td {
            border-bottom: none;
        }
        .model-card-title {
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.25rem;
        }
        .code-preview {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.84rem;
            color: #0F172A;
            white-space: pre-wrap;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def model_ui_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>医疗场景多模型 TruthFinder 可信聚合系统</h1>
            <p>输入身体状况描述 → 四模型风险提示 → 结构化归一化 → TruthFinder 聚合 → 生成系统综合建议</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_header(step_no: int, title: str, caption: str = "") -> None:
    caption_html = f'<div class="small-caption" style="margin-top: 0.2rem;">{html.escape(caption)}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="section-card">
            <div class="step-title">Step {step_no}</div>
            <h3 style="margin: 0.25rem 0 0.1rem 0; color: #0F172A;">{html.escape(title)}</h3>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_light_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
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


def try_parse_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```json\s*(\{.*?})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            return json.loads(text[left : right + 1])
        except Exception:
            return None
    return None


def _badge_tone(status: str) -> str:
    if status in {"已完成", "已归一化", "已聚合", "已生成", "可用"}:
        return "is-success"
    if status in {"调用异常", "失败", "未就绪"}:
        return "is-error"
    if status in {"运行中", "生成中"}:
        return "is-warning"
    if status in {"部分失败", "解析失败"}:
        return "is-warning"
    if status in {"待输入", "待运行", "未生成"}:
        return "is-pending"
    return "is-info"


def _badge_html(label: str, status: str) -> str:
    return f'<span class="status-badge {_badge_tone(status)}">{html.escape(label)}：{html.escape(status)}</span>'


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_medical_state() -> None:
    for key in [
        "medical_results",
        "medical_times",
        "medical_normalized_all",
        "medical_truthfinder_payload",
        "medical_final_advice",
        "medical_show_structured",
        "medical_show_normalized",
        "medical_show_truthfinder_input",
        "medical_model_running",
        "medical_error",
        "medical_zk_preview",
        "medical_zk_error",
    ]:
        st.session_state[key] = SESSION_DEFAULTS[key]


def render_medical_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer-card">
            <h4>医疗免责声明</h4>
            <p>
                本系统仅用于就医前风险提示和多模型意见聚合，不提供最终诊断、处方、药物剂量或治疗方案，不能替代医生、急诊或专业医疗机构。<br/>
                如出现胸痛持续不缓解、明显呼吸困难、意识改变、严重出血等情况，请及时线下就医或急诊。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_status() -> None:
    text_status = "已完成" if (st.session_state.get("medical_user_text") or "").strip() else "待输入"
    results = st.session_state.get("medical_results")
    normalized_all = st.session_state.get("medical_normalized_all")
    tf_payload = st.session_state.get("medical_truthfinder_payload")
    zk_preview = st.session_state.get("medical_zk_preview")

    if results:
        has_error = any(not (payload.get("ok")) for payload in (results or {}).values())
        model_status = "部分失败" if has_error else "已完成"
    else:
        model_status = "待运行"

    norm_status = "已归一化" if normalized_all else "待运行"
    tf_status = "已聚合" if tf_payload else ("未就绪" if not TRUTHFINDER_READY else "待运行")
    zk_status = "已生成" if zk_preview else "未生成"

    st.markdown(
        f"""
        <div class="section-card">
            <div class="small-caption">流程状态</div>
            <div style="margin-top: 0.45rem;">
                {_badge_html("用户描述", text_status)}
                {_badge_html("四模型调用", model_status)}
                {_badge_html("归一化", norm_status)}
                {_badge_html("TruthFinder", tf_status)}
                {_badge_html("ZK 预览", zk_status)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _json_dumps_pretty(data: Any) -> str:
    return json.dumps(_json_safe(data), ensure_ascii=False, indent=2)


def _stringify_key(value: Any) -> str:
    if isinstance(value, tuple):
        return " :: ".join(_stringify_key(item) for item in value)
    if isinstance(value, list):
        return " / ".join(_stringify_key(item) for item in value)
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return { _stringify_key(key): _json_safe(val) for key, val in value.items() }
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def build_medical_prompt(user_text: str) -> str:
    object_blocks = []
    for item in MEDICAL_OBJECTS:
        options = "\n".join(f"- {option}" for option in item["options"])
        plural_hint = "可以多个" if item["mode"] == "multi" else "只能选一个"
        object_blocks.append(
            f"{item['object_id']}（{item['label']}，{plural_hint}）:\n{options}"
        )
    object_text = "\n\n".join(object_blocks)

    return f"""
你是一个谨慎、可靠的就医前风险提示助手。你不能做最终诊断，不能开处方，不能给药物和剂量。

请只输出合法 JSON，不要输出 Markdown，不要输出代码块，不要输出解释性前缀。

请根据用户当前描述，输出：
1. user_explanation：给普通用户看的自然语言风险提示；
2. structured_analysis：六个 object 下的结构化判断。

严格要求：
1. 不能做最终诊断。
2. 不能开处方。
3. 不能给药物名称、剂量或治疗方案。
4. 不能说“肯定没事”“一定是某病”。
5. 条件性提醒不能覆盖当前紧急程度。
6. 如果只是“如果加重应急诊”，structured_analysis.urgency_level 必须表示当前描述下的紧急程度，而不是条件成立后的情况。
7. structured_analysis 不要输出长句，尽量直接从候选集合中选。
8. user_explanation 可以自然语言表达，但必须谨慎、清楚、不过度恐吓。
9. single object 只能输出一个候选值。
10. multi object 只能输出候选集合中的若干值。

六个 object 与候选集合如下：

{object_text}

输出 JSON 格式必须严格如下：
{{
  "user_explanation": "给普通用户看的自然语言风险提示",
  "structured_analysis": {{
    "danger_signal": "只能从候选集合中选择一个",
    "urgency_level": "只能从候选集合中选择一个",
    "possible_cause": ["只能从候选集合中选择，可以多个"],
    "risk_signal": ["只能从候选集合中选择，可以多个"],
    "low_risk_factor": ["只能从候选集合中选择，可以多个"],
    "consult_department": ["只能从候选集合中选择，可以多个"]
  }}
}}

用户描述：
\"\"\"{user_text.strip()}\"\"\"
""".strip()


def call_ollama_medical(model: str, prompt: str, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个谨慎、可靠的就医前风险提示助手。你不能做最终诊断，不能开处方。",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 1536,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


def parse_medical_model_output(raw_output: str) -> dict[str, Any]:
    parsed = try_parse_json(raw_output)
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "raw_output": raw_output,
            "parse_error": "无法解析为合法 JSON",
            "user_explanation": "",
            "structured_analysis": {},
            "error": "模型输出不是合法 JSON",
        }

    user_explanation = str(parsed.get("user_explanation", "") or "").strip()
    structured = parsed.get("structured_analysis", {})
    if not isinstance(structured, dict):
        return {
            "ok": False,
            "raw_output": raw_output,
            "parse_error": "structured_analysis 不是对象",
            "user_explanation": user_explanation,
            "structured_analysis": {},
            "error": "structured_analysis 不是合法对象",
        }

    return {
        "ok": True,
        "raw_output": raw_output,
        "user_explanation": user_explanation,
        "structured_analysis": structured,
    }


def _render_json_block(data: Any) -> None:
    st.markdown(f'<div class="code-preview">{html.escape(_json_dumps_pretty(data))}</div>', unsafe_allow_html=True)


def render_model_user_explanations(results: dict[str, Any], times: dict[str, float]) -> None:
    rows = []
    for model_name in MODELS:
        payload = results.get(model_name, {}) or {}
        if payload.get("ok"):
            status = "已完成"
            explanation = (payload.get("user_explanation") or "").strip() or "模型未返回 user_explanation"
        elif payload.get("parse_error"):
            status = "解析失败"
            explanation = payload.get("error") or "JSON 解析失败"
        else:
            status = "调用异常"
            explanation = payload.get("error") or "模型调用失败"
        rows.append(
            {
                "模型": model_ui_name(model_name),
                "状态": status,
                "耗时（秒）": f"{times.get(model_name, 0.0):.2f}",
                "用户可读分析": explanation,
            }
        )
    render_light_table(rows, ["模型", "状态", "耗时（秒）", "用户可读分析"])

    with st.expander("查看单模型原始输出与错误详情", expanded=False):
        for model_name in MODELS:
            payload = results.get(model_name, {}) or {}
            status = "已完成" if payload.get("ok") else ("解析失败" if payload.get("parse_error") else "调用异常")
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="model-card-title">{html.escape(model_ui_name(model_name))}</div>
                    <div class="small-caption">{html.escape(model_name)}</div>
                    <div style="margin-top: 0.45rem;">{_badge_html("状态", status)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if payload.get("error"):
                st.warning(payload["error"])
            if payload.get("user_explanation"):
                st.write(payload["user_explanation"])
            if payload.get("structured_analysis"):
                _render_json_block(payload["structured_analysis"])
            if payload.get("raw_output"):
                with st.expander(f"查看 {model_ui_name(model_name)} raw_output", expanded=False):
                    st.code(payload["raw_output"], language="json")


def _format_structured_cell(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(str(item) for item in value) if value else "未返回"
    if value is None:
        return "未返回"
    text = str(value).strip()
    return text or "未返回"


def render_structured_analysis_table(results: dict[str, Any]) -> None:
    rows = []
    for item in MEDICAL_OBJECTS:
        row = {"object": item["label"]}
        for model_name in MODELS:
            structured = (results.get(model_name, {}) or {}).get("structured_analysis", {}) or {}
            row[model_ui_name(model_name)] = _format_structured_cell(structured.get(item["object_id"]))
        rows.append(row)
    render_light_table(rows, ["object"] + [model_ui_name(model_name) for model_name in MODELS])


def _fact_table_rows(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in table:
        rows.append(
            {
                "模型": model_ui_name(str(row.get("model", ""))),
                "object": OBJECT_LABELS.get(str(row.get("object_id", "")), str(row.get("object_id", ""))),
                "facts": " / ".join(str(item) for item in row.get("facts", []) or []) or "(空)",
            }
        )
    return rows


def render_normalized_outputs(normalized_all: dict[str, Any]) -> None:
    normalized_table = build_medical_fact_table(normalized_all, source="normalized")
    truthfinder_input_table = build_medical_fact_table(
        normalized_all,
        source="from_model_fields",
        exclude_fallbacks=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="muted-text">
                <strong>四层区分：</strong> user_explanation 是模型给用户看的自然语言提示；
                structured_analysis 是模型结构化原始回答；
                normalized 是前端展示用归一化结果；
                TruthFinder 默认输入来自 from_model_fields + exclude_fallbacks=True。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("查看前端归一化结果（source=normalized）", expanded=True):
        render_light_table(_fact_table_rows(normalized_table), ["模型", "object", "facts"])

    with st.expander("查看 TruthFinder 输入预览（source=from_model_fields, exclude_fallbacks=True）", expanded=False):
        render_light_table(_fact_table_rows(truthfinder_input_table), ["模型", "object", "facts"])

    with st.expander("查看归一化 warnings 与安全补丁", expanded=False):
        for model_name in MODELS:
            payload = normalized_all.get(model_name, {}) or {}
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="model-card-title">{html.escape(model_ui_name(model_name))}</div>
                    <div class="small-caption">{html.escape(model_name)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            warnings = payload.get("warnings", []) or []
            safety_overrides = ((payload.get("patches", {}) or {}).get("safety_overrides", [])) or []
            if warnings:
                st.write("warnings:")
                _render_json_block(warnings)
            else:
                st.caption("无 warnings")
            if safety_overrides:
                st.write("safety_overrides:")
                _render_json_block(safety_overrides)
            else:
                st.caption("无 safety_overrides")


def render_truthfinder_results(payload: dict[str, Any]) -> None:
    t_score = payload.get("t_score", {}) or {}
    truth_rows = payload.get("truth_rows", []) or []
    debug_info = payload.get("debug_info", {}) or {}

    rank_rows = [
        {
            "排名": idx + 1,
            "模型": model_ui_name(model_name),
            "可信度": f"{score:.4f}",
        }
        for idx, (model_name, score) in enumerate(rank_models_by_trust(t_score))
    ]
    render_light_table(rank_rows, ["排名", "模型", "可信度"])

    summary_rows = []
    for row in truth_rows:
        selected_facts = row.get("selected_facts", []) or []
        selected_conf = row.get("selected_conf", []) or []
        summary_rows.append(
            {
                "object": OBJECT_LABELS.get(row.get("object_id", ""), row.get("object_id", "")),
                "聚合可信结果": " / ".join(selected_facts) if selected_facts else "(空)",
                "置信度": " / ".join(f"{float(conf):.3f}" for conf in selected_conf) if selected_conf else "0.000",
                "候选数量": len(row.get("candidates", []) or []),
            }
        )
    st.markdown("#### 六个 object 的聚合可信结果")
    render_light_table(summary_rows, ["object", "聚合可信结果", "置信度", "候选数量"])

    with st.expander("查看候选 fact 置信度明细", expanded=False):
        for row in truth_rows:
            st.markdown(
                f"**{OBJECT_LABELS.get(row.get('object_id', ''), row.get('object_id', ''))}**"
            )
            cand_rows = []
            for cand in row.get("candidates", []) or []:
                cand_rows.append(
                    {
                        "rank": cand.get("rank"),
                        "fact": cand.get("fact"),
                        "confidence": f"{float(cand.get('confidence', 0.0)):.4f}",
                        "is_selected": "是" if cand.get("is_selected") else "否",
                        "support_weight": f"{float(cand.get('support_weight', 0.0)):.3f}",
                        "support_by_model": " / ".join(
                            f"{model_ui_name(model)}:{float(weight):.3f}"
                            for model, weight in (cand.get("support_by_model", {}) or {}).items()
                        ) or "(空)",
                    }
                )
            render_light_table(
                cand_rows,
                ["rank", "fact", "confidence", "is_selected", "support_weight", "support_by_model"],
            )

    with st.expander("查看 TruthFinder debug 信息", expanded=False):
        _render_json_block(debug_info)


def build_final_medical_advice(
    truth_rows: list[dict[str, Any]],
    normalized_all: dict[str, Any],
    user_text: str,
) -> str:
    row_map = {row.get("object_id"): row for row in truth_rows}
    danger = (row_map.get("danger_signal", {}) or {}).get("selected_facts", ["信息不足"])
    urgency = (row_map.get("urgency_level", {}) or {}).get("selected_facts", ["信息不足"])
    risk_signal = (row_map.get("risk_signal", {}) or {}).get("selected_facts", [])
    cause = (row_map.get("possible_cause", {}) or {}).get("selected_facts", [])
    department = (row_map.get("consult_department", {}) or {}).get("selected_facts", [])

    override_rules = []
    for model_name in MODELS:
        payload = normalized_all.get(model_name, {}) or {}
        for item in ((payload.get("patches", {}) or {}).get("safety_overrides", [])) or []:
            rule = str(item.get("rule", "")).strip()
            if rule and rule not in override_rules:
                override_rules.append(rule)

    rule_text_map = {
        "strong-chest-risk": "系统安全规则提示胸部高风险模式，需要重点警惕持续胸痛或伴随气短、出汗等情况。",
        "neuro-risk": "系统安全规则提示神经系统高风险模式，需要警惕突发无力、言语不清或意识变化。",
        "severe-bleeding": "系统安全规则提示严重出血风险，应尽快急诊处理。",
        "chest-risk": "系统安全规则提示存在胸部相关风险信号，建议尽快线下评估。",
        "respiratory-risk": "系统安全规则提示呼吸相关风险，需要重点关注气短或呼吸困难。",
        "infection-with-consciousness-risk": "系统安全规则提示感染伴意识改变风险，应高度重视。",
        "infection-risk": "系统安全规则提示感染相关风险，需要关注高热及全身状态变化。",
        "abdominal-risk": "系统安全规则提示腹痛相关风险，需要警惕持续或加重的腹部症状。",
    }

    lines = [
        "以下为基于四模型结构化判断和 TruthFinder 聚合得到的就医前风险提示。",
        f"- 危险信号：{danger[0] if danger else '信息不足'}",
        f"- 整体紧急程度：{urgency[0] if urgency else '信息不足'}",
        f"- 需要重点关注的风险信号：{'、'.join(risk_signal) if risk_signal else '暂无明确聚合风险信号'}",
        f"- 可能原因方向：{'、'.join(cause) if cause else '暂未形成稳定方向'}",
        f"- 建议咨询科室：{' / '.join(department) if department else '不确定'}",
        "",
    ]

    if urgency and urgency[0] == "立即急诊":
        lines.append("建议：当前聚合结果倾向于立即急诊，请不要继续仅在线观察，尽快前往急诊或线下紧急评估。")
    elif urgency and urgency[0] == "尽快线下就医":
        lines.append("建议：当前聚合结果倾向于尽快线下就医，建议尽快安排医院或门诊评估，不建议仅长期自行观察。")
    elif urgency and urgency[0] == "普通门诊":
        lines.append("建议：当前更接近普通门诊评估场景，但仍应结合症状变化及时调整就医安排。")
    else:
        lines.append("建议：当前更接近短期观察或信息不足场景，若症状持续、加重或出现新的危险信号，应及时线下就医。")

    if risk_signal:
        lines.append(f"重点关注：请继续留意 {'、'.join(risk_signal)} 等症状是否持续、加重或新出现。")
    if override_rules:
        rule_messages = [rule_text_map[rule] for rule in override_rules if rule in rule_text_map]
        if rule_messages:
            lines.append("系统安全规则提示：" + " ".join(rule_messages))

    lines.append("安全提醒：若出现胸痛持续不缓解、明显呼吸困难、意识改变、严重出血等情况，应及时急诊。")
    lines.append("免责声明：本结果仅用于就医前风险提示，不构成诊断、处方、药物剂量或治疗方案，不能替代医生面诊。")
    return "\n".join(lines)


def render_zk_placeholder() -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="muted-text">
                零知识证明模块将在后续接入。当前医疗 TruthFinder 已可输出 ZK-friendly payload，包括 top1_choice、support_mask、imp/conf 等结构。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def maybe_generate_zk_preview() -> None:
    if not TRUTHFINDER_READY or build_medical_zk_payload is None:
        st.info("当前 medical_truthfinder 模块未提供 build_medical_zk_payload，暂不生成 ZK 输入预览。")
        return

    normalized_all = st.session_state.get("medical_normalized_all")
    if not normalized_all:
        st.warning("请先完成归一化。")
        return

    if st.button("生成 ZK 输入预览", use_container_width=True, key="btn_medical_zk_preview"):
        try:
            _, _, cand_map, debug_info = medical_truthfinder_run(
                models=MODELS,
                case_id="case_0",
                normalized_all=normalized_all,
                source="from_model_fields",
                exclude_fallbacks=True,
                support_mode="zk_top1",
                return_debug=True,
            )
            zk_payload = build_medical_zk_payload(
                models=MODELS,
                case_id="case_0",
                cand_map=cand_map,
                top1_choice=debug_info.get("top1_choice", {}),
                support_mask=debug_info.get("support_mask", {}),
                relation_mats=debug_info.get("relation_mats", {}),
                dep_avg=debug_info.get("dep_avg", {}),
            )
            st.session_state["medical_zk_preview"] = zk_payload
            st.session_state["medical_zk_error"] = ""
        except Exception as ex:
            st.session_state["medical_zk_preview"] = None
            st.session_state["medical_zk_error"] = f"{type(ex).__name__}: {ex}"

    if st.session_state.get("medical_zk_error"):
        st.error(st.session_state["medical_zk_error"])
    if st.session_state.get("medical_zk_preview"):
        with st.expander("查看 ZK 输入预览", expanded=False):
            _render_json_block(st.session_state["medical_zk_preview"])


def run_model_pipeline(user_text: str) -> None:
    reset_medical_state()
    st.session_state["medical_user_text"] = user_text
    prompt = build_medical_prompt(user_text)

    progress = st.progress(0.0)
    status = st.empty()
    results: dict[str, Any] = {}
    times: dict[str, float] = {}
    error_count = 0

    for idx, model_name in enumerate(MODELS, start=1):
        st.session_state["medical_model_running"] = model_name
        status.info(f"正在调用 {model_ui_name(model_name)} ({idx}/{len(MODELS)})")
        start = time.time()
        try:
            raw_output = call_ollama_medical(model_name, prompt)
            parsed_payload = parse_medical_model_output(raw_output)
            if not parsed_payload.get("ok"):
                error_count += 1
            results[model_name] = parsed_payload
        except Exception as ex:
            error_count += 1
            results[model_name] = {
                "ok": False,
                "raw_output": "",
                "parse_error": "",
                "user_explanation": "",
                "structured_analysis": {},
                "error": f"{type(ex).__name__}: {ex}",
            }
        times[model_name] = time.time() - start
        progress.progress(idx / len(MODELS))

    st.session_state["medical_model_running"] = ""
    st.session_state["medical_results"] = results
    st.session_state["medical_times"] = times
    st.session_state["medical_error"] = (
        f"共有 {error_count} 个模型未成功返回可解析结果。"
        if error_count
        else ""
    )
    if error_count:
        status.warning(st.session_state["medical_error"])
    else:
        status.success("四个模型调用完成。")


def main() -> None:
    inject_css()
    init_session_state()

    render_header()
    render_medical_disclaimer()
    render_flow_status()

    render_step_header(1, "健康问题输入", "先输入当前身体状况描述。")
    user_text = st.text_area(
        "请描述你的身体状况",
        key="medical_user_text",
        height=180,
        placeholder="例如：最近两天胸闷，爬楼梯时明显，休息后缓解，偶尔心慌，没有发烧咳嗽……",
    )

    render_step_header(2, "调用四模型分析", "依次调用四个本地 Ollama 模型，获取 user_explanation 和 structured_analysis。")
    if st.button("开始四模型分析", use_container_width=True, type="primary"):
        if not user_text.strip():
            st.warning("请先输入身体状况描述。")
        else:
            run_model_pipeline(user_text.strip())

    results = st.session_state.get("medical_results")
    times = st.session_state.get("medical_times")

    if results and times:
        render_step_header(3, "四模型用户可读回答", "优先展示 user_explanation，不把结构化字段和归一化结果混在这里。")
        if st.session_state.get("medical_error"):
            st.warning(st.session_state["medical_error"])
        render_model_user_explanations(results, times)

        render_step_header(4, "结构化六字段对比", "这一层仅展示 structured_analysis 原始结构化回答。")
        with st.expander("查看六个 object 下的结构化判断", expanded=st.session_state.get("medical_show_structured", False)):
            render_structured_analysis_table(results)

        render_step_header(5, "归一化", "normalized 用于前端展示；TruthFinder 默认输入来自 from_model_fields + exclude_fallbacks=True。")
        if st.button("运行归一化", use_container_width=True, key="btn_run_medical_normalize", type="primary"):
            try:
                normalized_all = normalize_all_models_medical_outputs(
                    {
                        model: {
                            "user_explanation": (results.get(model, {}) or {}).get("user_explanation", ""),
                            "structured_analysis": (results.get(model, {}) or {}).get("structured_analysis", {}),
                        }
                        for model in MODELS
                    },
                    user_text=st.session_state.get("medical_user_text", ""),
                )
                st.session_state["medical_normalized_all"] = normalized_all
                st.session_state["medical_show_normalized"] = True
                st.session_state["medical_show_truthfinder_input"] = True
                st.session_state["medical_error"] = ""
            except Exception as ex:
                st.session_state["medical_error"] = f"归一化失败：{type(ex).__name__}: {ex}"

        if st.session_state.get("medical_error") and "归一化失败" in st.session_state["medical_error"]:
            st.error(st.session_state["medical_error"])

    normalized_all = st.session_state.get("medical_normalized_all")
    if normalized_all:
        render_normalized_outputs(normalized_all)

        render_step_header(6, "TruthFinder 可信聚合", "这一层只聚合模型结构化字段的归一化结果，不默认混入用户文本补丁和安全补丁。")
        if not TRUTHFINDER_READY:
            st.error("TruthFinder 模块尚未就绪。")
            if TRUTHFINDER_IMPORT_ERROR:
                st.caption(TRUTHFINDER_IMPORT_ERROR)
        else:
            if st.button("运行 TruthFinder 可信聚合", use_container_width=True, key="btn_run_medical_tf", type="primary"):
                try:
                    t_score, s_score, cand_map, debug_info = medical_truthfinder_run(
                        models=MODELS,
                        case_id="case_0",
                        normalized_all=normalized_all,
                        source="from_model_fields",
                        exclude_fallbacks=True,
                        support_mode="multi",
                        return_debug=True,
                    )
                    truth_rows = explain_truth_per_medical_object(
                        case_id="case_0",
                        s_score=s_score,
                        cand_map=cand_map,
                        support=debug_info.get("support"),
                    )
                    st.session_state["medical_truthfinder_payload"] = {
                        "t_score": t_score,
                        "s_score": s_score,
                        "cand_map": cand_map,
                        "debug_info": _json_safe(debug_info),
                        "truth_rows": truth_rows,
                    }
                    st.session_state["medical_final_advice"] = build_final_medical_advice(
                        truth_rows,
                        normalized_all,
                        st.session_state.get("medical_user_text", ""),
                    )
                    st.session_state["medical_error"] = ""
                except Exception as ex:
                    st.session_state["medical_error"] = f"TruthFinder 运行失败：{type(ex).__name__}: {ex}"

            if st.session_state.get("medical_error") and "TruthFinder 运行失败" in st.session_state["medical_error"]:
                st.error(st.session_state["medical_error"])

    tf_payload = st.session_state.get("medical_truthfinder_payload")
    if tf_payload:
        render_truthfinder_results(tf_payload)

        render_step_header(7, "系统综合建议", "这一层基于聚合结果模板生成，不再额外调用大模型。")
        if not st.session_state.get("medical_final_advice"):
            st.session_state["medical_final_advice"] = build_final_medical_advice(
                tf_payload.get("truth_rows", []),
                normalized_all or {},
                st.session_state.get("medical_user_text", ""),
            )
        st.text_area(
            "系统综合建议",
            value=st.session_state.get("medical_final_advice", ""),
            height=260,
        )

    render_step_header(8, "ZK 证明占位区域", "当前仅展示 ZK-friendly 输入预览，不调用 snarkjs，不生成 witness/proof。")
    render_zk_placeholder()
    maybe_generate_zk_preview()


if __name__ == "__main__":
    main()
