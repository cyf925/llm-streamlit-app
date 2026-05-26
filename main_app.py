from __future__ import annotations

from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
TRANSLATION_APP_MAIN = PROJECT_ROOT / "translation_app" / "app.py"


def main() -> None:
    st.set_page_config(page_title="多场景可信聚合系统", layout="wide")

    st.sidebar.title("场景选择")
    scene = st.sidebar.selectbox("请选择功能场景", ["翻译场景"])

    st.title("多场景可信聚合系统")

    if scene == "翻译场景":
        st.subheader("翻译场景")
        if TRANSLATION_APP_MAIN.exists():
            st.info(f"请直接运行 `streamlit run {TRANSLATION_APP_MAIN}` 进入翻译场景前端。")
            st.code(f"streamlit run {TRANSLATION_APP_MAIN}", language="bash")
        else:
            st.error(f"未找到翻译场景入口文件：{TRANSLATION_APP_MAIN}")


if __name__ == "__main__":
    main()
