# -*- coding: utf-8 -*-
"""
BioRAG Streamlit 界面
=====================
运行方式：streamlit run app.py
"""
import streamlit as st
from openai import OpenAI
from app_core import build_index, ask

# ── 常量 ──────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
GEN_MODEL   = "Qwen/Qwen2.5-72B-Instruct"
BASE_URL    = "https://api.siliconflow.cn/v1"

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioRAG · 生物医学文献问答",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 BioRAG · 生物医学文献智能问答")
st.caption("上传文献PDF，基于真实语义检索，精准回答机制相关问题")

# ── 侧边栏：配置 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input("硅基流动 API Key", type="password",
                            help="在 siliconflow.cn 申请，免费模型可直接使用")
    st.divider()
    st.header("📚 知识库状态")
    if "chunks" in st.session_state and st.session_state.chunks:
        st.success(f"✅ 已加载 {len(st.session_state.chunks)} 个chunk")
        sources = list({c.source for c in st.session_state.chunks})
        st.caption(f"来源文件（{len(sources)}篇）：")
        for s in sources:
            st.caption(f"• {s}")
    else:
        st.info("尚未上传文献")
    st.divider()
    st.caption("本工具基于RAG（检索增强生成）架构，回答严格基于上传文献内容，不编造信息。")

# ── 主区域：分两列 ────────────────────────────────────────────────────────────
col_upload, col_qa = st.columns([1, 1.5], gap="large")

# ── 左列：上传文献 ────────────────────────────────────────────────────────────
with col_upload:
    st.subheader("📄 上传文献")
    uploaded_files = st.file_uploader(
        "支持多个PDF同时上传",
        type=["pdf"],
        accept_multiple_files=True,
        help="上传后点击「构建知识库」，系统会解析文本、生成语义向量"
    )

    if uploaded_files and api_key:
        if st.button("🚀 构建知识库", type="primary", use_container_width=True):
            client = OpenAI(api_key=api_key, base_url=BASE_URL)
            pdf_files = {f.name: f.read() for f in uploaded_files}

            progress_bar = st.progress(0, text="正在解析PDF并生成embedding...")
            status = st.empty()

            def on_progress(current, total, msg):
                progress_bar.progress(current / total, text=msg)
                status.caption(msg)

            with st.spinner("处理中，请稍候..."):
                chunks = build_index(pdf_files, client, EMBED_MODEL,
                                     progress_callback=on_progress)

            st.session_state.chunks = chunks
            st.session_state.client = client
            progress_bar.empty()
            status.empty()
            st.success(f"✅ 知识库构建完成！共 {len(chunks)} 个有效chunk")
            st.rerun()

    elif uploaded_files and not api_key:
        st.warning("请先在左侧输入 API Key")
    elif not uploaded_files:
        st.info("请选择PDF文件")

# ── 右列：问答 ────────────────────────────────────────────────────────────────
with col_qa:
    st.subheader("💬 文献问答")

    # 历史记录
    if "history" not in st.session_state:
        st.session_state.history = []

    # 显示历史对话
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            with st.expander("📎 引用来源", expanded=False):
                for r in item["retrieved"]:
                    color = "gray" if r["filtered"] else "blue"
                    label = "已过滤（相关度不足）" if r["filtered"] else f"相关度 {r['score']}"
                    st.markdown(f"**:{color}[{r['source']}]** — {label}")
                    if not r["filtered"]:
                        st.caption(r["text"][:200] + "...")

    # 输入框
    if "chunks" in st.session_state and st.session_state.chunks:
        query = st.chat_input("输入你的问题，例如：PTPN22通过什么机制影响JAK-STAT信号通路？")
        if query:
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("检索中..."):
                    result = ask(
                        query,
                        st.session_state.chunks,
                        st.session_state.client,
                        EMBED_MODEL, GEN_MODEL
                    )
                st.write(result["answer"])
                with st.expander("📎 引用来源", expanded=True):
                    for r in result["retrieved_chunks"]:
                        if r["filtered"]:
                            st.markdown(f"**:gray[{r['source']}]** — 相关度 {r['score']}（低于阈值，已过滤）")
                        else:
                            st.markdown(f"**:blue[{r['source']}]** — 相关度 {r['score']}")
                            st.caption(r["text"][:200] + "...")

            st.session_state.history.append({
                "question": query,
                "answer": result["answer"],
                "retrieved": result["retrieved_chunks"]
            })
    else:
        st.info("请先在左侧上传文献并构建知识库")

        # 示例问题展示
        st.divider()
        st.caption("可以问的问题示例：")
        examples = [
            "PTPN22如何通过与14-3-3τ的结合影响JAK-STAT信号通路？",
            "PTPN22 R620W变异是什么样的突变？",
            "14-3-3蛋白家族一共有几个亚型？",
            "检测T细胞增殖能力用的是什么实验方法？",
        ]
        for ex in examples:
            st.caption(f"• {ex}")
