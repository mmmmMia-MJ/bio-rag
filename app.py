# -*- coding: utf-8 -*-
"""
BioRAG Streamlit 界面 v2
========================
主要更新：
- 导入优化后的 app_core（batch embedding、chunk_size=500）
- 二次上传PDF时追加合并到现有知识库，不覆盖
- 侧边栏增加"清空知识库"按钮
- 保存/加载知识库功能保留
"""
import streamlit as st
from openai import OpenAI
from app_core import build_index, ask, save_index, load_index

EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
GEN_MODEL   = "Qwen/Qwen2.5-72B-Instruct"
BASE_URL    = "https://api.siliconflow.cn/v1"

st.set_page_config(
    page_title="BioRAG · 生物医学文献问答",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 BioRAG · 生物医学文献智能问答")
st.caption("上传文献PDF，基于真实语义检索，精准回答机制相关问题")

# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input("硅基流动 API Key", type="password",
                            help="在 siliconflow.cn 申请")
    st.divider()

    # 知识库状态
    st.header("📚 知识库状态")
    if "chunks" in st.session_state and st.session_state.chunks:
        sources = list({c.source for c in st.session_state.chunks})
        st.success(f"✅ {len(st.session_state.chunks)} 个chunk · {len(sources)} 篇文献")
        for s in sources:
            st.caption(f"• {s}")

        col1, col2 = st.columns(2)
        with col1:
            # 清空知识库
            if st.button("🗑️ 清空", use_container_width=True,
                         help="清空当前知识库，重新上传"):
                del st.session_state.chunks
                st.session_state.history = []
                st.rerun()
        with col2:
            # 下载知识库
            st.download_button(
                label="⬇️ 保存",
                data=save_index(st.session_state.chunks),
                file_name="biorag_index.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                help="下载知识库文件，下次直接加载，无需重新构建"
            )
    else:
        st.info("尚未加载知识库")

    # 加载已有知识库
    st.divider()
    st.caption("📂 加载已有知识库（.pkl 文件）")
    uploaded_index = st.file_uploader(
        "上传 biorag_index.pkl",
        type=["pkl"],
        key="index_uploader",
    )
    if uploaded_index is not None:
        if st.button("⚡ 加载", use_container_width=True):
            try:
                new_chunks = load_index(uploaded_index.read())
                # 同样支持追加：如果已有知识库，合并进去
                if "chunks" in st.session_state and st.session_state.chunks:
                    existing_ids = {c.chunk_id for c in st.session_state.chunks}
                    added = [c for c in new_chunks if c.chunk_id not in existing_ids]
                    st.session_state.chunks += added
                    st.success(f"✅ 追加 {len(added)} 个chunk")
                else:
                    st.session_state.chunks = new_chunks
                    st.success(f"✅ 加载 {len(new_chunks)} 个chunk")
                st.rerun()
            except Exception as e:
                st.error(f"加载失败：{e}")

    st.divider()
    st.caption("本工具基于RAG架构，回答严格基于上传文献内容，不编造信息。")

# ── 主区域 ────────────────────────────────────────────────────────────────────
col_upload, col_qa = st.columns([1, 1.5], gap="large")

# ── 左列：上传文献 ────────────────────────────────────────────────────────────
with col_upload:
    st.subheader("📄 上传文献")

    # 提示当前知识库里已有哪些文件，避免重复上传
    if "chunks" in st.session_state and st.session_state.chunks:
        existing_sources = {c.source for c in st.session_state.chunks}
        st.caption(f"当前知识库已有 {len(existing_sources)} 篇，新上传文献将**追加**合并")

    uploaded_files = st.file_uploader(
        "支持多个PDF同时上传",
        type=["pdf"],
        accept_multiple_files=True,
        help="上传后点击「构建并追加」，新文献会合并到现有知识库"
    )

    if uploaded_files and api_key:
        # 检查哪些是新文件（未在当前知识库中）
        existing_sources = set()
        if "chunks" in st.session_state and st.session_state.chunks:
            existing_sources = {c.source for c in st.session_state.chunks}

        new_files = {f.name: f for f in uploaded_files
                     if f.name not in existing_sources}
        dup_files = [f.name for f in uploaded_files
                     if f.name in existing_sources]

        if dup_files:
            st.warning(f"以下文件已在知识库中，将跳过：{', '.join(dup_files)}")

        if new_files:
            btn_label = f"🚀 构建并追加（{len(new_files)} 篇新文献）"
            if st.button(btn_label, type="primary", use_container_width=True):
                client = OpenAI(api_key=api_key, base_url=BASE_URL)
                st.session_state.client = client

                pdf_bytes = {name: f.read() for name, f in new_files.items()}

                progress_bar = st.progress(0, text="开始处理...")
                status = st.empty()

                def on_progress(current, total, msg):
                    progress_bar.progress(
                        current / total if total > 0 else 0, text=msg
                    )
                    status.caption(msg)

                with st.spinner("处理中..."):
                    new_chunks = build_index(
                        pdf_bytes, client, EMBED_MODEL,
                        progress_callback=on_progress
                    )

                # 追加合并（不覆盖）
                if "chunks" in st.session_state and st.session_state.chunks:
                    st.session_state.chunks += new_chunks
                else:
                    st.session_state.chunks = new_chunks

                progress_bar.empty()
                status.empty()
                st.success(
                    f"✅ 新增 {len(new_chunks)} 个chunk，"
                    f"知识库共 {len(st.session_state.chunks)} 个chunk"
                )
                st.rerun()
        else:
            st.info("所有上传的文件已在知识库中")

    elif uploaded_files and not api_key:
        st.warning("请先在左侧输入 API Key")
    else:
        st.info("请选择PDF文件")

# ── 右列：问答 ────────────────────────────────────────────────────────────────
with col_qa:
    st.subheader("💬 文献问答")

    if "history" not in st.session_state:
        st.session_state.history = []

    # 确保client存在（加载知识库后可能还没有client）
    if "client" not in st.session_state and api_key:
        st.session_state.client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # 显示历史对话
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            with st.expander("📎 引用来源", expanded=False):
                for r in item["retrieved"]:
                    color = "gray" if r["filtered"] else "blue"
                    label = ("已过滤（相关度不足）" if r["filtered"]
                             else f"相关度 {r['score']}")
                    st.markdown(f"**:{color}[{r['source']}]** — {label}")
                    if not r["filtered"]:
                        st.caption(r["text"][:200] + "...")

    if "chunks" in st.session_state and st.session_state.chunks:
        if not api_key:
            st.warning("请在左侧输入 API Key 后才能提问")
        else:
            query = st.chat_input(
                "输入你的问题，例如：PTPN22通过什么机制影响JAK-STAT信号通路？"
            )
            if query:
                if "client" not in st.session_state:
                    st.session_state.client = OpenAI(api_key=api_key, base_url=BASE_URL)

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
                                st.markdown(
                                    f"**:gray[{r['source']}]** — "
                                    f"相关度 {r['score']}（低于阈值，已过滤）"
                                )
                            else:
                                st.markdown(
                                    f"**:blue[{r['source']}]** — 相关度 {r['score']}"
                                )
                                st.caption(r["text"][:200] + "...")

                st.session_state.history.append({
                    "question": query,
                    "answer": result["answer"],
                    "retrieved": result["retrieved_chunks"]
                })
    else:
        st.info("请先上传文献构建知识库，或加载已有知识库文件")
        st.divider()
        st.caption("可以问的问题示例：")
        for ex in [
            "PTPN22如何通过与14-3-3τ的结合影响JAK-STAT信号通路？",
            "PTPN22 R620W变异是什么样的突变？",
            "14-3-3蛋白家族一共有几个亚型？",
            "检测T细胞增殖能力用的是什么实验方法？",
        ]:
            st.caption(f"• {ex}")
