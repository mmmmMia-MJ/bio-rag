# -*- coding: utf-8 -*-
"""
BioRAG核心逻辑模块 v2
=====================================================
主要优化：
- chunk_size 150 → 500（减少chunk数量，保留更多上下文）
- embedding 改为 batch 调用（每次最多50个chunk，大幅减少API调用次数）
- 删除 time.sleep(0.3)（不再人为减速）
- 过滤提前到chunking之后、embedding之前（垃圾chunk不进入embedding）
- embedding增加retry（防止API偶发失败）
"""
from __future__ import annotations
import re, time, io, pickle
from dataclasses import dataclass, field
from typing import List, Optional
from pypdf import PdfReader

# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    embedding: Optional[List[float]] = field(default=None, repr=False)

# ── PDF解析 ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, doc_id: str, source: str,
               chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    chunk_size 500：相比之前150，chunk数量减少约2/3，embedding调用次数大幅下降。
    overlap 50：保留前后文衔接，避免关键信息卡在边界。
    """
    sentences = re.split(r"(?<=[。；.!?\n])", text)
    sentences = [s for s in sentences if s.strip()]
    chunks, buf, idx = [], "", 0
    for sent in sentences:
        if len(buf) + len(sent) > chunk_size and buf:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk{idx}",
                doc_id=doc_id, source=source, text=buf.strip()
            ))
            idx += 1
            buf = buf[-overlap:] if overlap < len(buf) else buf
        buf += sent
    if buf.strip():
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk{idx}",
            doc_id=doc_id, source=source, text=buf.strip()
        ))
    return chunks

# ── 过滤无效chunk ─────────────────────────────────────────────────────────────

def filter_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    过滤提前：在embedding之前就清掉垃圾chunk，不浪费API调用。
    条件：总长度>=15字符，且中英文字母>=5个（排除页眉/纯数字/乱码）。
    """
    valid = []
    for c in chunks:
        t = c.text.strip()
        chinese_en = sum(1 for ch in t if '\u4e00' <= ch <= '\u9fff' or ch.isalpha())
        if len(t) >= 15 and chinese_en >= 5:
            valid.append(c)
    return valid

# ── Batch Embedding ───────────────────────────────────────────────────────────

MAX_EMBED_CHARS = 400
BATCH_SIZE = 50

def _embed_batch_with_retry(texts: List[str], client, embed_model: str,
                             max_retries: int = 3) -> List[Optional[List[float]]]:
    """
    batch调用embedding API，失败时整批retry。
    返回长度与texts相同的列表，失败的位置返回None。
    """
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=embed_model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return [None] * len(texts)

# ── 构建向量索引（主函数）────────────────────────────────────────────────────

def build_index(pdf_files: dict, client, embed_model: str,
                progress_callback=None) -> List[Chunk]:
    """
    pdf_files: {filename: bytes}
    progress_callback: callable(current, total, msg)
    """
    all_chunks: List[Chunk] = []

    # 1. 解析 + chunking
    for filename, content in pdf_files.items():
        text = extract_text_from_pdf(content)
        if not text:
            continue
        doc_id = re.sub(r"[^\w]", "_", filename[:30])
        chunks = chunk_text(text, doc_id=doc_id, source=filename)
        all_chunks.extend(chunks)

    # 2. 过滤（提前，不进embedding）
    all_chunks = filter_chunks(all_chunks)
    total = len(all_chunks)

    if progress_callback:
        progress_callback(0, total, f"共 {total} 个有效chunk，开始batch embedding...")

    # 3. Batch embedding
    done = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch = all_chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c.text[:MAX_EMBED_CHARS] for c in batch]
        embeddings = _embed_batch_with_retry(texts, client, embed_model)
        for chunk, emb in zip(batch, embeddings):
            chunk.embedding = emb
        done += len(batch)
        if progress_callback:
            progress_callback(done, total, f"embedding {done}/{total}")

    # 4. 移除失败的chunk
    all_chunks = [c for c in all_chunks if c.embedding is not None]
    return all_chunks

# ── 检索 ──────────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def retrieve(query: str, chunks: List[Chunk], client, embed_model: str,
             top_k: int = 4, sim_threshold: float = 0.4):
    resp = client.embeddings.create(model=embed_model, input=query[:MAX_EMBED_CHARS])
    query_emb = resp.data[0].embedding
    scored = [(c, cosine_sim(query_emb, c.embedding)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# ── Prompt + 生成 ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一个严谨的生物医学文献问答助手。请严格依据下面提供的【检索到的文献片段】回答问题。\n"
    "规则：\n"
    "1. 只能使用片段中出现的信息作答，不允许使用片段之外的知识编造内容。\n"
    "2. 如果检索到的片段都与问题无关或信息不足，必须明确回答「知识库中没有找到相关信息」，禁止编造。\n"
    "3. 回答时请标注引用的来源文件名。\n"
    "4. 回答请控制在200字以内，条理清晰。\n"
    "5. 专业术语请使用规范写法，不要重复堆砌。\n"
)

def build_prompt(query: str, retrieved, sim_threshold: float = 0.4) -> str:
    useful = [(c, s) for c, s in retrieved if s >= sim_threshold]
    if not useful:
        context_block = "（未检索到任何相关片段）"
    else:
        context_block = "\n\n".join(
            f"[{c.chunk_id}] (来源: {c.source}, 相关度: {s:.3f})\n{c.text}"
            for c, s in useful
        )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"【检索到的文献片段】\n{context_block}\n\n"
        f"【用户问题】\n{query}\n\n"
        f"请基于以上片段作答："
    )

def generate_answer(prompt: str, client, gen_model: str,
                    max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=gen_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
            )
            answer = resp.choices[0].message.content
            if answer and answer.strip():
                return answer
            time.sleep(5)
        except Exception:
            time.sleep(15)
    return "【模型多次重试后未返回有效内容，请稍后再试】"

def ask(query: str, chunks: List[Chunk], client,
        embed_model: str, gen_model: str,
        top_k: int = 4, sim_threshold: float = 0.4) -> dict:
    retrieved = retrieve(query, chunks, client, embed_model, top_k, sim_threshold)
    prompt = build_prompt(query, retrieved, sim_threshold)
    answer = generate_answer(prompt, client, gen_model)
    return {
        "answer": answer,
        "retrieved_chunks": [
            {"chunk_id": c.chunk_id, "source": c.source,
             "score": round(s, 3), "text": c.text,
             "filtered": s < sim_threshold}
            for c, s in retrieved
        ]
    }

# ── 知识库保存 / 加载 ─────────────────────────────────────────────────────────

def save_index(chunks: List[Chunk]) -> bytes:
    return pickle.dumps(chunks)

def load_index(data: bytes) -> List[Chunk]:
    return pickle.loads(data)
