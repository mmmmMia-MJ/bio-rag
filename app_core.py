# -*- coding: utf-8 -*-
"""
BioRAG 核心逻辑模块
"""
from __future__ import annotations
import re, time, io, pickle, logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BioRAG")

# ── 常量 ─────────────────────────────────────────────────────────────────────
MAX_EMBED_CHARS = 600
BATCH_SIZE = 50
DEFAULT_SIM_THRESHOLD = 0.45

# ── 分句保护用占位符 ─────────────────────────────────────────────────────────
_DOT = "⟨DOT⟩"
_ABBREVIATIONS = [
    "Fig", "Tab", "Eq", "al", "vs", "etc", "Dr", "Mr", "Mrs",
    "No", "Vol", "Ref", "Suppl", "approx", "min", "max",
]
_ABBR_RE = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in _ABBREVIATIONS) + r')\.',
    re.IGNORECASE,
)


# ── 数据结构 ─────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    page_num: int = -1
    embedding: Optional[List[float]] = field(default=None, repr=False)


# ── PDF 解析 ─────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text()
        if t and t.strip():
            pages.append((i + 1, t))
    return pages


# ── 智能分句 ─────────────────────────────────────────────────────────────────
def smart_split_sentences(text: str) -> List[str]:
    protected = text
    protected = re.sub(r'(\d)\.(\d)', rf'\1{_DOT}\2', protected)
    protected = re.sub(r'e\.g\.', f'e{_DOT}g{_DOT}', protected)
    protected = re.sub(r'i\.e\.', f'i{_DOT}e{_DOT}', protected)
    protected = _ABBR_RE.sub(rf'\1{_DOT}', protected)
    protected = re.sub(r'(?<=\s)([A-Za-z0-9])\.(?=\s)', rf'\1{_DOT}', protected)

    parts = re.split(r'(?<=[。；！？.!?])\s*|\n+', protected)

    sentences = []
    for p in parts:
        p = p.replace(_DOT, ".").strip()
        if p:
            sentences.append(p)
    return sentences


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, doc_id: str, source: str, page_num: int = -1,
               chunk_size: int = 500, overlap_sents: int = 2) -> List[Chunk]:
    sentences = smart_split_sentences(text)
    if not sentences:
        return []

    chunks = []
    idx = 0
    sent_i = 0

    while sent_i < len(sentences):
        buf = ""
        start_i = sent_i
        while sent_i < len(sentences):
            candidate = buf + ("" if not buf else "") + sentences[sent_i]
            if len(candidate) > chunk_size and buf:
                break
            buf = candidate
            sent_i += 1

        if buf.strip():
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk{idx}",
                doc_id=doc_id,
                source=source,
                text=buf.strip(),
                page_num=page_num,
            ))
            idx += 1

        overlap_start = max(start_i, sent_i - overlap_sents)
        if overlap_start >= sent_i:
            pass
        else:
            sent_i = overlap_start if sent_i < len(sentences) else sent_i

        if sent_i <= start_i:
            sent_i = start_i + 1

    return chunks


# ── 过滤无效 chunk ───────────────────────────────────────────────────────────
_REF_PATTERNS = re.compile(
    r'^(references|参考文献|bibliography|acknowledgements?|致谢)',
    re.IGNORECASE,
)

def filter_chunks(chunks: List[Chunk]) -> List[Chunk]:
    valid = []
    for c in chunks:
        t = c.text.strip()
        if len(t) < 30:
            continue
        letter_count = sum(1 for ch in t if '\u4e00' <= ch <= '\u9fff' or ch.isalpha())
        if letter_count < 10:
            continue
        if letter_count / max(len(t), 1) < 0.3:
            continue
        if _REF_PATTERNS.match(t):
            continue
        if re.match(r'^[\d\s\.\,\-\(\)]+$', t):
            continue
        valid.append(c)
    return valid


# ── Batch Embedding ──────────────────────────────────────────────────────────
def _embed_batch_with_retry(texts: List[str], client, embed_model: str,
                            max_retries: int = 3) -> List[Optional[List[float]]]:
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=embed_model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            logger.warning(f"Embedding batch 失败 (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                logger.error(f"Embedding batch 最终失败，返回 None")
                return [None] * len(texts)


# ── 构建向量索引 ─────────────────────────────────────────────────────────────
def build_index(pdf_files: dict, client, embed_model: str,
                progress_callback=None) -> List[Chunk]:
    all_chunks: List[Chunk] = []

    for filename, content in pdf_files.items():
        pages = extract_text_from_pdf(content)
        if not pages:
            logger.warning(f"PDF 无法提取文本: {filename}")
            continue
        doc_id = re.sub(r"[^\w]", "_", filename[:30])
        for page_num, page_text in pages:
            page_chunks = chunk_text(
                page_text, doc_id=doc_id, source=filename, page_num=page_num
            )
            all_chunks.extend(page_chunks)

    all_chunks = filter_chunks(all_chunks)
    total = len(all_chunks)
    logger.info(f"过滤后共 {total} 个有效 chunk，开始 embedding")

    if progress_callback:
        progress_callback(0, total, f"共 {total} 个有效chunk，开始batch embedding...")

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

    before = len(all_chunks)
    all_chunks = [c for c in all_chunks if c.embedding is not None]
    failed = before - len(all_chunks)
    if failed > 0:
        logger.warning(f"{failed} 个 chunk embedding 失败，已移除")

    logger.info(f"索引构建完成，最终 {len(all_chunks)} 个 chunk")
    return all_chunks


# ── 检索 ─────────────────────────────────────────────────────────────────────
def cosine_sim(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-9:
        return 0.0
    return float(np.dot(a, b) / norm)


def retrieve(query: str, chunks: List[Chunk], client, embed_model: str,
             top_k: int = 5, sim_threshold: float = DEFAULT_SIM_THRESHOLD):
    """
    v3 改进：
    - top_k 4→5 多检索一个候选
    - 在 retrieve 阶段就标记是否过阈值
    - 返回 List[Tuple[Chunk, float, bool]]  (chunk, score, is_relevant)
    """
    try:
        resp = client.embeddings.create(
            model=embed_model,
            input=query[:MAX_EMBED_CHARS]
        )
        query_emb = resp.data[0].embedding
    except Exception as e:
        logger.error(f"Query embedding 失败: {e}")
        return []

    scored = [(c, cosine_sim(query_emb, c.embedding)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    results = []
    for c, s in top:
        is_relevant = s >= sim_threshold
        results.append((c, s, is_relevant))
    return results


# ── Prompt 构建（修复 #6：system/user 分离）───────────────────────────────────
SYSTEM_PROMPT = (
    "你是一个严谨的生物医学文献问答助手。\n"
    "规则：\n"
    "1. 只能使用【检索到的文献片段】中的信息作答，严禁使用片段之外的知识编造内容。\n"
    "2. 如果所有片段都与问题无关或信息不足，必须明确回答「知识库中没有找到相关信息」。\n"
    "3. 回答时标注引用的来源文件名和chunk编号。\n"
    "4. 回答控制在300字以内，条理清晰。\n"
    "5. 专业术语使用规范写法。\n"
    "6. 如果问题涉及多个方面，分点作答。\n"
)


def build_prompt(query: str, retrieved) -> Tuple[str, str]:
    """
    返回 (system_msg, user_msg)，不再混在一起。
    retrieved: List[Tuple[Chunk, float, bool]]
    """
    relevant = [(c, s) for c, s, is_rel in retrieved if is_rel]
    filtered = [(c, s) for c, s, is_rel in retrieved if not is_rel]

    if not relevant:
        context_block = (
            "（未检索到任何相关片段，所有候选片段的相关度均低于阈值）\n"
            "以下是相关度最高但仍不足的片段，仅供参考：\n\n"
        )
        for c, s in filtered[:2]:
            context_block += f"[{c.chunk_id}] (来源: {c.source}, 页码: {c.page_num}, 相关度: {s:.3f})\n{c.text}\n\n"
    else:
        context_block = ""
        for c, s in relevant:
            context_block += (
                f"[{c.chunk_id}] (来源: {c.source}, 页码: {c.page_num}, "
                f"相关度: {s:.3f})\n{c.text}\n\n"
            )

    user_msg = (
        f"【检索到的文献片段】\n{context_block}\n"
        f"【用户问题】\n{query}\n\n"
        f"请基于以上片段作答："
    )
    return SYSTEM_PROMPT, user_msg


# ── 生成回答（修复 #6：system role 分离）─────────────────────────────────────
def generate_answer(system_msg: str, user_msg: str, client, gen_model: str,
                    max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=gen_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=800,
                temperature=0.3,
            )
            answer = resp.choices[0].message.content
            if answer and answer.strip():
                return answer
            logger.warning(f"生成返回空内容 (attempt {attempt+1})")
            time.sleep(5)
        except Exception as e:
            logger.error(f"生成失败 (attempt {attempt+1}): {e}")
            time.sleep(10)
    return "【模型多次重试后未返回有效内容，请稍后再试】"


# ── 主问答接口 ───────────────────────────────────────────────────────────────
def ask(query: str, chunks: List[Chunk], client,
        embed_model: str, gen_model: str,
        top_k: int = 5, sim_threshold: float = DEFAULT_SIM_THRESHOLD) -> dict:

    retrieved = retrieve(query, chunks, client, embed_model, top_k, sim_threshold)

    if not retrieved:
        return {
            "answer": "【查询失败，请检查 API Key 和网络连接】",
            "retrieved_chunks": [],
        }

    system_msg, user_msg = build_prompt(query, retrieved)
    answer = generate_answer(system_msg, user_msg, client, gen_model)

    return {
        "answer": answer,
        "retrieved_chunks": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "page_num": c.page_num,
                "score": round(s, 3),
                "text": c.text,
                "is_relevant": is_rel,
            }
            for c, s, is_rel in retrieved
        ],
    }


# ── 知识库保存 / 加载 ────────────────────────────────────────────────────────
def save_index(chunks: List[Chunk]) -> bytes:
    return pickle.dumps(chunks)


def load_index(data: bytes) -> List[Chunk]:
    return pickle.loads(data)
