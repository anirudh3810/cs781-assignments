# H2 — actual RAG component smoke checks before the full embedding batch.
import numpy as np
import faiss
assert chunk_text("abcde", 2) == ["ab", "cd", "e"]
assert chunk_text("", 2048) == []
assert "".join(chunk_text(text, 2048)) == text
assert any(doc["kind"] == "html" for doc in corpus["documents"]), "No genuine HTML extraction"
assert any(doc["kind"] == "pdf" for doc in corpus["documents"]), "No genuine PDF extraction"
small_chunks = chunks[:min(3, len(chunks))]
small_embeddings = normalized_f32(embedding_model.encode(small_chunks, batch_size=2, convert_to_numpy=True), 384)
assert np.isfinite(small_embeddings).all() and np.allclose(np.linalg.norm(small_embeddings, axis=1), 1, atol=1e-5)
small_index = faiss.IndexFlatIP(384)
small_index.add(small_embeddings)
small_d, small_i = small_index.search(small_embeddings[:1], min(2, len(small_chunks)))
assert small_i[0, 0] == 0 and small_d[0, 0] > 0.99
small_index_copy = faiss.deserialize_index(faiss.serialize_index(small_index))
assert small_index_copy.ntotal == len(small_chunks)
smoke_answer = _run_qwen_impl(build_rag_prompt("Briefly describe the provided evidence.", small_chunks[0][:500]), 48)
assert isinstance(smoke_answer, str) and smoke_answer.strip()
print("Real local RAG smoke answer:", smoke_answer)
record_check("rag_smoke", sample_rows=len(small_chunks), embedding_dimension=384)
del small_embeddings, small_index, small_index_copy
