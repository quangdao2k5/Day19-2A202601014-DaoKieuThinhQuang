# Thuyết minh kỹ thuật — Lab 19

1. **Dataset và preprocessing.** Pipeline dùng 10.000 dòng HackerNoon đã stream về local, exact-dedup từ 4.728 xuống 4.071 bài, sau đó chunk rolling-window với provenance `article_id::cNNNN`.
2. **Coreference.** `resolve_coref_batch()` chỉ thay đại từ khi antecedent ở cùng chunk; các trường hợp mơ hồ giữ nguyên trong `unresolved_mentions`. Đây là lựa chọn ưu tiên precision để tránh false edge.
3. **Schema extraction.** `extract_batch()` ép JSON, allowlist ba type (`Company`, `Person`, `Technology`) và relation allowlist; mọi triple có evidence, confidence, source chunk và ngày.
4. **Bulk ingestion.** `bulk_insert_nodes()` và `bulk_insert_edges()` dùng `UNWIND $rows` theo batch, không gọi Cypher từng triple. Neo4j thực tế có 49 nodes, 39 edges, và 0 edge thiếu provenance.
5. **Entity resolution.** Exact/alias trước, FAISS candidate sau, threshold 0.90, lexical `SequenceMatcher >= 0.72`, rồi Union-Find. Audit có 89 candidate rows; cặp gần nhất `cloud-related services`/`cloud services` đạt 0.882 nhưng vẫn bị `REJECT_THRESHOLD`, nên không có bằng chứng để tuyên bố guard đã chặn cặp >0.85 trong sample này.
6. **Flat baseline.** `IndexFlatIP` với `all-MiniLM-L6-v2`, top-k=6, index 1.000 chunk để giữ runtime lab ổn định.
7. **Hybrid retrieval.** LLM extract seed, exact/fuzzy seed match, BFS 2-hop từ Neo4j, textualize edge cùng `source_chunk_id`, rồi ghép graph context với vector context.
8. **Super-node.** Chính sách code là degree >100 thì lấy tối đa 50 edge mới nhất; `GLOBAL_EDGE_CAP=250`. Graph hiện có degree cao nhất 7 nên nhánh cap chưa được kích hoạt trên sample, nhưng policy được kiểm tra bằng hàm `test_supernode_policy()`.
9. **Benchmark.** 25/25 Golden questions có đủ factoid (2), multi-hop (12), cross-doc (11), answer reference và LLM-Judge rationale.
10. **Trade-off/scale.** GraphRAG tăng 194.96 token/câu nhưng tăng completeness +1.08, faithfulness +1.52, multi-hop +1.00. Ở 100k bài cần queue extraction, checkpoint, ANN/HNSW và partition theo thời gian thay vì cosine mọi cặp O(N²).
