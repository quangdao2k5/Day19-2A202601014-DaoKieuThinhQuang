# Reflection và Action Plan — Đào Kiều Thịnh Quang

| Nội dung bài giảng | Code | Bài học |
|---|---|---|
| Conservative coreference | `resolve_coref_batch()` | Precision quan trọng hơn coverage khi edge sẽ được traversal. |
| Schema/allowlist | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Giảm graph schema drift từ output LLM. |
| Entity resolution | `build_resolution_map()`, `UF` | ANN cần guard/audit, không được merge chỉ vì embedding gần. |
| Neo4j ingestion | `bulk_insert_*()` | `UNWIND` giảm round-trip và giữ provenance. |
| Hybrid retrieval/judge | `answer_graph_rag()`, `judge_answer()` | Đánh giá phải có rationale theo từng query. |

Lỗi khó nhất là quota Groq và giới hạn runner ngắn, làm các bước LLM dễ mất tiến trình. Cách giải quyết là dùng OpenAI có trả phí, cache FAISS offline và checkpoint từng Golden question; khi Neo4j tạm unavailable chỉ retry phần thất bại.

Đồ án phù hợp là trợ lý tra cứu sự kiện/công nghệ doanh nghiệp nhiều tài liệu. Schema dự kiến: `Company`, `Person`, `Technology`, `Product`, `Event`, `Document`; event và relation đều có date/provenance. Entity resolution dùng alias + ANN candidate + human review ở vùng sát threshold. Super-node dùng cap theo thời gian, domain filter và tăng hop có kiểm soát; FlatRAG vẫn là fallback khi graph thiếu cạnh.
