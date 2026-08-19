# Báo cáo Lab 19 — GraphRAG vs Flat RAG

**Học viên:** Đào Kiều Thịnh Quang

**Khóa học:** AICB-K34 · Track 3: GraphRAG

**Ngày thực hiện:** 19/08/2026

## 1. Thuyết minh kỹ thuật và phân tích lỗi

### 1. Coreference Resolution

Pipeline dùng phân giải đại từ theo hướng *conservative*: chỉ thay thế khi mô hình trả về thực thể rõ ràng; còn lại giữ nguyên và ghi `unresolved_mentions`. Ví dụ chunk `https://www.galvnews.com/news_ap/business/samsung-showcases-groundbreaking-logic-innovations-at-system-lsi-tech-day-2023/article_1f2bea2e-cdcd-518d-811f-4a6729d8a3de.html::c0000` nói về Samsung Electronics rồi dùng tham chiếu sở hữu “its blueprint”. Mẫu chạy này không ép thay tham chiếu đó vì có nguy cơ gán nhầm thành tên sản phẩm/công nghệ gần nhất. Đánh đổi là có thể bỏ sót một triple; nhưng tránh false edge kiểu `Technology --ANNOUNCED--> product` thay vì `Samsung --ANNOUNCED--> product`.

### 2. Entity Resolution Threshold và Lexical Guard

Ngưỡng cosine của entity matching là `0.90`; chỉ gộp khi type tương thích và lexical guard xác nhận token đặc trưng. Audit lần chạy này không có cặp nào vượt ngưỡng rồi bị reject (`outputs/entity_resolution_audit.csv` chỉ có header), vì graph nhỏ 39 cạnh sau allowlist. Tuy vậy guard vẫn cần thiết: `OpenAI` và `OpenAI’s app store` có embedding gần nhau nhưng không được gộp vì một bên là `Company`, bên kia là khái niệm/sự kiện. Không dùng guard sẽ làm query về marketplace bị quy về node công ty, làm sai quan hệ `PLANNED_MARKETPLACE`.

### 3. Đồ thị và Super-node Mitigation

| Hạng | Thực thể | Type | Degree |
|---:|---|---|---:|
| 1 | Google Cloud | Company | 7 |
| 2 | big data system and cloud platform | Technology | 5 |
| 3 | OpenAI | Company | 4 |

Số liệu lấy từ `outputs/top_degree_entities.csv`. Khi degree vượt ngưỡng, traversal lấy tối đa 50 cạnh mới nhất theo `published_date`. Cách này giữ context ngắn, ưu tiên tin mới và tránh một node phổ biến làm lấn át mọi evidence. Rủi ro: câu hỏi lịch sử có thể mất cạnh cũ; hệ production cần query theo khoảng thời gian hoặc tăng cap có điều kiện.

### 4. So sánh thực nghiệm

Đã chạy **25/25** Golden questions do giảng viên cập nhật, gồm factoid, multi-hop và cross-document. Judge dùng OpenAI, generator dùng `gpt-4.1-mini`; mọi câu trả lời và rationale nằm trong `outputs/graphrag_eval_results.csv`.

| Tiêu chí | Flat RAG | GraphRAG | Δ Graph − Flat |
|---|---:|---:|---:|
| Comprehensiveness (1–5) | 1.60 | 2.68 | +1.08 |
| Faithfulness (1–5) | 1.68 | 3.20 | +1.52 |
| Multi-hop reasoning (1–5) | 1.76 | 2.76 | +1.00 |
| Latency trung bình (s) | 4.77 | 4.16 | -0.61 |
| Token usage trung bình | 766.92 | 961.88 | +194.96 |

Kết quả mạnh nhất là nhóm multi-hop: completeness **3.17 vs 1.08**, faithfulness **3.83 vs 1.08**. Ví dụ `G5000-34`: FlatRAG không tìm được evidence liên kết Google Cloud và Amazon, còn GraphRAG nối các node/edge để nêu Meta–Llama 2/Code Llama, TII–Falcon LLM, Anthropic–Claude 2 và Cohere cho Amazon.

Ca lỗi GraphRAG điển hình là `G5000-45`. Cả hai hệ đều đề xuất hợp nhất event để tránh double count, nhưng GraphRAG nhận điểm reasoning thấp hơn (1 so với 4) vì graph extraction chưa mô hình hóa event node chung và provenance đủ chi tiết cho hai row 261/891. Khắc phục: thêm node `Event`, khóa dedup theo `(subject, relation, object, normalized_date/source cluster)`, và lưu nhiều provenance trên cùng edge.

### 5. Trade-off, Agent Control và scale 350MB

GraphRAG đổi thêm chi phí extraction, entity resolution và khoảng **195 token/câu** để lấy chất lượng evidence tốt hơn. Với graph nhỏ, latency generation đo được không cao hơn FlatRAG; khi scale, thời gian xây graph sẽ là overhead chính.

Một đề xuất agent đã không áp dụng là so cosine mọi cặp thực thể (`O(N²)`) trên toàn bộ dữ liệu. Cách đó dễ tràn RAM và tạo merge sai. Thay vào đó dùng exact normalization trước, ANN/vector candidate sau, rồi lexical/type guard và audit log.

Với 350MB (~100k bài), bottleneck đầu tiên là LLM extraction + embedding chứ không phải Cypher. Hướng xử lý: hàng đợi async có rate limit/retry, checkpoint theo batch, HNSW/FAISS ANN, `UNWIND` ingestion theo lô, partition theo thời gian/nguồn và re-index tăng dần.

## 2. Reflection và kế hoạch áp dụng

| Khái niệm | Module/hàm | Quan sát thực tế |
|---|---|---|
| Conservative coreference | `resolve_coref_batch()` | Giảm false edge, đổi lại có unresolved mention cần audit. |
| Schema allowlist | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Chặn relation tự do, giúp Cypher/query ổn định. |
| Bulk Cypher ingestion | `bulk_insert_nodes()`, `bulk_insert_edges()` | Đã ingest 49 nodes, 39 edges; provenance invalid = 0. |
| Entity resolution | `build_resolution_map()`, `UF` | Dùng threshold + lexical/type guard để không over-merge. |
| Super-node cap | `retrieve_graph_context()` | Kiểm soát context explosion và giữ evidence gần đây. |
| LLM-as-a-Judge | `judge_answer()` | Có score và rationale trên từng câu, không chỉ báo cáo trung bình. |

Lỗi khó nhất là quota Groq và giới hạn thời gian của runner khiến extraction/evaluation dễ dừng giữa chừng. Cách xử lý là chuyển generator/judge sang OpenAI có key trả phí, cache FAISS local, và checkpoint theo từng Golden question. Nhờ vậy một lỗi Neo4j tạm thời ở một câu chỉ cần retry câu đó, không mất 24 câu đã xong.

Trong đồ án thực tế, GraphRAG phù hợp cho hỏi đáp chính sách/sự kiện doanh nghiệp có quan hệ nhiều bước và nhiều tài liệu. Schema khởi đầu gồm `Company`, `Person`, `Technology`, `Product`, `Event`, `Document`; relation có provenance và thời gian. `Event` là phần cần ưu tiên để xử lý duplicate news. Với super-node, áp dụng cap theo thời gian kèm filter domain; với entity, dùng candidate ANN + human-review queue cho các merge confidence sát ngưỡng.

## 3. Tự đánh giá

| Tiêu chí | Điểm (1–5) | Ghi chú |
|---|---:|---|
| Hiểu GraphRAG | 4 | Giải thích được lúc graph giúp multi-hop và lúc evidence thiếu. |
| Kiểm soát AI coding agent | 4 | Giữ allowlist/audit/checkpoint, không chấp nhận shortcut O(N²). |
| Chất lượng knowledge graph | 3 | Provenance hợp lệ, nhưng extraction sample còn nhỏ và Event node chưa có. |
| Phân tích/debug | 4 | Đã xử lý quota, network retry, model routing và checkpoint runtime. |
