# Failure analysis — FlatRAG vs GraphRAG

## 1. FlatRAG thất bại: `G5000-34`

Question yêu cầu so sánh nhà cung cấp model/technology cho Google Cloud và Amazon. FlatRAG trả “không đủ thông tin” và nhận 1/5 ở cả ba tiêu chí. Root cause là vector retrieval lấy các đoạn tương tự riêng lẻ nhưng không gom được evidence từ hai nguồn. GraphRAG nối các evidence: Meta–Llama 2/Code Llama, TII–Falcon LLM, Anthropic–Claude 2 cho Google Cloud và Cohere cho Amazon; kết quả 5/5 ở cả ba tiêu chí. Mitigation được xác nhận là hybrid graph + vector context có provenance.

## 2. GraphRAG khó khăn: `G5000-45`

Question yêu cầu tránh double-count event khi hai row nói về việc Thales chọn L&T Technology Services và Qualcomm. Cả hai câu trả lời đều nêu cần hợp nhất event, nhưng GraphRAG nhận multi-hop reasoning 1/5. Root cause: graph hiện biểu diễn triple trực tiếp, chưa có `Event` node/khóa dedup để hợp nhất nhiều nguồn cho cùng sự kiện. Khắc phục: thêm node `Event`, event fingerprint `(participants, relation, normalized_date/source cluster)`, rồi gắn nhiều provenance vào event/edge thay vì tạo quan hệ độc lập.

## Kết luận

GraphRAG hữu ích khi evidence phải nối qua nhiều entity/document; nó không tự chữa được thiếu extraction hoặc schema chưa biểu diễn event. Các rationale và answer nguyên văn được lưu ở `outputs/graphrag_eval_results.csv`.
