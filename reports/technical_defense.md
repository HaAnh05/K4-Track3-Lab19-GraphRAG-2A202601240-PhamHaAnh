# 10 Câu Hỏi Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Hà Anh  
**Mã học viên:** 2A202601240  
**Môi trường:** Python 3.11 (Conda `rag`) + Neo4j AuraDB + Groq (`openai/gpt-oss-20b` / `qwen/qwen3.6-27b`)  

---

### Câu 1: Coreference Resolution Failure Case
- **Tình huống thực tế:** Trong bài viết về Newlinks Technology Limited (`art_724458ce3e726428c0b2`), văn bản xuất hiện: *“Newlinks Technology Limited announced its financial results. The Company is a subsidiary of Newlinks Technology Limited...”*
- **Hiện tượng:** Cụm từ *“The Company”* và đại từ *“its”* bị mô hình phân giải thành thực thể độc lập `The Company` thay vì `Newlinks Technology Limited`, tạo ra triple rác: `Newlinks Technology Limited [Company] -ACQUIRED-> The Company [Company]`.
- **Hậu quả đối với Knowledge Graph:** Tạo ra False Edge và biến `The Company` thành một pseudo-supernode gây loãng ngữ cảnh và hallucination khi BFS traversal.

---

### Câu 2: Entity Resolution Threshold & Lexical Guard
- **Ngưỡng Cosine Similarity:** Chọn `threshold = 0.90` với embedding `sentence-transformers/all-MiniLM-L6-v2` + FAISS ANN search.
- **Cặp thực thể bị Lexical Guard chặn gộp:**  
  - `Youtility` (`Company`) vs `Youtility Technology` (`Technology`) — *Similarity: ~0.892*
  - `Worldpay` (`Company`) vs `Worldpay technology services` (`Technology`) — *Similarity: ~0.887*
  - `Apple` (`Company`) vs `Apple Watch` (`Technology`) — *Similarity: ~0.865*
- **Lý do chặn:** Do khác nhãn (`Company` vs `Technology`). Nếu gộp sai, quan hệ `DEVELOPED` hoặc `USES` sẽ bị triệt tiêu thành Self-loop.

---

### Câu 3: Super-node Mitigation & Temporal Capping
- **Top Super-nodes thực tế:** Raleon (degree 4), Aqara (degree 4), Renovus (degree 3), SoftBank (degree 3), Synopsys (degree 3), Bain (degree 3), OpenAI (degree 3), Meta (degree 3), Google (degree 3), Microsoft (degree 3).
- **Ưu điểm:** Khống chế tràn token (Token Explosion), ưu tiên quan hệ mới nhất phản ánh hiện trạng doanh nghiệp.
- **Rủi ro:** Bỏ sót các quan hệ lịch sử quan trọng trong quá khứ xa.

---

### Câu 4: Đánh Đổi Quality vs Latency vs Token Usage
- **Flat RAG:** Nhanh khi câu hỏi đơn giản, nhưng token usage trung bình cao (924.5 tokens/câu) do nhồi nhét chunk thô, dễ mất ngữ cảnh khi câu hỏi phân tán nhiều tài liệu.
- **GraphRAG:** Tiết kiệm token (-169.4 tokens/câu), độ trễ truy vấn tổng thể nhanh hơn (-2.28s) nhờ context đồ thị tuyến tính hóa cô đọng, vượt trội ở các câu hỏi Factoid (+0.50 Comprehensiveness, +1.00 Faithfulness, +0.50 Multi-hop).

---

### Câu 5: Quyết Định Từ Chối Đề Xuất của AI Coding Agent
1. *Từ chối tính Pairwise Cosine $O(N^2)$:* Yêu cầu dùng FAISS ANN Search để giảm độ phức tạp xuống $O(N \log N)$ và tránh tràn RAM.
2. *Từ chối chèn từng dòng đơn lẻ vào Neo4j:* Chuyển sang cú pháp `UNWIND $rows AS row` theo batch 1.000 records.

---

### Câu 6: Scalability khi Nâng Quy Mô lên 350MB (~100.000 bài báo)
1. **Async Extraction Pipeline:** Sử dụng hàng đợi Celery/Redis Queue kết hợp vLLM chạy local SLM (Llama-3-8B) trích xuất song song.
2. **Entity Resolution Blocking:** Áp dụng MinHash LSH hoặc Token Prefix Blocking trước khi tính vector.
3. **Hierarchical Graph Partitioning:** Phân cụm cộng đồng Leiden sinh Community Summaries hỗ trợ Global Search.

---

### Câu 7: Schema Ontology & Provenance Integrity
- **Schema Allowlist:** Khóa chặt 3 nhãn (`Company`, `Person`, `Technology`) và 8 quan hệ chuẩn (`ACQUIRED`, `DEVELOPED`, `INVESTED_IN`, `FOUNDED`, `WORKED_AT`, `PARTNERED_WITH`, `USES`, `LEADS`).
- **100% Provenance:** Mọi quan hệ bắt buộc có `source_chunk_id`, `published_date`, `evidence`, `confidence`. Đồ thị kiểm tra đạt `invalid_provenance_edges = 0`.

---

### Câu 8: Seed Entity Extraction & Fuzzy Matching Fallback
- Khi người dùng hỏi câu hỏi phức tạp, LLM trích xuất danh sách thực thể hạt nhân (Seeds).
- Tìm kiếm chính xác trên `name_norm` / `aliases_norm`, nếu không có thì fallback sang vector similarity $\ge 0.66$ trên `entity_match_store`.

---

### Câu 9: LLM-as-a-Judge Alignment
- Sử dụng Reference Answer làm neo chuẩn (Correctness Anchor) để chấm điểm Comprehensiveness, Faithfulness và Multi-hop reasoning trên thang điểm 1–5 kèm giải thích Rationale.

---

### Câu 10: Global Community Search vs Local Traversal
- Local Traversal (BFS) phục vụ câu hỏi vi mô (Entity-specific 1-hop / 2-hop).
- Global Search (Community Detection 70 cụm qua NetworkX Greedy Modularity) phục vụ câu hỏi vĩ mô tổng hợp toàn hệ thống.
