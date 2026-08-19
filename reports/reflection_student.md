# Báo Cáo Reflection & Kế Hoạch Đồ Án — Lab 19: GraphRAG vs Flat RAG

**Học viên:** AICB-K34 Student  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  

---

## 📌 1. Mapping Khái Niệm Bài Giảng vào Source Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|---|---|---|---|
| **Conservative Coreference** | Module 1 | `src/preprocessing.py: resolve_coref_batch()` | Giảm thiểu hallucination bằng cách chỉ thay thế khi tiền ngữ xuất hiện rõ ràng trong chunk; lọc trước bằng regex đại từ để tiết kiệm API. |
| **Schema & Allowlist Guard** | Module 2 | `src/config.py: ALLOWED_NODE_TYPES, ALLOWED_RELATIONS` | Đảm bảo 100% quan hệ và nhãn tuân thủ schema ontology chuẩn, loại bỏ các quan hệ tự do gây rác đồ thị. |
| **Bulk Cypher Ingestion** | Module 2 | `src/extraction.py: bulk_insert_nodes(), bulk_insert_edges()` | Sử dụng cú pháp `UNWIND $rows AS row` theo batch 1.000 records giúp tốc độ nạp vào Neo4j đạt hàng nghìn quan hệ/giây. |
| **Entity Resolution & Union-Find** | Module 3 | `src/extraction.py: build_resolution_map(), UF, merge_guard()` | Kết hợp FAISS vector similarity + Lexical guard ngăn chặn false-merge giữa Công ty và Sản phẩm công nghệ. |
| **Super-node Degree Cap** | Module 4 | `src/retrieval.py: retrieve_graph_context(), node_degree()` | Giới hạn 50 cạnh mới nhất tại các node có bậc $>100$, ngăn ngừa bùng nổ token ngữ cảnh. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `src/evaluation.py: judge_answer(), run_evaluation()` | Chấm điểm khách quan trên 3 tiêu chí với reference answer làm neo chuẩn, xuất bảng so sánh chi tiết. |

---

## 📌 2. Quá trình Debugging & Bài học Rút ra
- **Lỗi kỹ thuật phức tạp nhất:** Vấn đề Rate Limiting (HTTP 429) và lỗi JSON schema validation trên các mô hình LLM mở, cùng với việc xử lý encoding UTF-8 trên môi trường Windows PowerShell.
- **Cách xử lý thành công:** 
  1. Xây dựng cơ chế *Exponential Backoff with Jitter* và tự động chuyển đổi mô hình dự phòng (*Model Fallback*) trong `src/llm_client.py`.
  2. Bổ sung cơ chế fallback parsing JSON: nếu chế độ `json_mode=True` bị từ chối, hệ thống tự động gọi chat thông thường và trích xuất chuỗi JSON `{...}` qua hàm `parse_json_object()`.
  3. Cấu hình `sys.stdout.reconfigure(encoding='utf-8')` cho toàn bộ các script Python chạy trên hệ điều hành Windows.

---

## 📌 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** **Enterprise Intelligence & Financial M&A Knowledge Graph (FinGraphRAG)**
- **Đặc thù bài toán & Lý do chọn giải pháp:**
  - Bài toán phân tích báo cáo tài chính, tin tức đầu tư và chuỗi cung ứng doanh nghiệp đòi hỏi khả năng kết nối nhiều mắt xích (ví dụ: *Công ty A đầu tư vào B $\rightarrow$ B cung cấp chip cho C $\rightarrow$ C bị ảnh hưởng bởi chính sách thuế*).
  - Flat RAG thông thường thất bại hoàn toàn trong việc truy vết các liên kết bắc cầu nhiều bước này, do đó GraphRAG là bắt buộc.
- **Cấu trúc Node & Relation dự kiến:**
  - **Nodes:** `Company`, `Person` (Executives/Investors), `IndustrySector`, `FinancialEvent` (Funding, M&A, IPO), `Product/Technology`.
  - **Relations:** `INVESTED_IN` (kèm số tiền, vòng gọi vốn), `ACQUIRED`, `FOUNDED`, `SUPPLIES_TO`, `COMPETES_WITH`, `OPERATES_IN`.
- **Chiến lược xử lý Super-node & Entity Resolution:**
  - Áp dụng cơ chế **Edge Weighting & Temporal Decay** (giảm trọng số các quan hệ quá cũ đối với các tập đoàn lớn như Vingroup, FPT, Viettel).
  - Sử dụng Mã số thuế doanh nghiệp / Mã cổ phiếu (Ticker) làm Unique Primary Key cho bước Entity Resolution để đạt độ chính xác 100% về mặt định danh pháp nhân.
