# Báo Cáo Phân Tích Ca Lỗi (Failure Analysis) — Lab 19: GraphRAG vs Flat RAG

**Học viên:** AICB-K34 Student  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  

---

## 🔍 Ca 1: Flat RAG Thất Bại — GraphRAG Thành Công Vượt Trội

### Thông tin ca truy vấn:
- **Question ID:** `G5000-31` (*Multi-hop Chronological Reasoning*)
- **Câu hỏi:** *“Order OpenAI's ecosystem moves from March through July 2023 using the selected sources: plug-ins, open-source model planning, marketplace planning, and AP collaboration.”*
- **Reference Answer:**  
  *March: ChatGPT gained support for about a dozen application plug-ins. May: OpenAI was reported to be preparing a new open-source language model. June: OpenAI was reported to plan an app-store/marketplace for developers to sell AI models built on its technology. July: AP and OpenAI announced a collaboration to share select news content and technology for generative-AI use cases.*

### Kết quả Thực nghiệm & Phân tích Root Cause:
1. **Phản hồi của Flat RAG:**  
   *"I'm sorry, but none of the supplied chunks contain information about OpenAI's plug-in releases, open-source model planning, marketplace planning, or AP collaboration between March and July 2023..."*  
   $\to$ **Điểm LLM Judge:** Faithfulness = 1/5, Multi-hop = 1/5, Comprehensiveness = 1/5.  
   - *Nguyên nhân gốc rễ (Root Cause):* Vector Similarity thuần túy bị phân mảnh. Câu hỏi nhắc tới nhiều mốc thời gian và khái niệm phân bố rải rác trên nhiều bài báo khác nhau, khiến Top-6 chunks vector chỉ thu thập được các bài báo rác có độ tương đồng từ khóa cục bộ.

2. **Phản hồi của GraphRAG:**  
   GraphRAG trích xuất hạt nhân `OpenAI`, duyệt các cạnh thời gian (`ORDER BY published_date DESC`), gom đủ các sự kiện liên quan và trả về bảng tổng hợp thời gian chi tiết từng tháng có trích dẫn `source_chunk_id` và `published_date`.  
   $\to$ **Điểm LLM Judge:** Comprehensiveness = 3/5, Faithfulness = 4/5, Multi-hop = 3/5.

---

## 🔍 Ca 2: Thách Thức của GraphRAG khi Thiếu Dữ Liệu Đồ Thị

### Thông tin ca truy vấn:
- **Question ID:** `G5000-26` (*Multi-hop Cross-entity*)
- **Câu hỏi:** *“What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capability is mentioned alongside it?”*

### Kết quả Thực nghiệm & Phân tích Root Cause:
1. **Hiện tượng:**  
   Cả Flat RAG và GraphRAG đều không tìm thấy thông tin chi tiết và phản hồi an toàn từ chối bịa đặt: *"Context does not contain sufficient information about Amazon's July AI expansion..."*
2. **Root Cause:**  
   Do giới hạn `EXTRACTION_MAX_CHUNKS = 400` trong tập dữ liệu mẫu, bài viết cụ thể về Amazon AI Expansion không nằm trong tập chunk trích xuất đồ thị.
3. **Bài học & Giải pháp Khắc phục:**  
   - GraphRAG phụ thuộc vào độ bao phủ (Extraction Coverage).  
   - Khi triển khai production, bắt buộc phải dùng **Hybrid GraphRAG (Subgraph Context + Top-K Vector Chunks Fallback)** để đảm bảo khi đồ thị thiếu mắt xích thì Vector Search vẫn có thể bù đắp thông tin thô.
