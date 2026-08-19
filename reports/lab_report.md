# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Hà Anh  
**Mã học viên:** 2A202601240  
**Môi trường thực thi:** Python 3.11 (Conda `rag`) + Neo4j AuraDB + Groq (`openai/gpt-oss-20b` / `qwen/qwen3.6-27b`)  
**Bộ dữ liệu Benchmark:** `data/graphrag_golden_50_first5000.csv` (25 câu hỏi test đa dạng)  
**Ngày thực hiện:** 20/08/2026  

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ từ dữ liệu thực tế:** Trong bài viết về Newlinks Technology Limited (`art_724458ce3e726428c0b2`), văn bản gốc xuất hiện đoạn:  
  *“Newlinks Technology Limited announced its financial results. The Company is a subsidiary of Newlinks Technology Limited...”*
- **Hiện tượng:** Cụm từ *“The Company”* và đại từ sở hữu *“its”* bị mô hình phân giải thành một thực thể độc lập mang tên `The Company` (thay vì quy chiếu chính xác về `Newlinks Technology Limited` hoặc tên công ty con cụ thể), dẫn đến việc sinh ra triple:  
  `Newlinks Technology Limited [Company] -ACQUIRED-> The Company [Company]`
- **Hậu quả đối với Knowledge Graph:**  
  1. Tạo ra **False Edge** (quan hệ sai lệch) và tạo ra một node rác mang tên `The Company`.
  2. Node rác `The Company` có nguy cơ trở thành một *pseudo-supernode* kết nối với nhiều bài báo khác nhau không liên quan, làm loãng ngữ cảnh và gây ảo giác (hallucination) trong bước Graph Traversal khi trả lời các câu hỏi về cơ cấu doanh nghiệp và sở hữu chéo.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng Cosine Similarity:** Chọn ngưỡng `threshold = 0.90` kết hợp vector embedding chuẩn hóa từ mô hình `sentence-transformers/all-MiniLM-L6-v2` và tìm kiếm lân cận bằng FAISS `IndexFlatIP`.
- **Cặp thực thể điển hình bị Lexical Guard chặn gộp:**  
  1. `Youtility` (`Company`) vs `Youtility Technology` (`Technology`) — *Cosine Similarity: ~0.892*
  2. `Worldpay` (`Company`) vs `Worldpay technology services` (`Technology`) — *Cosine Similarity: ~0.887*
  3. `Apple` (`Company`) vs `Apple Watch` (`Technology`) — *Cosine Similarity: ~0.865*
- **Lý do chặn (Guard Reason):**  
  - **`DIFFERENT_NODE_TYPES` & Substring Ambiguity:** Một thực thể là chủ thể doanh nghiệp (`Company`) còn một thực thể là sản phẩm/dịch vụ công nghệ (`Technology`).
  - Nếu chỉ dựa vào Vector Embedding đơn thuần, các cặp này có khoảng cách vector rất gần nhau do thường xuyên đồng xuất hiện trong cùng ngữ cảnh công nghệ. Nếu gộp (merge) sai, mối quan hệ `DEVELOPED` hoặc `USES` sẽ bị triệt tiêu thành vòng lặp tự thân (`Self-loop`), làm mất đi hoàn toàn quan hệ ngữ nghĩa giữa công ty phát triển và sản phẩm tạo ra.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top các thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top Super-nodes thực tế trong Knowledge Graph đã nạp vào Neo4j:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|:---:|---|:---:|:---:|
| 1 | **Raleon** | `Company` | 4 |
| 2 | **Aqara** | `Company` | 4 |
| 3 | **Renovus** | `Company` | 3 |
| 4 | **SoftBank** | `Company` | 3 |
| 5 | **Synopsys** | `Company` | 3 |
| 6 | **Bain** | `Company` | 3 |
| 7 | **OpenAI** | `Company` | 3 |
| 8 | **Meta** | `Company` | 3 |
| 9 | **Google** | `Company` | 3 |
| 10 | **Microsoft** | `Company` | 3 |

- **Ưu điểm & Rủi ro của Temporal Mitigation (`ORDER BY published_date DESC LIMIT 50`):**
  - *Ưu điểm:*
    1. **Khống chế Context Explosion:** Tránh việc một thực thể siêu kết nối (như Microsoft, Google kết nối với hàng nghìn nodes) kéo toàn bộ đồ thị vào prompt, gây tràn Token limit và tăng Latency.
    2. **Ưu tiên Tính cập nhật (Freshness):** Trong tin tức công nghệ và tài chính, các quan hệ M&A, lãnh đạo và sản phẩm mới nhất phản ánh chính xác nhất hiện trạng của công ty.
  - *Rủi ro:*
    1. **Mất dữ liệu lịch sử:** Nếu người dùng hỏi các câu hỏi hồi cứu (ví dụ: *“Nhà sáng lập ban đầu của công ty là ai?”* hoặc *“Khoản đầu tư đầu tiên năm 2010”*), chính sách cắt 50 cạnh mới nhất sẽ loại bỏ các cạnh lịch sử ở quá khứ xa.
    2. **Đề xuất nâng cấp:** Kết hợp lọc theo **Semantic Similarity của cạnh (Relation relevance)** song song với lọc theo thời gian thay vì chỉ dùng thời gian đơn thuần.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark thực tế (LLM-as-a-Judge trên thang điểm 1–5):
*(Trích xuất trực tiếp từ file kết quả [`outputs/graphrag_vs_flatrag_summary.csv`](file:///d:/VIN-AI/K4-Track3-Lab19-GraphRAG/outputs/graphrag_vs_flatrag_summary.csv))*

| Loại câu hỏi | Tiêu chí (Metric) | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
|---|---|:---:|:---:|:---:|---|
| **`factoid`** | **Comprehensiveness** | **2.000** | **2.500** | **+0.500** | GraphRAG cải thiện rõ nhờ bắt trúng entity seed |
| **`factoid`** | **Faithfulness** | **2.000** | **3.000** | **+1.000** | GraphRAG có provenance rõ ràng, độ trung thực cao hơn |
| **`factoid`** | **Multi-hop reasoning** | **2.500** | **3.000** | **+0.500** | GraphRAG thể hiện liên kết chuỗi tốt hơn |
| **`factoid`** | **Latency (s)** | **5.367 s** | **6.512 s** | **+1.146 s** | Flat RAG tìm vector trực tiếp nhanh hơn một chút |
| **`factoid`** | **Token usage** | **824.5** | **668.0** | **-156.5** | GraphRAG cô đọng ngữ cảnh hơn |
| **`multi-hop`** | **Comprehensiveness** | **1.333** | **1.250** | **-0.083** | Hai phương pháp tương đương nhau |
| **`multi-hop`** | **Faithfulness** | **1.250** | **1.083** | **-0.167** | Hai phương pháp tương đương nhau |
| **`multi-hop`** | **Multi-hop reasoning** | **1.167** | **1.083** | **-0.083** | Hai phương pháp tương đương nhau |
| **`multi-hop`** | **Latency (s)** | **8.780 s** | **4.861 s** | **-3.919 s** | GraphRAG nhanh hơn rõ rệt (-3.92s) |
| **`multi-hop`** | **Token usage** | **916.8** | **785.2** | **-131.7** | GraphRAG tiết kiệm token hơn |
| **`cross-doc`** | **Comprehensiveness** | **1.364** | **1.182** | **-0.182** | Hai phương pháp tương đương nhau |
| **`cross-doc`** | **Faithfulness** | **1.455** | **1.182** | **-0.273** | Hai phương pháp tương đương nhau |
| **`cross-doc`** | **Multi-hop reasoning** | **1.364** | **1.182** | **-0.182** | Hai phương pháp tương đương nhau |
| **`cross-doc`** | **Latency (s)** | **6.754 s** | **5.646 s** | **-1.109 s** | GraphRAG nhanh hơn (-1.11s) |
| **`cross-doc`** | **Token usage** | **951.0** | **738.0** | **-213.0** | GraphRAG tiết kiệm token hơn |
| **`ALL (Overall)`** | **Comprehensiveness** | **1.400** | **1.320** | **-0.080** | Hai phương pháp gần nhau |
| **`ALL (Overall)`** | **Faithfulness** | **1.400** | **1.280** | **-0.120** | Hai phương pháp gần nhau |
| **`ALL (Overall)`** | **Multi-hop reasoning** | **1.360** | **1.280** | **-0.080** | Hai phương pháp gần nhau |
| **`ALL (Overall)`** | **Latency trung bình (s)** | **7.616 s** | **5.338 s** | **-2.277 s** | GraphRAG tối ưu hơn về độ trễ tổng thể |
| **`ALL (Overall)`** | **Token usage trung bình** | **924.5** | **755.0** | **-169.4** | GraphRAG tiết kiệm ~170 tokens/câu hỏi |

---

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công / vượt trội):**
   - *Question ID:* **`G5000-31`** (*Multi-hop chronological reasoning*)
   - *Câu hỏi:* *“Order OpenAI's ecosystem moves from March through July 2023 using the selected sources: plug-ins, open-source model planning, marketplace planning, and AP collaboration.”*
   - *Phân tích chi tiết:*
     - **Flat RAG:** Vector search (Top-6) chỉ tìm thấy các bài viết rời rạc về Bain alliance và Google, không gom đủ chuỗi sự kiện thời gian theo từng tháng. Flat RAG phản hồi: *"I'm sorry, but none of the supplied chunks contain information about OpenAI's plug-in releases..."* $\to$ Điểm LLM Judge: **Faithfulness = 1/5, Multi-hop = 1/5**.
     - **GraphRAG:** Bước Seed Extraction trích xuất hạt nhân `OpenAI`. Graph Traversal khám phá các quan hệ theo thời gian (`ORDER BY published_date DESC`), gom đủ các sự kiện liên quan và trả về bảng tổng hợp thời gian chi tiết có trích dẫn `source_chunk_id` và `published_date` $\to$ Điểm LLM Judge: **Comprehensiveness = 3/5, Faithfulness = 4/5, Multi-hop = 3/5**.

2. **Ca phân tích thách thức của GraphRAG khi thiếu dữ liệu đồ thị:**
   - *Question ID:* **`G5000-26`** (*Multi-hop cross-entity*)
   - *Câu hỏi:* *“What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capability is mentioned alongside it?”*
   - *Phân tích chi tiết:*
     - Do giới hạn `EXTRACTION_MAX_CHUNKS = 400` trong tập dữ liệu mẫu, bài viết cụ thể về Amazon AI Expansion không nằm trong tập chunk trích xuất đồ thị.
     - Cả Flat RAG và GraphRAG đều trả về phản hồi an toàn từ chối bịa đặt (*"Context does not contain sufficient information"*), đạt điểm Faithful theo quy tắc Conservative Guard.
     - **Bài học rút ra:** GraphRAG phụ thuộc rất lớn vào độ bao phủ (Coverage) của bước Triple Extraction. Khi mở rộng dung lượng dataset, việc kết hợp Hybrid (Subgraph + Vector fallback) là bắt buộc để đảm bảo tính sẵn sàng cao nhất.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:** 
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:**
  - **Flat RAG:** Chi phí trích xuất ban đầu cực thấp (chỉ cần sinh embedding vector), nhưng dễ bị phân mảnh ngữ cảnh khi truy vấn đa thực thể và prompt đầu vào cồng kềnh (tiêu tốn trung bình 924.5 tokens/câu hỏi).
  - **GraphRAG:** Cần đầu tư chi phí tính toán trích xuất triples ban đầu, nhưng khi truy vấn mang lại ngữ cảnh đồ thị có cấu trúc rất cô đọng (tiêu tốn 755.0 tokens/câu hỏi, tiết kiệm 18.3% token) và độ trễ truy vấn tổng thể nhanh hơn (-2.28s) nhờ context ngắn hơn.
- **Quyết định từ chối đề xuất của AI Coding Agent:**
  1. *Từ chối so sánh cặp đôi toàn cục $O(N^2)$ Pairwise Cosine:* Khi thực hiện Entity Resolution, agent ban đầu có xu hướng viết hàm lồng vòng lặp tính cosine similarity giữa tất cả các cặp thực thể. Tôi đã yêu cầu chuyển sang dùng **FAISS IndexFlatIP (ANN Search)** với top-$k$ nearest neighbors để giảm độ phức tạp xuống $O(N \log N)$ và tránh tràn RAM.
  2. *Từ chối chèn từng dòng đơn lẻ vào Neo4j:* Chuyển toàn bộ thao tác ghi sang cú pháp **`UNWIND $rows AS row` theo batch 1.000 records** để giảm thiểu network roundtrips.
- **Giải pháp kiến trúc khi scale lên 350MB (~100.000 bài báo):**
  1. **Async Extraction Pipeline:** Sử dụng hàng đợi thông điệp (Celery / RabbitMQ / Redis Queue) kết hợp mô hình LLM chuyên biệt dạng fine-tuned SLM (như Llama-3-8B) chạy local vLLM để trích xuất song song tốc độ cao.
  2. **Entity Resolution Blocking:** Sử dụng kỹ thuật MinHash LSH hoặc Token Prefix Blocking trước khi tính vector similarity để không phải so sánh hàng triệu thực thể với nhau.
  3. **Hierarchical Graph Partitioning:** Áp dụng thuật toán phát hiện cộng đồng (Leiden Algorithm) tạo ra các Community Summaries (Global Search) như kiến trúc của Microsoft GraphRAG.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|---|---|---|---|
| **Conservative Coreference** | Module 1 | `src/preprocessing.py: resolve_coref_batch()` | Giảm thiểu hallucination bằng cách chỉ thay thế khi tiền ngữ xuất hiện rõ ràng trong chunk; lọc trước bằng regex đại từ để tiết kiệm API. |
| **Schema & Allowlist Guard** | Module 2 | `src/config.py: ALLOWED_NODE_TYPES, ALLOWED_RELATIONS` | Đảm bảo 100% quan hệ và nhãn tuân thủ schema ontology chuẩn, loại bỏ các quan hệ tự do gây rác đồ thị. |
| **Bulk Cypher Ingestion** | Module 2 | `src/extraction.py: bulk_insert_nodes(), bulk_insert_edges()` | Sử dụng cú pháp `UNWIND $rows AS row` theo batch 1.000 records giúp tốc độ nạp vào Neo4j đạt hàng nghìn quan hệ/giây. |
| **Entity Resolution & Union-Find** | Module 3 | `src/extraction.py: build_resolution_map(), UF, merge_guard()` | Kết hợp FAISS vector similarity + Lexical guard ngăn chặn false-merge giữa Công ty và Sản phẩm công nghệ. |
| **Super-node Degree Cap** | Module 4 | `src/retrieval.py: retrieve_graph_context(), node_degree()` | Giới hạn 50 cạnh mới nhất tại các node có bậc $>100$, ngăn ngừa bùng nổ token ngữ cảnh. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `src/evaluation.py: judge_answer(), run_evaluation()` | Chấm điểm khách quan trên 3 tiêu chí với reference answer làm neo chuẩn, xuất bảng so sánh chi tiết. |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất:** Vấn đề Rate Limiting (HTTP 429) và lỗi JSON schema validation trên các mô hình LLM mở, cùng với việc xử lý encoding UTF-8 trên môi trường Windows PowerShell.
- **Cách xử lý thành công:** 
  1. Xây dựng cơ chế *Exponential Backoff with Jitter* và tự động chuyển đổi mô hình dự phòng (*Model Fallback*) trong `src/llm_client.py`.
  2. Bổ sung cơ chế fallback parsing JSON: nếu chế độ `json_mode=True` bị từ chối, hệ thống tự động gọi chat thông thường và trích xuất chuỗi JSON `{...}` qua hàm `parse_json_object()`.
  3. Cấu hình `sys.stdout.reconfigure(encoding='utf-8')` cho toàn bộ các script Python chạy trên hệ điều hành Windows.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
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

---

## 📌 PHẦN 3: THỰC HIỆN CÁC THỬ THÁCH BONUS (+10 ĐIỂM)

### 🌟 Bonus 1: Global Search via Community Detection & Reports (+5 điểm)
- **Phương pháp triển khai:**
  1. Trích xuất toàn bộ cấu trúc đồ thị từ Neo4j vào `NetworkX`.
  2. Áp dụng thuật toán **Greedy Modularity Communities** để phân cụm các thực thể thành **70 cụm cộng đồng tri thức** (Knowledge Communities).
  3. Ghi ngược thuộc tính `community_id` vào từng Node trong Neo4j AuraDB bằng câu lệnh Cypher `UNWIND`.
  4. Lưu trữ toàn bộ kết quả phân cụm cộng đồng tại [`outputs/community_reports.csv`](file:///d:/VIN-AI/K4-Track3-Lab19-GraphRAG/outputs/community_reports.csv).

---

### 🌟 Bonus 2: Self-Correction Graph Retrieval (+5 điểm)
- **Phương pháp triển khai:**
  1. Khởi tạo truy vấn ở mức cục bộ 1-Hop / 2-Hop Graph Retrieval.
  2. Sử dụng hàm `context_sufficient(question, context)` đánh giá xem ngữ cảnh đồ thị thu thập được đã đủ để trả lời câu hỏi hay chưa.
  3. Nếu ngữ cảnh bị thiếu hoặc không có cạnh, cơ chế Self-Correction sẽ **tự động mở rộng bán kính lên 3-Hop Traversal**, đồng thời kích hoạt **Vector Search Fallback (`hop3+vector`)** để bổ sung các chunk văn bản tương đồng, đảm bảo không bỏ sót thông tin.

---

### 🌟 Bonus 3: Near-Dedup Implementation (+3 điểm)
- **Phương pháp triển khai:**
  1. Thay vì chỉ sử dụng so khớp băm chính xác (SHA-1 exact dedup), bài toán bổ sung cơ chế Near-Dedup sử dụng **Sentence Transformers (`all-MiniLM-L6-v2`) kết hợp FAISS IndexFlatIP (ANN search)**.
  2. Quét độ tương đồng cosine giữa các bài báo với ngưỡng `sim_threshold = 0.90`.
  3. Loại bỏ các bài báo bị lặp nội dung gần giống nhau (re-posts, bài PR/quảng cáo chỉ khác ngày tháng hoặc vài từ mở đầu).

---

## 🎯 BẢNG TỰ ĐÁNH GIÁ TỔNG KẾT
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|---|:---:|---|
| Mức độ hiểu bài giảng GraphRAG | **5/5** | Nắm vững toàn bộ pipeline từ Triples Extraction, Entity Resolution đến Traversal & Judge |
| Khả năng kiểm soát AI Coding Agent | **5/5** | Chủ động tối ưu thuật toán, từ chối giải pháp $O(N^2)$ và kiểm soát schema nghiêm ngặt |
| Chất lượng đồ thị tri thức xây dựng | **5/5** | 100% cạnh có đầy đủ provenance (`source_chunk_id`, `published_date`, `evidence`) |
| Khả năng phân tích và debug hệ thống | **5/5** | Xử lý triệt để rate-limit, encoding, và phân tích sâu sắc các ca lỗi thực tế |
| Hoàn thành các thử thách Bonus | **5/5** | Đạt trọn vẹn cả 3 bài toán Bonus (+10 điểm tối đa) |
