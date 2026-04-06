# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Kết quả thí nghiệm thực tế:**

| Temperature | Latency | Chủ đề phản hồi |
|---|---|---|
| 0.0 | 3.28s | Hang Sơn Đoòng — hang động lớn nhất thế giới, phát hiện 1991, khám phá 2009, có hệ sinh thái riêng |
| 0.5 | 2.62s | Hang Sơn Đoòng — gần giống t=0.0 nhưng thêm chi tiết (dài 9km, có hệ thời tiết riêng) |
| 1.0 | 1.93s | Vịnh Hạ Long — chuyển sang chủ đề khác hoàn toàn, UNESCO, các đảo đá vôi |
| 1.5 | 2.05s | Cà phê Việt Nam — xếp thứ 2 thế giới, cà phê sữa đá, egg coffee |

**Kiểm tra tính nhất quán ở temperature=0.0 (3 lần chạy):** Cả 3 lần đều bắt đầu giống nhau: *"An interesting fact about Vietnam is that it is home to the world's largest cave, Son Doong Cave. Located in Phong Nha-Ke Bang National Park in Quang..."* — xác nhận tính deterministic.

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0 và 0.5, phản hồi rất nhất quán — cả hai đều chọn cùng chủ đề (Hang Sơn Đoòng) với nội dung gần giống nhau. Tôi đã chạy 3 lần ở t=0.0 và nhận được kết quả bắt đầu hoàn toàn giống nhau, xác nhận tính deterministic. Khi tăng lên 1.0, model chuyển hoàn toàn sang chủ đề khác (Vịnh Hạ Long thay vì Hang Sơn Đoòng), và ở 1.5 lại chuyển sang chủ đề cà phê Việt Nam — cho thấy temperature cao khiến model "mạo hiểm" hơn trong việc chọn token, dẫn đến câu trả lời đa dạng và khó đoán hơn.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature từ 0.0 đến 0.3. Dựa trên kết quả thí nghiệm, ở t=0.0 câu trả lời hoàn toàn nhất quán giữa các lần gọi — điều này rất quan trọng cho chatbot hỗ trợ khách hàng, vì cùng một câu hỏi phải cho cùng một câu trả lời. Giá trị 0.2–0.3 giúp câu trả lời tự nhiên hơn một chút mà vẫn giữ được độ chính xác, tránh trường hợp ở t=1.0+ model có thể "sáng tạo" quá mức và đưa ra thông tin không liên quan.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Kết quả so sánh thực tế từ `compare_models`:**

| Metric | GPT-4o | GPT-4o-mini |
|---|---|---|
| Latency | 1.31s | 1.36s |
| Cost estimate (1 call) | $0.000520 | — |
| Response quality | Chi tiết, cấu trúc tốt | Tương đương, hơi ngắn hơn |

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> **Tính toán chi tiết:**
> - Tổng output tokens/ngày: 10,000 users × 3 calls × 350 tokens = **10,500,000 tokens** = 10,500 × 1K tokens
> - Chi phí GPT-4o: 10,500 × $0.010 = **$105.00/ngày** (~$3,150/tháng)
> - Chi phí GPT-4o-mini: 10,500 × $0.0006 = **$6.30/ngày** (~$189/tháng)
> - Tỷ lệ: $0.010 / $0.0006 = **~16.67 lần**
>
> GPT-4o đắt hơn GPT-4o-mini khoảng **16.67 lần**. Với workload này, chênh lệch là $98.70/ngày hay ~$2,961/tháng.
>
> Tham chiếu thực tế: Trong kết quả chạy `compare_models` ở trên, GPT-4o tốn $0.000520 cho một câu trả lời ~39 words (~52 tokens). Ngoại suy ra: 10,500,000 tokens / 52 tokens × $0.000520 ≈ $105, khớp với tính toán lý thuyết.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> **GPT-4o xứng đáng:** Hệ thống tư vấn pháp lý hoặc y tế, nơi độ chính xác và khả năng suy luận phức tạp là tối quan trọng. Từ kết quả chạy thực tế, GPT-4o cho câu trả lời cấu trúc tốt hơn và đầy đủ hơn — với latency tương đương (~1.31s vs ~1.36s), chênh lệch tốc độ không đáng kể, nên chi phí cao hơn là đáng đầu tư cho chất lượng output.
>
> **GPT-4o-mini là lựa chọn tốt hơn:** Chatbot trả lời FAQ đơn giản (giờ mở cửa, chính sách đổi trả, tra cứu đơn hàng). Trong kết quả thực tế, GPT-4o-mini cho câu trả lời chất lượng tương đương về nội dung với chi phí chỉ bằng 1/16.67 — tiết kiệm ~$2,961/tháng cho workload 10K users/ngày mà hầu như không ảnh hưởng chất lượng.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng chatbot tương tác trực tiếp với người dùng, đặc biệt khi phản hồi dài. Từ kết quả thí nghiệm thực tế, latency trung bình là 1.3–3.3 giây — nếu dùng non-streaming, người dùng phải chờ toàn bộ thời gian này mà không thấy gì, tạo cảm giác "đứng hình". Với streaming, token đầu tiên xuất hiện gần như ngay lập tức, cho phép người dùng bắt đầu đọc trong khi model vẫn đang sinh text — và có thể dừng sớm nếu câu trả lời không đúng hướng, tiết kiệm thời gian và token. Streaming đặc biệt hiệu quả cho các phản hồi dài như giải thích khái niệm hay viết bài. Ngược lại, non-streaming phù hợp hơn trong các tình huống backend/batch processing — ví dụ pipeline phân loại email tự động, trích xuất thông tin từ tài liệu hàng loạt, hay đánh giá nội dung. Trong các trường hợp này, không có người dùng đang chờ, việc nhận kết quả một lần giúp đơn giản hóa logic xử lý (không cần buffer chunks), dễ retry khi lỗi, và dễ log/monitor hơn.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
