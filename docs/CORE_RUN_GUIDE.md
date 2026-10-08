# Chạy phần bắt buộc Lab 22 trên Colab T4

## File sử dụng

Mở `colab/Lab22_DPO_T4_Core.ipynb` trên Google Colab. Đây là bản chỉ chạy NB0–NB4, có mã nguồn và bộ kiểm tra bên trong, không cần clone repo. Hai notebook T4/BigGPU cũ vẫn là bản có toàn bộ bonus.

## Cấu hình DevQuota

1. Trong Colab, chọn biểu tượng chìa khoá **Secrets** ở thanh bên trái.
2. Thêm secret **`OPENAI_API_KEY`** bằng key DevQuota hợp lệ; bật **Notebook access**.
3. Endpoint mặc định: `https://sv.devquote.shop/v1`. Nếu tài khoản dùng endpoint khác, thêm secret **`OPENAI_BASE_URL`** và bật quyền truy cập.
4. Model đã đặt theo lựa chọn của bạn: **`gpt-6-sol`**. Nếu DevQuota dùng ID khác, chỉ sửa `JUDGE_MODEL` trong cell đầu.
5. Không dán key vào cell, output hoặc gửi key trong chat.

Key trên Windows trước đó trả HTTP 401; lần chạy Colab ngày 08/10/2026 đã gọi API thành công với `gpt-6-sol`, sanity 12/12. Notebook kiểm tra API trước training. Hai lượt preflight cần trả JSON chọn đúng câu trả lời tiếng Việt ở cả hai vị trí, nếu không thì dừng để sửa key/model. Không tự chuyển sang reward model local.

## Cách chạy

1. **Runtime → Change runtime type → T4 GPU**.
2. Upload notebook Core qua **File → Upload notebook** nếu chưa mở.
3. Điền `STUDENT_NAME` và `STUDENT_CLASS` trong cell đầu nếu muốn lưu thông tin này vào bằng chứng.
4. **Runtime → Run all**. Chờ cài dependencies, kiểm tra API/CUDA và tests. Notebook không chạy bonus.
5. Chạy lần lượt NB0, NB1, NB2, NB3, NB4; xem output để phát hiện lỗi. NB3 có thể mất 40–60 phút theo đề, chưa gồm tải/cài.
6. Cell cuối kiểm tra đủ 4 ảnh, 8 câu cố định, ≥50 held-out được chấm và SHA của file output; tạo/tải **`Lab22_Core_Evidence.zip`**.
7. Tải thêm notebook có output: **File → Download → Download .ipynb**. Lưu thành **`Lab22_DPO_T4_executed.ipynb`** để tránh bị công cụ tạo notebook ghi đè.
8. Gửi cả ZIP và notebook đã chạy để hoàn thiện REFLECTION và kiểm tra bài nộp. Giữ runtime mở để chạy verify cuối sau khi cập nhật REFLECTION.

## Thời gian, chi phí và kết quả

- GPU: theo đề, phần bắt buộc khoảng 1.5–2 giờ, chưa gồm tải/cài/lỗi runtime; đây là ước tính, chưa đo ở phiên của bạn.
- NB4 không tải hai reward model local. API dùng khoảng 142 lượt gọi bình thường với 58 câu: 2 preflight + 24 sanity + 116 chấm hai vị trí; retry có thể làm tăng số lượt. Chi phí thực tế phụ thuộc DevQuota, không mặc định là miễn phí.
- `judge_summary.json` lưu win rate, CI95, sanity tiếng Việt, position consistency và chỉ số độ dài. Với API, `per_judge` của panel local không áp dụng; kết quả ghi rõ `openai:gpt-6-sol`.
- SFT/DPO metrics lưu thời gian huấn luyện và peak **allocated/reserved** của PyTorch; các giá trị này không phải toàn bộ VRAM mọi tiến trình.
- `data/pref/samples.json` giữ 3 cặp mẫu để phân tích, `stats.json` giữ số lượng/tỉ lệ độ dài; `reward_history.json` giữ log train/held-out.

## REFLECTION và verify

REFLECTION vẫn là template cho đến khi có kết quả thật. Vì vậy `verify.py` ở cuối có thể báo chưa điền §1/2/3/4/6. Notebook lưu nguyên stdout và exit code trong `submission/verify_colab.txt` và `verify_status.json`; nó không đổi tiêu chí để tạo pass giả.

Sau khi REFLECTION đã hoàn thiện, ghi nó vào `/content/lab22/submission/REFLECTION.md`, rồi chạy:

```python
!python /content/lab22/scripts/verify.py
```

Kỳ vọng exit code 0. Chạy lại cell export để ZIP chứa báo cáo và log mới. Verify phải chạy trong Colab gốc vì adapter config chứa đường dẫn `/content/lab22/models/sft-merged`; chuyển về Windows có thể báo WRONG REF dù provenance Colab vẫn đúng. Không sửa đường dẫn để giả thành huấn luyện trên Windows.

## Sao lưu và xử lý lỗi

- Sau NB1/2/3/4 có ZIP checkpoint evidence trong `/content/lab22`. Có thể tải thủ công từ bảng Files nếu phải rời phiên.
- ZIP không chứa trọng số/tokenizer và không đủ để tiếp tục training sau khi mất phiên. Muốn phục hồi, sao lưu riêng model SFT đã merge và adapter/tokenizer sang Drive trước khi hết runtime; nếu không, chạy lại NB1.
- OOM: đặt `MAX_LEN=512` trong cell đầu, restart runtime và chạy lại các bước phụ thuộc. Không trộn adapter cũ với split mới.
- HTTP 401: kiểm tra key DevQuota và Notebook access. HTTP 400/model not found: kiểm tra ID model hoặc các tham số được endpoint hỗ trợ, gửi nguyên thông báo lỗi đã loại secrets để chẩn đoán.
- API trả nội dung rỗng/không đúng JSON: preflight sẽ dừng; không bỏ assert để tiếp tục.
- Sau khi chạy xong, đủ bằng chứng local/Colab chưa đồng nghĩa đã nộp. Cần commit/push lên GitHub public và nộp URL vào LMS khi bài hoàn chỉnh.

## Trạng thái chuẩn bị

NB0 đã thực thi trên CPU và giữ output. NB1–NB4 đã chạy trên T4 và ZIP đã được audit; 70 tests Windows qua. SFT mất 10 phút 42 giây, DPO 27 phút 31 giây (chỉ thời gian training). Xem `submission/LOCAL_PROGRESS.md` và `submission/READINESS.md` để biết các mục còn thiếu.

## Hoàn tất lần chạy đã gửi ZIP

1. Giữ notebook và runtime gốc đang mở. Upload `submission/APPLY_REFLECTION_COLAB.py` từ máy vào bảng **Files** của Colab (tới `/content`).
2. Tạo một cell cuối và chạy:

```python
%run /content/APPLY_REFLECTION_COLAB.py
```

Helper cập nhật REFLECTION và bảng ảnh dễ đọc, giữ bản gốc, kiểm tra hash outputs/verifier rồi chạy checker nguyên bản. Chỉ khi exit 0 mới tải `Lab22_Core_Evidence_Final.zip`. Không cần chạy lại training nếu runtime còn đầy đủ artifacts.

3. Tải thêm **File → Download → Download .ipynb**, gửi lại notebook có output và ZIP cuối. Nếu helper báo lỗi, gửi log để xử lý; không bấm Run all chỉ để cập nhật báo cáo.

Notebook Core nguồn đang giữ archive nguồn của lần chạy đã audit, gồm REFLECTION template ban đầu. Báo cáo hoàn thiện nằm riêng ở `submission/REFLECTION.md`; không chạy builder để ghi đè snapshot chỉ vì báo cáo đã thay đổi. `build_core_colab.py --check` có thể báo khác payload do báo cáo sau thực nghiệm; các source hashes gốc vẫn được giữ để đối chiếu lần chạy.

Repo đã đổi tên thành `K4-L3-Track3-Day22-NguyenNhanSam-2A202602672`, origin đã cập nhật; folder Windows giữ tên cũ. Chưa commit/push thay đổi bài lab hoặc nộp LMS.
