# Kiểm tra mức sẵn sàng — Lab 22 Core

Học viên: Nguyễn Nhân Sâm — 2A202602672 — lớp 3A, khóa IV. Cập nhật 08/10/2026.

| Tiêu chí rubric | Điểm | Bằng chứng / trạng thái |
|---|---:|---|
| NB0: my_dpo_loss và asserts | 6 | Có code, tests và notebook CPU 9/9 code cells đã chạy. |
| NB0: giải thích displacement | 4 | Có giải thích và ví dụ trong notebook NB0. |
| NB1: SFT loss giảm, merged reference | 8 | Loss 1.8841 → 1.2836; config DPO chỉ đúng reference Colab; log verify gốc không báo thiếu merged model. Trọng số không có trong ZIP; cần giữ bản runtime/Drive. |
| NB2: split theo prompt, 3 cặp mẫu | 8 | 800/100 prompt distinct, overlap 0, fingerprint khớp; phân tích 3 cặp tại REFLECTION §1. |
| NB2: length bias | 4 | Stats 65.875% chosen dài hơn và PNG có đủ. |
| NB3: reference SFT | 6 | Adapter config `/content/lab22/models/sft-merged`, không đổi sang đường Windows. |
| NB3: bốn đường reward train/held-out | 10 | PNG + reward_history.json; chosen/rejected được tách riêng. |
| NB3: giải thích diagnosis | 8 | REFLECTION §3 phân tích cả hai reward tăng và giới hạn nhãn INTENDED. |
| NB4: 8 cố định + ≥50 held-out | 6 | 58 rows, 4 helpfulness + 4 safety + 50 held-out; held-out thuộc eval. |
| NB4: judge, CI, sanity, length checks | 10 | API theo lựa chọn người học/labcoach; sanity 12/12, WR/CI tính lại khớp; API consistency 57/58. Không có panel/chấm chéo. |
| Phản tư §3/§4/§6 | 20 | Điền số đo thật, ví dụ h4/s4, nêu hạn chế và chi phí chưa đo. |
| Tái lập từ môi trường sạch | 5 | Có Core notebook + manifest 40 nguồn khớp lần chạy và bản executed: 46/46 code cells đã chạy, không error output; code khớp snapshot gốc ngoài cell finalization thêm. Chưa chạy lại môi trường sạch độc lập lần thứ hai. |
| Full verify exit 0 | 5 | **Đã nhận log Colab exit 0** trong ZIP cuối; hash báo cáo/ảnh khớp finalization, dữ liệu huấn luyện và chấm không đổi. Windows vẫn khác runtime và không có merged weights. |

Các mục cần hoàn tất trước khi nộp:

1. Đã chạy helper Colab và nhận `Lab22_Core_Evidence_Final.zip`, full verify exit 0; không cần chạy helper lại.
2. Đã nhận notebook có output và ZIP cuối; bản executed được lưu riêng, không ghi đè notebook nguồn. Không bấm Run all chỉ để cập nhật báo cáo.
3. Đã kiểm tra notebook không có error output và không thấy mẫu token API/HF/GitHub/private key thông thường; source khớp snapshot, giữ bản executed riêng. Đây là kiểm tra theo mẫu, không phải cam kết phát hiện mọi loại secret.
4. Commit/push bằng chứng và notebook lên repo public theo quyền người học. Hiện mới đổi tên repo và origin; chưa push thay đổi bài lab.
5. Nộp URL GitHub public vào LMS; chưa xác nhận nộp.

Rủi ro chất lượng cần giữ trong giải thích: held-out WR 50%, CI chứa 0.5; 36/58 outputs giống nhau; tất cả outputs còn tool markers; cả chosen/rejected tăng. Rubric chấm bằng chứng và giải thích, không yêu cầu DPO phải có điểm cao.

Repo public đã xác minh đổi tên: https://github.com/Nguyen-Sam-sheep-zzz/K4-L3-Track3-Day22-NguyenNhanSam-2A202602672 . Folder checkout trên Windows vẫn giữ tên cũ để không làm gián đoạn workspace hiện tại.
