# Kiểm tra cuối — Lab 22 Core

Học viên: Nguyễn Nhân Sâm — 2A202602672 — lớp 3A, khóa IV.

Bộ bằng chứng bắt buộc đã có ở workspace: nguồn NB0–NB4, notebook executed, metrics/config/split, parquet, raw outputs và kết quả API judge, bốn ảnh, REFLECTION, log full verify exit 0 từ runtime Colab gốc.

- Notebook lưu riêng tại `colab/Lab22_DPO_T4_executed.ipynb`; 86 cells, 46/46 code cells đã chạy, 44 cells có output, không error output. SHA256: `bda9f1e5a387c9d2ea05b6b6642152baf394744eeef79c0d0ea87ef396586384`.
- Code sequence khớp notebook Core snapshot ngoài cell thêm để áp dụng báo cáo. Cell finalization có execution_count 46, dù vị trí hiển thị gần đầu notebook, nên được thực thi sau 45 cell gốc.
- Log verify exit 1 do template ban đầu vẫn còn trong notebook để bảo toàn lịch sử. Kết quả mới nhất ở cell finalization và `submission/verify_colab.txt` là exit 0. Không xóa log cũ để giả rằng lần đầu đã pass.
- ZIP cuối có 27 files, SHA256 `70d01ddd34226fa128dd1662f9aeb3ecc107276a4b8c7732931af29add731a57`; hash REFLECTION/ảnh khớp finalization; metrics/split/judge/outputs không đổi so với ZIP gốc đã audit.
- Không thấy mẫu token OpenAI/HF/GitHub/private key thông thường trong notebook; đây là kiểm tra theo mẫu. Notebook lấy khóa từ Colab Secrets, không ghi key vào source.
- Lần kiểm tra tests gần nhất: 70 passed, không skip. Checker gốc và tests chuẩn không thay đổi. Full verifier Windows vẫn exit 1 do không có merged model và absolute reference Colab khác Windows; không sửa adapter config để tạo pass giả.
- REFLECTION là snapshot đã viết trước khi nhận notebook/ZIP cuối. Các câu về chưa nhận bằng chứng trong phần trạng thái cuối của báo cáo là lịch sử tại lúc viết; trạng thái hiện tại được cập nhật ở đây, `READINESS.md`, `LOCAL_PROGRESS.md`, `local_status.json` và các audit tương ứng. Giữ báo cáo khớp hash của bản đã verify tại Colab.

Kết quả chính: SFT loss giảm; DPO held-out reward accuracy 70%, margin +0.083938. Generated-answer held-out win rate 50%, CI95 42–57%; chưa phát hiện cải thiện rõ rệt. Cả chosen/rejected tăng, 36/58 answers giống nhau và tất cả outputs còn tool markers; báo cáo đã phân tích các hạn chế này. Rubric đánh giá bằng chứng và giải thích, không yêu cầu điểm mô hình tuyệt đối cao.

Repo public đã đổi tên và origin cập nhật: https://github.com/Nguyen-Sam-sheep-zzz/K4-L3-Track3-Day22-NguyenNhanSam-2A202602672 . Chưa commit/push thay đổi bài lab hoặc nộp LMS; chưa xác minh bộ bằng chứng này đã xuất hiện trên GitHub. Không có bonus hoặc lần tái chạy độc lập môi trường sạch thứ hai.
