# Tiến độ Lab 22 — phần bắt buộc

Ngày cập nhật: 08/10/2026. Nhánh local: `feature/lab22-core`.

| Hạng mục | Trạng thái |
|---|---|
| Học viên | Nguyễn Nhân Sâm, 2A202602672, lớp 3A, khóa IV. |
| NB0 | Code và tests; notebook đã thực thi 9/9 code cells CPU, không error output. |
| NB1 | Có SFT metrics/PNG: 1.000 mẫu, 642.1644 giây; peak allocated 4.5730 GB. |
| NB2 | 800 train / 100 eval, không trùng prompt; stats, 3 cặp mẫu, PNG. |
| NB3 | Có config/split/reward history/metrics/PNG: 1.650.6945 giây, held-out reward accuracy 70%. |
| NB4 | API DevQuota `gpt-6-sol` đã chạy; 58 cặp, sanity 12/12, held-out WR 50%, CI95 42–57%. |
| Audit | Hash source/split/output và summary tính lại khớp; raw evidence giữ nguyên. |
| REFLECTION | Đã điền bằng số đo thật, phân tích hạn chế và ví dụ; bonus chưa thực hiện. |
| Notebook executed NB1–NB4 | Đã nhận, lưu `colab/Lab22_DPO_T4_executed.ipynb`: 46/46 code cells chạy, 44 cells có output, không error output; source snapshot khớp. |
| Full verify | Đã nhận ZIP cuối và log full verify exit 0 tại `/content/lab22`; hashes báo cáo/ảnh và dữ liệu gốc khớp. |
| GitHub | Repo đã đổi tên và origin cập nhật; public. Chưa commit/push thay đổi bài lab. |
| LMS | Chưa xác nhận nộp. |

Xem `READINESS.md` để đối chiếu từng hàng rubric. Không cần training lại để cập nhật báo cáo/verify nếu runtime Colab vẫn còn đầy đủ artifacts. Nếu runtime mất và chưa sao lưu weights, ZIP nhỏ không đủ khôi phục model.
