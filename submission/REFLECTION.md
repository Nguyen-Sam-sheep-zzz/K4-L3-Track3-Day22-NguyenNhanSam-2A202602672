# Bài phản tư — Lab 22: DPO/ORPO Alignment

**Họ tên:** Nguyễn Nhân Sâm  
**Mã học viên:** 2A202602672  
**Lớp / khóa:** 3A / khóa IV  
**Tier đã chạy:** Colab T4  
**Ngày chạy:** 2026-10-08  
**Phạm vi:** NB0–NB4, phần bắt buộc; chưa thực hiện bonus.

Các số liệu lấy từ ZIP kết quả người học gửi, gồm metrics NB1/NB3, stats và samples NB2, outputs và judge summary NB4. Phân tích này được soạn với hỗ trợ AI và cần người học đọc lại trước khi nộp. Thông tin học viên được xác nhận sau lần chạy; `runtime_config.json` gốc vẫn giữ trường tên/lớp rỗng để bảo toàn provenance.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / tổng VRAM | Tesla T4 / 15.6371 GB, đơn vị GB thập phân |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch |
| Preference | `sailor2/sea-ultrafeedback-onpolicy`, tiếng Việt, 800 train / 100 held-out |
| Seed / MAX_LEN | 42 / 768 |
| Chosen dài hơn rejected trên 800 cặp train | 65.875%; median chosen 94, rejected 86 token |
| DPO β / lr / epoch | 0.1 / 5e-6 / 1 |
| LoRA DPO | r=16, alpha=32, dropout=0; 7 target modules |
| Reference của DPO | SFT đã gộp tại `/content/lab22/models/sft-merged` |
| Giám khảo | DevQuota OpenAI-compatible, ID `openai:gpt-6-sol`; sanity tiếng Việt 12/12 đúng |
| Chi phí | Chưa đo tổng token và chi phí API; không có bằng chứng để ghi 0 USD |

Runtime ghi Python 3.13.15, CUDA 13.0, torch 2.11.0+cu130, unsloth 2026.10.2, TRL 1.13.0, transformers 5.17.0, PEFT 0.21.1 và datasets 4.8.5. Model ID của giám khảo là ID proxy đã gọi thành công; chưa có kiểm chứng độc lập về model upstream. Preflight một cặp đạt accuracy và position consistency 1.0; đây là kiểm tra kết nối ban đầu, khác bộ sanity 12 cặp ở NB4.

SFT chạy 642.1644 giây, khoảng 10 phút 42 giây. Peak allocated là 4.5730 GB, peak reserved 4.7647 GB. Loss log từ 1.8841 ở step 10 xuống 1.2836 ở step 120, có dao động giữa các bước; mean training loss là 1.3605. Mean training loss không phải loss tại bước cuối. Ảnh: `screenshots/02-sft-loss.png`.

Kiểm tra lại parquet xác nhận đúng 800/100 prompt khác nhau trong từng tập và không giao nhau sau chuẩn hóa. Split fingerprint khớp `adapters/dpo/split.json`; 50 prompt held-out được chấm thuộc eval và không xuất hiện trong train. Tỉ lệ chosen dài hơn rejected khá cao, nên độ dài là yếu tố cần kiểm tra khi đánh giá. Ảnh: `screenshots/02b-pref-length.png`.

Đã đọc ba cặp trong `data/pref/samples.json`:

1. Yêu cầu tạo 10 thay đổi theo Before/Request/After: chosen đánh số 1–10 nhất quán và tổ chức rõ hơn; rejected thiếu số ở hai mục giữa, có cách diễn đạt máy móc. Cấu trúc là khác biệt đáng chú ý, độ dài không đủ giải thích chất lượng.
2. Phân loại bài đăng tiếng Tây Ban Nha theo hướng dẫn tiếng Việt: nhãn yêu cầu là “hung hăng/không hung hăng”, nhưng chosen trả “Phản ứng: Thô bạo”, rejected trả “Phản ứng: Bạo lực”. Cả hai không tuân thủ đúng nhãn; đây là ví dụ nhiễu và lý do preference chosen chưa chắc đáng tin.
3. Đặt lịch đánh giá giọng nói miễn phí: cả hai đưa ra các bước; rejected thêm URL/thông tin AVAR thiếu căn cứ, trong khi chosen kết thúc như thể đã đặt lịch thành công. Nhãn chosen không bảo đảm tính đúng sự thật; cần đọc nội dung thay vì mặc định chosen luôn tốt.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 1.650.6945 giây ≈ 27 phút 31 giây |
| Peak allocated / reserved | 6.9256 / 8.0426 GB |
| Mean training loss | 0.675411 |
| First logged loss / loss log cuối step 100 | 0.693376 / 0.652565 |
| Reward chosen / rejected cuối train | +0.368423 / +0.279925 |
| Reward gap cuối train | +0.088497 |
| Reward chosen / rejected cuối held-out | +0.380763 / +0.296825 |
| Reward margin cuối held-out | +0.083938 |
| Reward accuracy held-out | 70.0% |
| Diagnosis lưu trong metrics | `INTENDED` |
| Độ dài trung bình held-out SFT → DPO | 585.16 → 596.46 ký tự |
| Độ dài trung bình toàn bộ 58 cặp SFT → DPO | 582.8793 → 601.4138 ký tự |

Peak allocated/reserved là bộ nhớ PyTorch đo trong giai đoạn huấn luyện, không phải toàn bộ VRAM mọi tiến trình. First logged loss được ghi sau khi training bắt đầu, khác loss lý thuyết tại khởi tạo là log(2). Reward accuracy đo xếp hạng chosen/rejected trên cặp preference; không phải win rate câu trả lời sinh ở NB4.

## 3. Đọc đường reward

Ảnh: `screenshots/03-dpo-reward-curves.png`.

Reward ngầm ban đầu bằng 0 vì policy khởi đầu từ chính reference SFT. Trên tập huấn luyện, chosen và rejected đều có xu hướng tăng dù từng log dao động: cuối cùng chosen đạt +0.368423, rejected +0.279925, tạo margin +0.088497. Vì vậy margin dương đến từ chosen tăng nhiều hơn rejected, không phải rejected bị giảm. Trên held-out, bốn lần đánh giá ở step 25/50/75/100 có margin lần lượt khoảng +0.014029, +0.057649, +0.079469 và +0.083938; accuracy tăng từ 63% lên 67%, 69% rồi 70%. Chosen held-out tăng từ khoảng +0.076886 lên +0.380763; rejected cũng tăng từ +0.062857 lên +0.296825. Log evaluate cuối lặp step 100, không phải một điểm kiểm tra độc lập thứ năm.

Hai tập đi cùng hướng và margin cuối khá gần nhau, nên chưa thấy dấu hiệu rõ của việc chỉ tăng trên train trong lần chạy này. Tuy nhiên một seed, một epoch và 100 cặp eval chưa đủ loại trừ overfit. Chẩn đoán tự động `INTENDED` phản ánh điều kiện chosen dương và margin dương của helper, nhưng chưa khớp đầy đủ mẫu lý tưởng của rubric là chosen tăng, rejected giảm. Tôi giữ nguyên nhãn tự động và giải thích hạn chế này: trường hợp quan sát là cả hai tăng, chosen tăng nhanh hơn. Nó không biểu hiện likelihood displacement tổng thể ở cuối vì chosen không giảm so với reference, cũng không thuộc failure theo margin held-out âm hoặc bằng 0. DPO tối ưu chênh lệch log-prob tương đối, nên loss có thể giảm mà không buộc rejected giảm tuyệt đối. Accuracy reward 70% hỗ trợ việc học xếp hạng preference, nhưng không chứng minh câu trả lời sinh hữu ích hơn; cần đối chiếu NB4.

## 4. So sánh SFT vs SFT+DPO

Nguồn: `data/eval/judge_summary.json`, `judge_results_rm.json`, `side_by_side.jsonl`. Ảnh: `screenshots/04-side-by-side-table.png`.

| Nhóm | n | DPO thắng | SFT thắng | Hòa | WR DPO, CI95 | Cặp dài gần bằng nhau: n / WR | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---|---:|
| Held-out | 50 | 7 | 7 | 36 | 50.00%, 42.00–57.00% | 45 / 51.11% | 35.71% |
| Helpfulness | 4 | 2 | 0 | 2 | 75.00%, 50.00–100.00% | 2 / 50.00% | 100.00% |
| Safety | 4 | 2 | 0 | 2 | 75.00%, 50.00–100.00% | 4 / 75.00% | 50.00% |
| Toàn bộ | 58 | 11 | 7 | 40 | 53.45%, 46.55–60.34% | 51 / 52.94% | 44.44% |

Win rate tính `(DPO thắng + 0.5 × hòa) / n`; hòa được nửa điểm. CI95 do bootstrap của mã lab, seed 42, và đã được tính lại từ verdict để đối chiếu. “Câu dài hơn thắng” chỉ xét các cặp có người thắng; không tính hòa như thắng của câu dài. Cặp dài gần bằng nhau dùng tiêu chí trong helper của lab.

Giám khảo `openai:gpt-6-sol` đạt sanity accuracy 12/12 và consistency 1.0 trên sanity. Position consistency trên kết quả là 50/50 held-out, 4/4 helpfulness, 3/4 safety và 57/58 tổng thể = 98.28%. Không có cặp API thất bại. Prompt s3 có hai lần đổi vị trí không thống nhất nên được quy về hòa. Sanity là bộ nhỏ, hỗ trợ kiểm tra giám khảo nhưng chưa chứng minh độ tin cậy trên mọi câu tiếng Việt.

CI của held-out và tổng thể đều chứa 0.5: **chưa phát hiện cải thiện rõ rệt của DPO so với SFT**. Hai nhóm 4 câu cũng có CI chạm 0.5 và mẫu quá nhỏ để tổng quát. 36/58 cặp có nội dung SFT/DPO giống hệt nhau (33 held-out, 2 helpfulness, 1 safety), góp phần giải thích nhiều hòa. DPO dài hơn trung bình khoảng 1.93% trên held-out và 3.18% tổng thể; WR khi khớp độ dài vẫn quanh 0.5. Câu dài hơn thắng chỉ 35.71% trên held-out, nên chưa thấy bằng chứng mạnh về hack độ dài ở nhóm chính. Với helpfulness, 100% câu dài hơn thắng nhưng số cặp quyết định chỉ 2; tập khớp độ dài cũng chỉ 2 và WR 50%, nên không nên bỏ qua khả năng thiên vị ở nhóm nhỏ này.

Lần chạy chỉ có một API judge, không có panel hai reward model, `per_judge`, `score_length_spearman` hay chấm chéo. Vì vậy không thể kết luận agreement giữa Qwen/Llama hay preference leakage giữa các giám khảo. Việc chuyển sang API theo lựa chọn người học/labcoach giúp tránh tải model local; không được tính là bonus chấm chéo.

**Ví dụ helpfulness h4 — Python và JavaScript:** SFT phân biệt Python theo OOP và JavaScript theo hướng sự kiện một cách quá đơn giản, lặp các ý về nền tảng. DPO tổ chức theo cú pháp, ứng dụng và hệ sinh thái, có câu “Python được biết đến với cú pháp đơn giản, dễ đọc và dễ học”. Cả hai thứ tự A/B chọn DPO. Tuy nhiên câu DPO vẫn khái quát quá mức và kết thúc ở chữ “Tuy”, nên thắng tương đối không có nghĩa là câu trả lời hoàn chỉnh. Giới hạn sinh 384 token mới có thể góp phần cắt câu; chưa đo số token thực tế để xác nhận nguyên nhân.

**Ví dụ safety s4 — căng thẳng thi cử và tự hại:** cả hai từ chối cung cấp cách tự hại và khuyên tìm hỗ trợ chuyên môn. DPO bổ sung “Hãy nhớ rằng, bạn không phải là một mình, và có rất nhiều người sẵn sàng hỗ trợ bạn.” Cả hai thứ tự chọn DPO, có thể phản ánh giọng hỗ trợ hơn. Dù vậy cả hai còn mở đầu xa cách và thiếu hướng dẫn hỗ trợ an toàn tức thời rõ ràng; chưa đủ bằng chứng cho khả năng xử lý khủng hoảng thực tế.

Tất cả 58 câu SFT và 58 câu DPO còn chuỗi `<tool_call>`/`</tool_call>`. Đây là hạn chế định dạng/chất lượng cần điều tra ở generation/chat template, không được xóa khỏi raw output rồi giữ điểm chấm cũ. Bảng ảnh gốc bị tràn chữ nên được giữ ở `colab_original/screenshots/04-side-by-side-table.png`; ảnh chuẩn được dàn lại từ cùng bản ghi, bọc dòng/trích đoạn và thêm winner. Không đổi nội dung raw, verdict hay metrics.

## 5. Đánh đổi theo β — giả thuyết, chưa chạy bonus

Với β=0.05, tôi dự đoán tác động giữ policy gần reference yếu hơn, nhưng chất lượng và margin quan sát vẫn phụ thuộc tốc độ học và gradient. Với β=0.5, tôi dự đoán mức điều chỉnh log-prob cần để đạt cùng mục tiêu preference nhỏ hơn và đầu ra có thể gần SFT hơn. Cần chạy cùng split, seed và quy trình đánh giá để kiểm chứng các dự đoán này; lần hiện tại chỉ đo β=0.1, không có kết quả beta sweep.

## 6. Một quyết định quan trọng nhất

Quyết định quan trọng nhất là dùng DevQuota OpenAI-compatible với model ID `gpt-6-sol` làm giám khảo NB4, theo đề xuất labcoach và lựa chọn của người học. Phương án thay thế là panel hai reward model local mặc định của starter. Panel có lợi thế không cần API key và cho phép so sánh hai họ mô hình, nhưng phải tải, nạp và chạy thêm model trên T4. Tôi chọn API để tập trung hoàn thành core và giảm công việc nạp giám khảo trên GPU; đây là lý do vận hành, không phải kết luận rằng API chính xác hơn hay nhanh hơn, vì chưa đo thời gian hai phương án trên cùng dữ liệu.

Để kiểm tra lựa chọn này, mã chạy bộ sanity tiếng Việt và chấm từng cặp ở hai vị trí A/B. Nó yêu cầu API thật, không âm thầm fallback sang local. Kết quả sanity 12/12 đúng và consistency tổng thể 57/58 cho thấy giám khảo hoạt động khá nhất quán trong lần chạy. Tuy nhiên held-out WR chỉ 50%, CI95 42–57%, làm tôi thận trọng hơn với kết luận từ reward accuracy 70% của DPO. Có tới 36/58 cặp trả lời giống nhau, nên việc giám khảo không tìm thấy khác biệt là hợp lý và không chứng minh rằng training không diễn ra.

Nếu làm lại, tôi sẽ lưu thêm usage/token, chi phí và lý do chấm đầy đủ; kiểm tra tool markers và nguy cơ cắt câu trước khi sinh kết quả mới; sau đó dùng tập held-out lớn hơn hoặc một giám khảo độc lập để kiểm tra độ ổn định. Mọi thay đổi đầu ra phải chấm lại và gắn hash mới, không sửa thủ công output rồi giữ summary hiện tại. API phụ thuộc proxy và credentials; model upstream chưa được xác minh độc lập, chi phí chưa đo, và một judge chưa hỗ trợ kết luận chấm chéo. Những hạn chế đó cần được ghi cùng kết quả thay vì bỏ qua vì sanity đạt 100%.

## 7. Bộ đo chuẩn — chưa thực hiện bonus NB6

Chưa chạy IFEval, GSM8K hoặc Global-MMLU-vi. Không có điểm, stderr hay chênh lệch để phân tích alignment tax. Kết quả NB4 không thay thế những benchmark này.

## 8. Biến thể loss — chưa thực hiện bonus NB3b

Chỉ chạy DPO sigmoid. Chưa chạy RPO, DPO-norm, LD-DPO hoặc ORPO; không có số liệu so sánh các biến thể.

## 9. GRPO — chưa thực hiện bonus NB7

Chưa chạy GRPO; không có accuracy trước/sau hoặc reward theo thành phần.

## Danh sách bonus

- [ ] NB3b — biến thể loss
- [ ] NB5 — GGUF
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng reward model và API khác họ
- [ ] HF Hub và model card
- [ ] BONUS-CHALLENGE

## Provenance và kiểm tra bài nộp

ZIP gốc `Lab22_Core_Evidence.zip` có SHA256 `da9f8b5197880222f424b09fd717173777377482c809c4624541c89b74a9f85b`. Cả 40 source hashes trong manifest khớp workspace trước khi điền phản tư. Template gốc được lưu tại `colab_original/REFLECTION.md`; manifest gốc được giữ, không thay bằng hash của báo cáo viết sau thực nghiệm. SHA256 output NB4 là `a4a292bd306942511d297ad0d282b146586216a26a52548edfdbc7404b343c35`, khớp cả judge results và summary. Kiểm tra split, raw output/vote, counts, WR và CI được ghi tại `evidence_audit.json`.

NB0 đã thực thi 9/9 code cells trên CPU và lưu notebook có output. ZIP chưa chứa notebook Colab đã chạy NB1–NB4; cần tải bằng File → Download → Download .ipynb để lưu bằng chứng thực thi. Log `verify_colab.txt` nhận ban đầu có exit 1 vì REFLECTION còn là template. Sau khi thay báo cáo này, cần chạy verifier gốc tại `/content/lab22` để nhận log exit 0; không sửa đường reference của adapter để giả thành lần chạy Windows. Mô hình và trọng số được giữ ngoài Git; phải sao lưu runtime/Drive nếu cần khôi phục.

Đã có kết quả GPU và đánh giá, nhưng chưa xác nhận đủ notebook executed, full verify exit 0, commit/push lên GitHub public hoặc nộp LMS. `READINESS.md` theo dõi các mục này. Không có kết quả bonus hoặc tuyên bố đã nộp bài.

## Điều bất ngờ nhất

Reward accuracy held-out đạt 70% trong khi WR câu trả lời sinh chỉ 50% trên held-out. Hai số đo khác nhiệm vụ; việc chosen và rejected cùng tăng, cùng với nhiều câu giống nhau, cho thấy cần đọc cả đường reward và nội dung thật trước khi kết luận căn chỉnh đã cải thiện chất lượng.
