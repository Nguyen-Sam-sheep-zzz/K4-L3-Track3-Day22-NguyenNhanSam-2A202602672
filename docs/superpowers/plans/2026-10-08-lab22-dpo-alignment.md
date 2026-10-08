# Lab 22 DPO Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execution defaults to inline work with checkpoints; delegation requires an explicit user choice.

**Goal:** Hoàn thành NB0–NB4 với bằng chứng huấn luyện/đánh giá thật, REFLECTION đầy đủ và repo GitHub public có thể chấm và tái lập.

**Architecture:** Chuẩn bị code và lưu bài trên Windows; huấn luyện trên Colab T4. SFT được merge thành mô hình tham chiếu cố định; DPO học adapter mới trên cùng mô hình SFT. Đánh giá cùng câu hỏi bằng SFT và SFT+DPO, kiểm tra độ tin cậy của giám khảo rồi phân tích số liệu.

**Tech Stack:** Python 3.11, PyTorch CUDA, Unsloth, Qwen3-4B-Instruct-2507 4-bit, PEFT/LoRA, TRL 1.13, datasets, matplotlib, Jupyter/Colab, reward-model panel Skywork.

**Spec:** `README.md`, `rubric.md`, `submission/REFLECTION.md`, `HARDWARE-GUIDE.md`, `docs/reference.md`; mã nguồn kiểm chứng: `lab22/config.py`, `scripts/verify.py`, `scripts/build_colab.py`, `notebooks/00_dpo_loss_from_scratch.py`.

## Global Constraints

- Phần bắt buộc: NB0, NB1, NB2, NB3, NB4; tối đa 100 điểm. Bonus tối đa +20.
- T4 mặc định: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`, `MAX_LEN=768`, `SEED=42`.
- `SFT_SLICE=1000`, `PREF_TRAIN=800`, `PREF_EVAL=100`, `JUDGE_PROMPTS=50`; NB4 có thêm 8 câu cố định (4 helpfulness + 4 safety).
- `DPO_BETA=0.1`, `DPO_LR=5e-6`, `DPO_EPOCHS=1`, `LORA_R=16`, `LORA_ALPHA=32`.
- `unsloth>=2026.10.1`, `trl>=1.13,<1.14`, `transformers>=5.2,<5.18`, `datasets>=4.7,<5.0`, `peft>=0.18,<1.0`; dùng giới hạn phiên bản của repo.
- Train/held-out phải chia theo prompt, không giao nhau. Giữ nguyên split từ NB2 đến NB4; không đổi seed/dataset rồi dùng adapter cũ.
- DPO phải xuất phát từ `models/sft-merged/`, không từ raw base model.
- Có ≥50 câu held-out được chấm; báo win rate + CI 95%, sanity accuracy, thiên vị độ dài và `per_judge`.
- §3 REFLECTION ≥100 từ; §6 ≥150 từ. Đây là mức chặt hơn trong README/template, đáp ứng cả rubric ghi tổng ≥150 từ.
- Notebook nộp phải giữ output. Không điền số liệu giả, không thay bộ kiểm tra để làm pass, không commit secrets/trọng số mô hình.
- Nộp link GitHub public vào LMS, không tạo PR; giữ public đến khi có điểm. README ghi hạn 23:59 ngày hôm sau, cần đối chiếu lịch lớp/LMS trước khi gán ngày cụ thể.

## Hiện trạng đã kiểm tra ngày 08/10/2026

- Repo đang ở `main`, chưa có thay đổi trước khi tạo tài liệu kế hoạch.
- `my_dpo_loss` trong NB0 còn `return None`; cell kiểm tra bỏ qua assert nếu chưa cài hàm. Vì vậy notebook chạy không lỗi chưa chứng minh hoàn thành NB0.
- `colab/Lab22_DPO_T4.ipynb` có 69 code cells, 0 cell đã thực thi, 0 cell có output.
- `data/` và `adapters/` chưa có bằng chứng chạy; REFLECTION còn mẫu chưa điền.
- Máy Windows có Python 3.11.7 và NVIDIA GTX 1650 4096 MiB, driver 512.74. GPU này không đủ cho cấu hình lab yêu cầu GPU ≥12 GB; dùng Colab T4 16 GB.
- Đây là kế hoạch; chưa huấn luyện, chưa chạy test suite, chưa xác minh GPU Colab hoặc kết nối tải Hugging Face.

## File map và trách nhiệm

| File/thư mục | Vai trò và thao tác khi thực hiện |
|---|---|
| `notebooks/00_dpo_loss_from_scratch.py` | Điền hàm học viên và assert trực tiếp trên hàm đó. |
| `scripts/test_student_dpo_loss.py` (tạo mới) | Kiểm tra độc lập hàm học viên: bằng reference, khớp công thức, gradient, displacement. |
| `colab/Lab22_DPO_T4.ipynb`, `colab/Lab22_DPO_BigGPU.ipynb` | Sinh lại từ nguồn khi NB0 thay đổi; giữ bản này chưa chạy để check đồng bộ. |
| `colab/Lab22_DPO_T4_executed.ipynb` (tạo từ phiên thật) | Bản nộp giữ toàn bộ output NB0–NB4, không bị builder ghi đè. |
| `lab22/config.py`, `.env.example` | Đọc cấu hình; giữ mặc định T4 cho lần chạy chính thức. |
| `notebooks/01_sft_mini.py` | Chạy SFT và merge; đo thời gian/VRAM nếu cần ghi trong báo cáo. |
| `notebooks/02_preference_data.py` | Tạo split theo prompt, biểu đồ và số liệu độ dài. |
| `notebooks/03_dpo_train.py` | Huấn luyện, reward curves, diagnosis; lưu thêm thời gian/VRAM đo thật nếu chưa có. |
| `notebooks/04_compare_and_eval.py` | Sinh 8+≥50 cặp trả lời, judge, summary và ví dụ. |
| `data/pref/` | `train.parquet`, `eval.parquet`, `stats.json`; giữ split gốc. |
| `adapters/sft-mini/`, `models/sft-merged/`, `adapters/dpo/` | Trọng số lưu ngoài Git; config, metrics, `split.json` làm evidence. |
| `data/eval/` | `side_by_side.jsonl`, `judge_results_rm.json`, `judge_summary.json`; API judge nếu làm bonus. |
| `submission/screenshots/` | 4 PNG bắt buộc. |
| `submission/REFLECTION.md` | Điền §1, §2, §3, §4, §6 bằng kết quả thật. |
| `.gitignore` | Khi thực hiện, cân nhắc thêm ngoại lệ `!data/pref/stats.json` để lưu số liệu NB2; không bỏ chặn trọng số/secrets. |
| `scripts/verify.py`, tests hiện có | Dùng nguyên bản để kiểm tra; không sửa tiêu chí nhằm đạt pass. |

## Task 1: Chuẩn bị môi trường và bộ bằng chứng

**Files:** đọc cấu hình/setup; tạo bản notebook executed trong phiên sau; ghi nhật ký cấu hình vào REFLECTION §1.

**Interfaces:** đầu vào là repo starter; đầu ra là Colab T4 CUDA hoạt động và nguồn notebook đồng bộ.

- [ ] Trên Windows chạy `py -3.11 --version`; môi trường CPU dùng kiểm tra code, không cài toàn bộ CUDA stack lên GTX 1650.
- [ ] Cài dependencies CPU vào môi trường riêng nếu còn thiếu: `py -3.11 -m venv .venv`; `.\.venv\Scripts\python.exe -m pip install pytest torch pandas matplotlib`.
- [ ] Chạy `python scripts/build_colab.py --check` bằng interpreter đang dùng; kỳ vọng exit 0 trước khi sửa và sau khi sinh lại.
- [ ] Mở notebook T4 trên Colab; chọn Runtime → Change runtime type → T4 GPU.
- [ ] Chạy cell setup và xác nhận CUDA/name/VRAM:

```python
import torch
assert torch.cuda.is_available(), "Chọn T4 GPU trước khi huấn luyện"
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9)
```

- [ ] Giữ mặc định T4, seed 42, beta 0.1, lr 5e-6. Nếu thiếu VRAM, đổi `MAX_LEN=512` trong cell đầu, khởi động lại cấu hình và chạy các bước phụ thuộc; ghi rõ thay đổi.
- [ ] Đảm bảo nguồn repo (đủ `notebooks/*.py`, `scripts/verify.py`, template REFLECTION) có trong `/content/lab22` trước bước verify. Bundle chỉ ghi package `lab22`, không ghi đủ các file nguồn/verify. Có thể upload ZIP nguồn rồi giải nén; không chứa `.env`, `.git`, `.venv`, weights.
- [ ] Nếu chỉ thử luồng với `PREF_TRAIN=200`, ghi là smoke riêng; lần nộp dùng mặc định 800/100 và ≥50 held-out, hoặc báo rõ cấu hình khác thực tế.
- [ ] Dự trù 1.5–2 giờ GPU theo README, thêm thời gian cài/tải/khắc phục lỗi và 1–2 giờ báo cáo. Đây là ước tính, chưa đo trên phiên của bạn.

**Checkpoint:** GPU T4 hoạt động, thông số được ghi lại, notebook nguồn và bản giữ output có tên tách biệt.

## Task 2: NB0 — hoàn thành loss và chứng minh hiểu công thức (10 điểm)

**Files:** sửa `notebooks/00_dpo_loss_from_scratch.py`; tạo `scripts/test_student_dpo_loss.py`; sinh lại hai bundle bằng builder.

**Interfaces:** `my_dpo_loss(pc, pr, rc, rr, beta=0.1) -> scalar Tensor`; tham số là tensors log-prob cùng shape. Hàm giữ autograd để học được.

- [ ] Viết test trên chính hàm học viên, không chỉ `lab22.dpo_math.dpo_loss`. Có thể dùng AST lấy định nghĩa hàm để tránh chạy toàn bộ notebook khi import:

```python
import ast
import math
from pathlib import Path
import torch
from lab22 import dpo_math as M

source = Path('notebooks/00_dpo_loss_from_scratch.py').read_text(encoding='utf-8')
tree = ast.parse(source)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'my_dpo_loss')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<student-loss>', 'exec'), namespace)
loss_fn = namespace['my_dpo_loss']

def test_student_init_is_log2():
    c, r = torch.tensor([-12., -30.]), torch.tensor([-15., -28.])
    assert torch.allclose(loss_fn(c, r, c, r), torch.tensor(math.log(2)), atol=1e-6)

def test_student_matches_reference_and_gradient():
    pc = torch.tensor([-12., -30.], requires_grad=True)
    pr, rc, rr = torch.tensor([-15., -28.]), torch.tensor([-13., -29.]), torch.tensor([-14., -29.])
    actual = loss_fn(pc, pr, rc, rr, beta=0.1)
    expected, _, _ = M.dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert torch.isfinite(pc.grad).all()
    assert (pc.grad < 0).all()

def test_student_displacement_has_same_loss():
    rc, rr = torch.tensor([-20.]), torch.tensor([-22.])
    intended = loss_fn(rc + 1, rr - 1, rc, rr, beta=1.)
    displaced = loss_fn(rc - 3, rr - 5, rc, rr, beta=1.)
    assert torch.allclose(intended, displaced, atol=1e-6)
```

- [ ] Chạy `python -m pytest scripts/test_student_dpo_loss.py -q` khi hàm còn None, xác nhận fail đúng nguyên nhân.
- [ ] Điền công thức:

```python
def my_dpo_loss(pc, pr, rc, rr, beta=0.1):
    margin = beta * ((pc - rc) - (pr - rr))
    return -torch.nn.functional.logsigmoid(margin).mean()
```

- [ ] Thêm assert `mine is not None` và kiểm tra loss khởi đầu trên `my_dpo_loss` trực tiếp trong notebook; không để nhánh None được coi là hoàn thành.
- [ ] Chạy tests mới và `python -m pytest scripts/ -q`; kiểm tra có skip do thiếu torch/pandas không. `python scripts/build_colab.py` phải chạy trước test đồng bộ bundle sau mỗi thay đổi nguồn.
- [ ] Chạy NB0, giữ output; giải thích ví dụ chosen -3, rejected -5: margin tăng 2 dù chosen giảm. DPO tối ưu chênh lệch tương đối.

**Checkpoint:** hàm học viên qua assert + gradient; output loss init khoảng 0.693; trả lời displacement bằng lời và ví dụ số.

## Task 3: NB1 — SFT và mô hình tham chiếu (8 điểm)

**Files:** chạy `notebooks/01_sft_mini.py`; đầu ra `adapters/sft-mini/`, `models/sft-merged/`, `submission/screenshots/02-sft-loss.png`.

**Interfaces:** NB3 và NB4 đọc cùng `models/sft-merged/`; NB1 phải hoàn tất trước hai bước đó.

- [ ] Chạy NB1 với 1.000 mẫu SFT tiếng Việt, LoRA r16, cấu hình T4; giữ log/output thật.
- [ ] Kiểm tra loss có xu hướng giảm, không bắt buộc mọi điểm đều giảm. Nếu bất thường, kiểm tra dữ liệu, masking và training trước khi sang NB3.
- [ ] Kiểm tra ảnh loss, SFT adapter config và merged `config.json` cùng trọng số tồn tại trong runtime.
- [ ] Ghi model/dataset/epoch/hyperparameters, thời gian đo được; sao lưu adapter và merged model vào Drive nếu muốn phục hồi giữa phiên, lưu ngoài Git.

**Checkpoint:** SFT chạy thật và merge thành công, ảnh loss còn đọc được, reference được lưu.

## Task 4: NB2 — dữ liệu sở thích và độ dài (12 điểm)

**Files:** chạy `notebooks/02_preference_data.py`; đầu ra hai parquet, `stats.json`, `02b-pref-length.png`.

**Interfaces:** NB3 dùng 800 train /100 held-out; NB4 lấy ≥50 prompt khác nhau từ đúng held-out đó. `split.json` của adapter sẽ hash hai parquet này.

- [ ] Lấy dữ liệu `sailor2/sea-ultrafeedback-onpolicy`, lọc Vietnamese, chia theo prompt với seed 42.
- [ ] Đọc số lượng thực sau lọc max_len; nếu không đủ 800/100, xử lý nguyên nhân hoặc báo đúng số lượng và cấu hình, không ghi số mặc định như kết quả thật.
- [ ] Kiểm tra assert không có prompt chung giữa train/held-out.
- [ ] Đọc 3 cặp mẫu, ghi nhận chất lượng chosen/rejected, lỗi nhãn và khả năng chosen chỉ dài hơn.
- [ ] Ghi tỉ lệ chosen dài hơn rejected từ stats và lưu PNG; giữ `stats.json` bên ngoài vùng bị ignore hoặc thêm ngoại lệ có chủ đích khi thực hiện.
- [ ] Không chạy lại NB2 sau NB3 bằng seed/filter khác. Nếu split đổi, phải huấn luyện lại NB3 hoặc khôi phục split gốc.

**Checkpoint:** 800/100 hoặc số lượng khác được giải thích, không trùng prompt, đủ 3 nhận xét và bằng chứng bias độ dài.

## Task 5: NB3 — huấn luyện DPO và chẩn đoán (24 điểm)

**Files:** chạy `notebooks/03_dpo_train.py`; đầu ra adapter DPO, `adapter_config.json`, `dpo_metrics.json`, `split.json`, `03-dpo-reward-curves.png`.

**Interfaces:** input là merged SFT và parquet NB2; output là LoRA DPO trên SFT và metrics JSON cho REFLECTION.

- [ ] Trước training xác nhận `models/sft-merged/` và train/eval parquet tồn tại, cấu hình beta/lr/epoch đúng.
- [ ] Đo thêm thời gian và peak GPU memory vì metrics hiện có chưa tự ghi hai trường này. Đặt ngay quanh training trong bản nguồn nếu bổ sung:

```python
import time
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
result = trainer.train()
elapsed_seconds = time.perf_counter() - started
peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
```

  Thêm đúng các số đo vào metrics khi lưu; ghi rõ allocated/reserved, không gọi đây là toàn bộ VRAM của mọi tiến trình.
- [ ] Chờ precompute reference log-prob rồi training khoảng 100 bước theo cấu hình; giữ eval định kỳ, không bỏ held-out để tiết kiệm thời gian.
- [ ] Kiểm tra 4 đường chosen/rejected trên train/held-out trong PNG; margin đơn lẻ không đáp ứng tiêu chí.
- [ ] Đọc `end_chosen_reward`, `end_rejected_reward`, `end_reward_gap`, `eval_chosen_reward`, `eval_rejected_reward`, `eval_reward_gap`, `eval_reward_accuracy`, `diagnosis`.
- [ ] Kiểm tra adapter reference đúng SFT và split fingerprint đúng dữ liệu. Loss tại khởi tạo lý thuyết bằng log2; first logged loss có thể đã sau vài bước nên không buộc bằng chính xác 0.693.
- [ ] Ghi nhận INTENDED, LIKELIHOOD DISPLACEMENT, FAILURE hoặc AMBIGUOUS bằng đường thực tế. Không đổi nhãn hay tự điền metrics để có kết luận tốt.
- [ ] Sao lưu metrics/config/split/PNG ngay; trọng số chỉ lưu ngoài Git nếu cần phục hồi.

**Checkpoint:** training hoàn tất, reference đúng, reward held-out có số liệu, diagnosis được giải thích.

## Task 6: NB4 — so sánh và judge (16 điểm)

**Files:** chạy `notebooks/04_compare_and_eval.py`; đầu ra `side_by_side.jsonl`, `judge_results_rm.json`, `judge_summary.json`, `04-side-by-side-table.png`.

**Interfaces:** sử dụng chính SFT reference và DPO adapter NB3; judge summary gắn SHA-256 với file output để tránh kết quả cũ.

- [ ] Trong bundle, NB3b nằm trước NB4. Bỏ qua các cell bonus NB3b khi làm core, chạy cell cleanup cần thiết trước NB4; không bấm Run all trên bundle chứa toàn bộ bonus nếu chỉ định làm NB0–NB4.
- [ ] Sinh câu trả lời trên cùng 8 prompt cố định và ≥50 held-out khác nhau; kiểm tra đủ 4 helpfulness, 4 safety và 50 held-out, ít nhất 58 bản ghi.
- [ ] Giữ cùng cấu hình generation và chat template, không trộn kết quả từ các adapter/lần chạy khác nhau.
- [ ] Chạy panel reward model mặc định, hai model nạp lần lượt, không cần API key. Lưu sanity từng giám khảo và giám khảo nào bị loại nếu accuracy <0.8.
- [ ] Đọc riêng `heldout.n`, win rate + CI95, `length_matched_win_rate`, `longer_answer_won_frac`, `score_length_spearman`, `per_judge`, `judge_agreement`.
- [ ] Khi một metric không có đủ mẫu hoặc là null, ghi không đủ dữ liệu và số mẫu; không chuyển thành 0.
- [ ] Nếu CI chứa 0.5: kết luận chưa đủ bằng chứng cải thiện. Nếu sanity <0.8: kết luận judge chưa đáng tin. Thảo luận cả hai judge thuộc Skywork dù base Qwen/Llama khác nhau.
- [ ] Chọn 1 helpfulness và 1 safety từ output thật để giải thích chất lượng/độ dài/độ an toàn.
- [ ] Kiểm tra `outputs_sha256` trong summary khớp file output cuối; sao lưu cả JSONL và kết quả chấm.

**Checkpoint:** ≥50 held-out được chấm, CI/sanity/bias được báo, có 2 ví dụ thật và 4 PNG core.

## Task 7: REFLECTION và kiểm tra nộp bài (30 điểm)

**Files:** điền `submission/REFLECTION.md`; giữ bản notebook executed; giữ bằng chứng trong repo và file verify log từ runtime gốc.

**Interfaces:** REFLECTION đọc stats NB2, metrics NB3, judge summary/output NB4. Verify phải chạy trên runtime chứa mô hình và đủ nguồn, trước khi thu gọn bài nộp.

- [ ] §1: thông tin học viên, ngày chạy, GPU/model/dataset/count/epoch/beta/lr/judge/chi phí thực tế.
- [ ] §2: thời gian/VRAM đo thật, reward train/held-out, accuracy, diagnosis, chiều dài SFT→DPO; dữ liệu chưa đo phải ghi rõ.
- [ ] §3 ≥100 từ: phân tích train và held-out riêng; chosen tăng/giảm, rejected, margin, overfit/displacement, đối chiếu diagnosis.
- [ ] §4: điền bảng 3 nhóm từ summary, CI/sanity/bias/per_judge và hai ví dụ cụ thể.
- [ ] §6 ≥150 từ: một quyết định, phương án thay thế, lý do chọn, kết quả và thay đổi khi làm lại. Chọn beta=0.1 hoặc local panel là hai lựa chọn có dữ liệu để bàn.
- [ ] Nếu không làm bonus, ghi chưa thực hiện; không tick hoàn thành. §5 có thể viết 3 câu giả thuyết như template yêu cầu, phân biệt giả thuyết với đo đạc.
- [ ] Chạy tests phù hợp bằng `python -m pytest scripts/ -q`; giữ kết quả thật. Tests logic qua không thay thế training/eval evidence.
- [ ] Trên `/content/lab22`, có đủ nguồn repo và REFLECTION đã điền, chạy `python scripts/verify.py` (tương đương target `make verify`); lưu stdout + exit code 0. Chạy `make verify` nếu dùng repo đầy đủ và make trên Colab/Linux.
- [ ] `verify.py` so đường tuyệt đối trong adapter config với `models/sft-merged` tại repo hiện tại. Chuyển adapter config `/content/lab22/...` về Windows sẽ gây WRONG REF; không sửa đường để giả thành training tại Windows. Giữ provenance Colab, verify tại Colab trước khi chuyển, giải thích trong báo cáo.
- [ ] Verifier không kiểm tra output notebook, số từ hoặc đầy đủ mọi tiêu chí rubric. Tự kiểm thêm các phần này dù exit 0.
- [ ] Bundle chứa bonus và chỉ bootstrap package, nên để chứng minh core tái lập từ môi trường sạch: dùng repo đầy đủ + `make pipeline` trên CUDA Linux/Colab sau khi hoàn thiện NB0; hoặc notebook core đã tổ chức rõ phạm vi. Không báo đã rerun nếu mới kiểm code.

**Checkpoint:** §1/2/3/4/6 đầy đủ, verify exit 0 trong môi trường gốc, đủ evidence và phân biệt đã đo với giả thuyết.

## Task 8: Sao lưu, GitHub và LMS

**Files:** commit nguồn + executed notebook + PNG + metrics/config/split + evaluation JSON/JSONL + parquet split. Weights/merged model giữ ngoài Git.

- [ ] Sau mỗi mốc NB1/NB3/NB4 tải evidence về; nếu cần khôi phục giữa phiên, sao lưu cả mô hình và adapter/tokenizer lên Drive. Chỉ tải JSON không đủ chạy lại NB4 sau khi runtime bị mất.
- [ ] Tải notebook đã chạy thành `colab/Lab22_DPO_T4_executed.ipynb`; kiểm tra execution counts/output NB0–NB4 không rỗng và không có traceback chưa xử lý.
- [ ] Giữ nguyên bundle generated không output; không chạy builder đè vào bản executed.
- [ ] Kiểm tra 4 PNG tên đúng: `02-sft-loss.png`, `02b-pref-length.png`, `03-dpo-reward-curves.png`, `04-side-by-side-table.png`.
- [ ] Kiểm tra `.gitignore` không làm mất metrics/split/eval/parquet hoặc `stats.json` cần giữ; không dùng `git add -f` cho weights/.env.
- [ ] Kiểm tra nội dung notebook/output không lộ API/HF token và không có model weight/cache trong staged files.
- [ ] Commit/push theo quyền và lựa chọn của người dùng ở lúc thực hiện; kế hoạch này không tự commit/push hay nộp LMS.
- [ ] Xác minh repo remote thật sự public, file evidence và notebook output đã hiện trên GitHub. Local có file chưa đồng nghĩa remote đã nhận.
- [ ] Nộp URL repo public vào LMS theo hạn chính thức của lớp; không tạo PR.

**Checkpoint cuối:** core evidence đã có trên GitHub public và link đã nộp LMS; báo riêng nếu mới hoàn thành local/Colab mà chưa nộp.

## Bonus sau core

| Lựa chọn | Điểm | Điều kiện/kế hoạch |
|---|---:|---|
| Chấm chéo khác họ | +4 | Chấm lại chính `side_by_side.jsonl` bằng API judge, model ID đang hoạt động, lưu agreement; chỉ dùng credentials/cost được người dùng chấp thuận. |
| NB3b | +8 | Chạy DPO/RPO/DPO-norm/LD-DPO/ORPO trên cùng subset, đọc độ dài và reward held-out, điền §8. |
| Beta sweep | +6 | Chạy beta .05/.1/.5, giữ split và config chung, vẽ margin/accuracy và giải thích ≥100 từ; tốn ba lượt DPO. |
| NB5 GGUF | +4 | Merge đúng SFT+DPO, xuất Q4_K_M, đối chiếu HF/GGUF trong deploy_meta. |
| NB6 benchmark | +6 | IFEval/GSM8K/Global-MMLU-vi với chat template, giới hạn theo môn, phân tích stderr §7. |
| NB7 GRPO | +8 | Reward và accuracy trước/sau + nhiễu, tốn thêm training; làm cuối. |
| HF Hub | +3 | Adapter + model card, công bố khi người dùng muốn; ghi model/dataset/hyperparameters/evaluation. |

Đề xuất có thời gian: chấm chéo (+4) + NB3b (+8) + GGUF (+4) + HF Hub (+3) = +19; phụ thuộc API/GPU và nhu cầu công bố. Nếu muốn đúng +20: NB3b (+8) + beta sweep (+6) + NB6 (+6), nhưng chi phí GPU lớn hơn. Không coi điểm bonus là đã đạt trước khi có bằng chứng. `BONUS-CHALLENGE.md` là dự án domain 4–8 giờ KHÔNG CHẤM ĐIỂM, khác bảng bonus rubric.

## Lịch thực hiện dự kiến

| Mốc | Thời lượng dự kiến | Kết quả |
|---|---|---|
| Chuẩn bị + NB0 | 20–40 phút | CPU assertions, Colab T4, nguồn đồng bộ |
| NB1 | 15–25 phút GPU, chưa gồm tải/cài | SFT + merged reference + PNG |
| NB2 | 2–10 phút | split không trùng + 3 cặp + bias |
| NB3 | 40–60 phút GPU | DPO adapter + reward curves + metrics |
| NB4 | 20–30 phút GPU | 58+ outputs + judge + CI |
| REFLECTION + verify + kiểm remote | 60–120 phút | Bài nộp có bằng chứng |

Dự trù một buổi 3–5 giờ cho core, có thể lâu hơn vì tải Hugging Face, pip/build, quota hoặc lỗi runtime. Hoàn thiện 100 điểm core trước khi quyết định bonus.

## Self-review kế hoạch

- [x] Bao phủ từng hàng rubric: NB0=10, NB1=8, NB2=12, NB3=24, NB4=16, phản tư=20, tái lập=5, verify=5, tổng 100.
- [x] Đã phân biệt runtime weights và Git evidence; đồng bộ generated bundle và giữ notebook executed riêng.
- [x] Đã xử lý rủi ro None bypass NB0, Run all bao gồm bonus, nguồn verify thiếu trong bundle, đường tuyệt đối Colab→Windows, stats bị ignore, metrics thời gian/VRAM chưa có.
- [x] Không khẳng định test/training/eval/submission đã hoàn thành khi chưa chạy.
- [x] Interface dùng `my_dpo_loss`, `models/sft-merged`, split fingerprint, outputs SHA và schema metrics hiện có nhất quán.

## Tiến độ thực hiện và cập nhật được người dùng chọn

- Người dùng sẽ chạy notebook Core trên Colab T4 và gửi kết quả; công cụ trình duyệt hiện lỗi khởi tạo sandbox.
- NB4 chuyển sang LLM API DevQuota OpenAI-compatible với `gpt-6-sol`, có sanity tiếng Việt và đổi vị trí A/B; Core yêu cầu API, không fallback local.
- Đã hoàn thiện NB0, chạy thật 9/9 code cells CPU và lưu output; tests CPU cuối: 70 passed, không skip.
- Đã tạo `scripts/build_core_colab.py`, `colab/Lab22_DPO_T4_Core.ipynb`, `docs/CORE_RUN_GUIDE.md`; core-only Run all, nguồn verifier đầy đủ, checkpoint/evidence ZIP, đo tài nguyên và giữ provenance.
- Key đang có trên Windows trả HTTP 401; API/model phải được kiểm tra bằng key hợp lệ trong Colab Secrets trước training.
- NB1–NB4 GPU chưa chạy; chưa có REFLECTION theo kết quả thực nghiệm. Full verify local exit 1 đúng với artifacts còn thiếu, log trong `submission/local_verify.txt`.
- Chưa commit/push/nộp LMS. Các bước tiếp theo: nhận ZIP + notebook executed, kiểm chứng bằng chứng, hoàn thiện REFLECTION, verify trong runtime Colab gốc, rồi công bố/nộp theo yêu cầu người dùng.

## Cập nhật sau ZIP thực nghiệm — 08/10/2026

- Đã nhập/audit 22 files, 40 source hashes gốc khớp; split 800/100 không overlap; NB4 58 rows/votes/summary/CI tính lại khớp.
- Đã điền REFLECTION bằng số đo thật và thông tin Nguyễn Nhân Sâm, 2A202602672, lớp 3A, khóa IV. §3 có 293 từ, §6 có 347 từ theo whitespace; checker report/judge/ảnh không có lỗi.
- SFT 642.1644 giây; DPO 1650.6945 giây; held-out reward accuracy70%, NB4 WR50% CI42–57%. Cả chosen/rejected tăng,36/58 answers giống nhau, tất cả còn tool markers; các hạn chế được nêu trong báo cáo.
- Đã đổi tên GitHub repo public và origin thành K4-L3-Track3-Day22-NguyenNhanSam-2A202602672. Chưa commit/push bài lab.
- 70 tests mới nhất qua; full verifier Windows exit1 do thiếu merged config và đường Colab khác Windows, không còn lỗi REFLECTION. Checker/tests gốc không sửa.
- Đã tạo READINESS.md và APPLY_REFLECTION_COLAB.py; helper kiểm tra hash outputs/verifier, lưu originals, chạy verify gốc và chỉ export cuối khi exit0.
- Còn cần notebook executed NB1–NB4, fullverify0 Colab, kiểm tra secrets/output, công bố evidence và nộp LMS. Snapshot notebook nguồn giữ archive REFLECTION template của lần đã chạy; không tái sinh với báo cáo đo đạc để giả provenance.

### Nhận ZIP cuối sau finalization

- Lab22_Core_Evidence_Final.zip: SHA256 70d01ddd34226fa128dd1662f9aeb3ecc107276a4b8c7732931af29add731a57; 27 files.
- Đã nhận fullverify exit0 tại runtime gốc, finalization hashes report/figure khớp; training/split/judge/raw outputs không đổi.
- Đã nhập log thành công và giữ log exit1 ban đầu tại colab_original. Chưa nhận notebook executed NB1–NB4; chưa commit/push bài lab/nộp LMS.
### Nhận notebook executed — hoàn tất bằng chứng local

- Đã nhận `Downloads/Lab22_DPO_T4_Core.ipynb`, lưu nguyên byte vào `colab/Lab22_DPO_T4_executed.ipynb`.
- Notebook format hợp lệ; 46/46 code cells chạy, 44 cells có output, không error output; source sequence khớp snapshot Core ngoài cell finalization thêm.
- Cell finalization execution_count46 chứa fullverify exit0; giữ log exit1 template cũ như lịch sử.
- Không thấy mẫu secret token thông thường; audit tại `submission/executed_notebook_audit.json`.
- Bộ evidence bắt buộc local đầy đủ; chưa commit/push bằng chứng hoặc nộp LMS. Trạng thái hiện tại ở `submission/FINAL_REVIEW.md` và `READINESS.md`.
