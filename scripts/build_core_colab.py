#!/usr/bin/env python3
"""Generate a standalone T4 notebook for NB0–NB4, with sources and evidence export.

The existing full/bonus bundles and verifier remain authoritative and unchanged.
Run: python scripts/build_core_colab.py [--check]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

if __package__:
    from . import build_colab as B
else:
    import build_colab as B

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "colab/Lab22_DPO_T4_Core.ipynb"
CORE = [stem for stem, kind in B.STAGES if kind.startswith("core")]


def source_files(root: Path) -> list[Path]:
    paths = [
        root / name for name in (
            "README.md", "rubric.md", "HARDWARE-GUIDE.md", "requirements.txt", "requirements-biggpu.txt",
            "pyproject.toml", "Makefile", ".gitignore", ".env.example", "docs/reference.md",
            "submission/REFLECTION.md", "submission/screenshots/README.md",
        )
    ]
    for folder in ("lab22", "notebooks", "scripts"):
        paths.extend(sorted((root / folder).glob("*.py")))
    return [path for path in paths if path.is_file()]


def source_archive(root: Path) -> tuple[str, dict[str, str]]:
    buffer = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(root):
            name = path.relative_to(root).as_posix()
            raw = path.read_bytes()
            manifest[name] = hashlib.sha256(raw).hexdigest()
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, raw)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), manifest


def export_evidence(root: Path, output: Path) -> Path:
    """Allowlist small evidence; weights, tokenizers, caches and .env are excluded."""
    paths = set()
    for pattern in (
        "adapters/*/adapter_config.json", "adapters/*/dpo_metrics.json", "adapters/*/sft_metrics.json",
        "adapters/*/split.json", "adapters/*/reward_history.json", "data/pref/*.parquet",
        "data/pref/stats.json", "data/pref/samples.json", "data/eval/*.json", "data/eval/*.jsonl",
        "submission/screenshots/*.png", "submission/*.md", "submission/*.json", "submission/*.txt",
        "notebooks/00_dpo_loss_from_scratch.ipynb",
    ):
        paths.update(root.glob(pattern))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    return output


def tagged_code(text: str, role: str) -> dict:
    cell = B.code(text)
    cell["metadata"]["lab22_role"] = role
    return cell


def render_core() -> dict:
    payload, manifest = source_archive(REPO)
    specs = [s for s in B.requirements() if not s.startswith(("llama-cpp-python", "lm-eval", "vllm", "flash-attn"))]
    specs.append("pytest>=8.3,<10.0")
    cells = [
        B.md(
            "# Lab 22 — Phần bắt buộc NB0–NB4 · Colab T4\n\n"
            "Chọn **Runtime → Change runtime type → T4 GPU**, rồi **Run all**. "
            "Notebook này không chạy bonus. Mọi nguồn cần thiết được đóng gói bên trong; không cần clone repo.\n\n"
            "Sau NB4 tải `Lab22_Core_Evidence.zip` và **File → Download → Download .ipynb** "
            "để giữ output. Gửi lại cả hai file để hoàn thiện REFLECTION bằng số liệu thật. "
            "Giữ runtime mở để chạy kiểm tra cuối sau khi hoàn thiện REFLECTION.\n\n"
            "Colab hết phiên sẽ mất model: ZIP bằng chứng không chứa weights. Nếu cần phục hồi "
            "giữa phiên, sao lưu riêng `models/sft-merged`, adapter và tokenizer sang Drive.\n\n"
            "**NB4 dùng DevQuota API / gpt-6-sol.** Trong bảng Secrets (biểu tượng chìa khoá), "
            "thêm `OPENAI_API_KEY`, bật Notebook access. `OPENAI_BASE_URL` là secret tuỳ chọn nếu "
            "endpoint khác `https://sv.devquote.shop/v1`. Không dán key vào code. API được thử "
            "trước training; thiếu key hoặc model không hoạt động sẽ dừng, không chạy local judge. "
            "Khoảng 142 lượt gọi bình thường: 2 preflight + 24 sanity + 116 chấm hai vị trí cho 58 câu; "
            "có thể thêm lượt retry. Chi phí phụ thuộc biểu giá/tài khoản DevQuota."
        ),
        tagged_code(
            "import os\nfrom pathlib import Path\n"
            'os.environ["COMPUTE_TIER"] = "T4"\n'
            'os.environ["JUDGE_PROVIDER"] = "openai"\n'
            'os.environ["JUDGE_MODEL"] = "gpt-6-sol"\n'
            'os.environ["JUDGE_REQUIRE_API"] = "1"\n'
            'os.environ["JUDGE_MAX_TOKENS"] = "4096"\n'
            'os.environ["OPENAI_BASE_URL"] = "https://sv.devquote.shop/v1"\n'
            'from google.colab import userdata\n'
            'try:\n'
            '    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")\n'
            'except Exception:\n'
            '    raise RuntimeError("Thêm OPENAI_API_KEY vào Colab Secrets và bật Notebook access") from None\n'
            'try:\n'
            '    _base_url = userdata.get("OPENAI_BASE_URL")\n'
            'except Exception:\n'
            '    _base_url = None\n'
            'if _base_url:\n'
            '    os.environ["OPENAI_BASE_URL"] = _base_url\n'
            'os.environ["SEED"] = "42"\n'
            'os.environ["WANDB_DISABLED"] = "true"\n'
            '# Nếu OOM, bỏ comment MAX_LEN rồi restart runtime và chạy lại từ đầu.\n'
            '# os.environ["MAX_LEN"] = "512"\n'
            'WORK = Path("/content/lab22")\n'
            'STUDENT_NAME = ""  # Điền họ tên nếu muốn lưu cùng cấu hình; không phải API key.\n'
            'STUDENT_CLASS = ""\n', "settings"
        ),
        tagged_code("!pip install -q " + " ".join(f'"{s}"' for s in specs), "install"),
        tagged_code(
            "import base64, io, json, zipfile\n"
            "from pathlib import Path\n"
            "WORK.mkdir(parents=True, exist_ok=True)\n"
            "(WORK / 'colab').mkdir(exist_ok=True)\n"
            f"_payload = {payload!r}\n"
            "with zipfile.ZipFile(io.BytesIO(base64.b64decode(_payload))) as _archive:\n"
            "    _archive.extractall(WORK)\n"
            '(WORK / "submission/source_manifest.json").write_text(\n'
            f"    json.dumps({manifest!r}, ensure_ascii=False, indent=2), encoding='utf-8')\n"
            "print('Nguồn được khôi phục:', len(" + repr(manifest) + "), 'file')", "bootstrap"
        ),
        tagged_code(
            "import os, sys, json, platform, subprocess, importlib.metadata\n"
            "from datetime import datetime\nfrom zoneinfo import ZoneInfo\n"
            "os.chdir(WORK)\nsys.path.insert(0, str(WORK))\n"
            "from lab22 import judge as J\n"
            "J.require_api_judge(os.environ['JUDGE_PROVIDER'], os.environ['JUDGE_MODEL'])\n"
            "_call = J.make_caller(os.environ['JUDGE_PROVIDER'], os.environ['JUDGE_MODEL'],\n"
            "                      max_tokens=int(os.environ['JUDGE_MAX_TOKENS']))\n"
            "_probe = J.api_sanity_accuracy(_call, pairs=J.SANITY_PAIRS[:1])\n"
            "assert _probe['accuracy'] == 1.0, 'API probe chưa trả đúng JSON/đáp án ở cả hai vị trí; kiểm tra key/model'\n"
            "print('API preflight:', os.environ['JUDGE_MODEL'], _probe)\n"
            "import unsloth  # patch trước torch/transformers/trl\nimport torch\n"
            "assert torch.cuda.is_available(), 'Cần chọn T4 GPU trước khi chạy'\n"
            "from lab22 import config as C\nC.ensure_dirs()\n"
            "_gpu = torch.cuda.get_device_properties(0)\n"
            "_runtime = {\n"
            "    'student_name': STUDENT_NAME, 'student_class': STUDENT_CLASS,\n"
            "    'started_at': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),\n"
            "    'python': platform.python_version(), 'gpu': _gpu.name, 'vram_gb': _gpu.total_memory / 1e9,\n"
            "    'cuda': torch.version.cuda, 'seed': C.SEED, 'max_len': C.MAX_LEN, 'tier': C.COMPUTE_TIER,\n"
            "    'base_model': C.BASE_MODEL, 'sft_slice': C.SFT_SLICE,\n"
            "    'pref_train_requested': C.PREF_TRAIN, 'pref_eval_requested': C.PREF_EVAL,\n"
            "    'beta': C.DPO_BETA, 'lr': C.DPO_LR, 'epochs': C.DPO_EPOCHS,\n"
            "    'judge_provider': C.JUDGE_PROVIDER, 'judge_model': C.JUDGE_MODEL, 'api_preflight': _probe,\n"
            "    'versions': {name: importlib.metadata.version(name) for name in\n"
            "                 ['torch', 'unsloth', 'trl', 'transformers', 'peft', 'datasets']},\n"
            "}\n"
            "(C.REPO_ROOT / 'submission/runtime_config.json').write_text(\n"
            "    json.dumps(_runtime, ensure_ascii=False, indent=2), encoding='utf-8')\n"
            "print(json.dumps(_runtime, ensure_ascii=False, indent=2))\n"
            "subprocess.run([sys.executable, 'scripts/build_colab.py'], check=True)\n"
            "subprocess.run([sys.executable, '-m', 'pytest', 'scripts/', '-q'], check=True)", "preflight"
        ),
    ]
    cells[2]["metadata"]["lab22_requirements"] = specs
    for i, stem in enumerate(CORE):
        if i:
            cells.append(B.code(B.RELEASE_GPU))
        header = B.md(f"---\n# {stem} — bắt buộc")
        header["metadata"]["lab22_stage"] = stem
        cells.append(header)
        cells.extend(B.percent_cells(REPO / "notebooks" / f"{stem}.py"))
        if stem != "00_dpo_loss_from_scratch":
            cells.append(tagged_code(
                "from scripts.build_core_colab import export_evidence\n"
                f"_checkpoint = export_evidence(C.REPO_ROOT, C.REPO_ROOT / 'Lab22_Checkpoint_{i}.zip')\n"
                "print('Checkpoint bằng chứng:', _checkpoint)\n"
                "print('Tải checkpoint này nếu phải rời phiên; không có trọng số để phục hồi model.')",
                "checkpoint"
            ))
    cells.extend([
        B.md(
            "## Kiểm tra kết quả và tải bằng chứng\n\n"
            "Kiểm tra core training/eval trước. `verify.py` đầy đủ còn yêu cầu REFLECTION được điền; "
            "kết quả verify bên dưới được giữ nguyên, không đổi tiêu chí. "
            "Sau khi nhận REFLECTION hoàn chỉnh, ghi file đó vào `submission/REFLECTION.md`, "
            "chạy lại `!python scripts/verify.py`, rồi chạy lại cell export này."
        ),
        tagged_code(
            "import hashlib, subprocess, sys, json\n"
            "from pathlib import Path\n"
            "_eval = C.EVAL_DIR\n"
            "_summary = json.loads((_eval / 'judge_summary.json').read_text(encoding='utf-8'))\n"
            "_rows = [json.loads(line) for line in (_eval / 'side_by_side.jsonl').read_text(encoding='utf-8').splitlines() if line]\n"
            "assert _summary['heldout']['n'] >= 50, 'Chưa chấm đủ 50 held-out'\n"
            "assert sum(r['category'] == 'helpfulness' for r in _rows) == 4\n"
            "assert sum(r['category'] == 'safety' for r in _rows) == 4\n"
            "assert _summary['outputs_sha256'] == hashlib.sha256((_eval / 'side_by_side.jsonl').read_bytes()).hexdigest()\n"
            "for _png in ('02-sft-loss', '02b-pref-length', '03-dpo-reward-curves', '04-side-by-side-table'):\n"
            "    assert (C.SCREENSHOTS / (_png + '.png')).stat().st_size > 0\n"
            "_verify = subprocess.run([sys.executable, 'scripts/verify.py'], capture_output=True, text=True, encoding='utf-8')\n"
            "print(_verify.stdout)\nprint('Full verify exit code:', _verify.returncode)\n"
            "(C.REPO_ROOT / 'submission/verify_colab.txt').write_text(_verify.stdout + _verify.stderr, encoding='utf-8')\n"
            "(C.REPO_ROOT / 'submission/verify_status.json').write_text(\n"
            "    json.dumps({'exit_code': _verify.returncode, 'core_eval_checks_passed': True}, indent=2), encoding='utf-8')\n"
            "from scripts.build_core_colab import export_evidence\n"
            "_zip = export_evidence(C.REPO_ROOT, C.REPO_ROOT / 'Lab22_Core_Evidence.zip')\n"
            "print('ZIP bằng chứng:', _zip)\n"
            "from google.colab import files\nfiles.download(str(_zip))\n"
            "print('Tải thêm File → Download → Download .ipynb để giữ output!')", "export"
        ),
    ])
    for index, cell in enumerate(cells):
        cell["id"] = f"core-{index:03d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    notebook = render_core()
    if args.check:
        if not TARGET.is_file() or json.loads(TARGET.read_text(encoding="utf-8")) != notebook:
            print("Core bundle stale: run python scripts/build_core_colab.py")
            return 1
        print("Core bundle is current")
        return 0
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET} ({len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
