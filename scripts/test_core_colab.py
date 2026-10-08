"""Core-only Colab must run NB0–NB4 and reconstruct the unmodified verifier."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def builder():
    path = REPO / "scripts/build_core_colab.py"
    assert path.is_file(), "A standalone core-only Colab builder is required"
    spec = importlib.util.spec_from_file_location("build_core_colab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_colab_has_only_required_stages():
    notebook = builder().render_core()
    stages = [c["metadata"]["lab22_stage"] for c in notebook["cells"] if "lab22_stage" in c["metadata"]]
    assert stages == [
        "00_dpo_loss_from_scratch", "01_sft_mini", "02_preference_data", "03_dpo_train", "04_compare_and_eval"
    ]


def test_core_export_import_works_in_colab_namespace(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(REPO)!r}); from scripts.build_core_colab import export_evidence"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_restores_sources_and_frozen_verifier(tmp_path):
    notebook = builder().render_core()
    bootstrap = next(c for c in notebook["cells"] if c["metadata"].get("lab22_role") == "bootstrap")
    namespace = {"WORK": tmp_path}
    exec("".join(bootstrap["source"]), namespace)
    for relative in ["scripts/verify.py", "lab22/config.py", "notebooks/03_dpo_train.py", "submission/REFLECTION.md"]:
        assert (tmp_path / relative).read_bytes() == (REPO / relative).read_bytes()
    assert not (tmp_path / ".env").exists()


def test_core_colab_does_not_install_bonus_native_builds():
    notebook = builder().render_core()
    install = next(c for c in notebook["cells"] if c["metadata"].get("lab22_role") == "install")
    specs = install["metadata"]["lab22_requirements"]
    assert not any(s.startswith(("llama-cpp-python", "lm-eval", "vllm", "flash-attn")) for s in specs)
    assert any(s.startswith("unsloth") for s in specs)
    assert any(s.startswith("trl") for s in specs)


def test_bootstrap_can_build_original_bundles(tmp_path):
    import subprocess
    import sys

    bootstrap = next(c for c in builder().render_core()["cells"] if c["metadata"].get("lab22_role") == "bootstrap")
    exec("".join(bootstrap["source"]), {"WORK": tmp_path})
    result = subprocess.run(
        [sys.executable, str(tmp_path / "scripts/build_colab.py")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "colab/Lab22_DPO_T4.ipynb").is_file()
    assert (tmp_path / "colab/Lab22_DPO_BigGPU.ipynb").is_file()


def test_core_archive_keeps_evidence_and_excludes_weights_and_secrets(tmp_path):
    module = builder()
    for name in [
        "adapters/dpo/dpo_metrics.json", "adapters/dpo/adapter_config.json", "adapters/dpo/split.json",
        "adapters/dpo/adapter_model.safetensors", "models/sft-merged/config.json", ".env",
        "data/pref/stats.json", "data/pref/train.parquet", "data/eval/judge_summary.json",
        "submission/screenshots/02-sft-loss.png", "submission/REFLECTION.md",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    import zipfile

    archive = module.export_evidence(tmp_path, tmp_path / "evidence.zip")
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
    assert "adapters/dpo/dpo_metrics.json" in names
    assert "data/pref/train.parquet" in names
    assert "data/pref/stats.json" in names
    assert "submission/REFLECTION.md" in names
    assert ".env" not in names
    assert "adapters/dpo/adapter_model.safetensors" not in names
    assert "models/sft-merged/config.json" not in names
