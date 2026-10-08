"""Exercise the student's NB0 function, without importing/executing the notebook."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def student_loss():
    tree = ast.parse((REPO / "notebooks/00_dpo_loss_from_scratch.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "my_dpo_loss")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<student-loss>", "exec"), namespace)
    return namespace["my_dpo_loss"]


def test_student_init_is_log2(student_loss):
    c, r = torch.tensor([-12.0, -30.0]), torch.tensor([-15.0, -28.0])
    actual = student_loss(c, r, c, r)
    assert isinstance(actual, torch.Tensor), "NB0 must return a scalar Tensor, not None"
    assert actual.ndim == 0
    assert actual.item() == pytest.approx(math.log(2), abs=1e-6)


def test_student_matches_hand_calculated_closed_form(student_loss):
    # beta * ((-10 - -12) - (-20 - -15)) = 3.5.
    t = torch.tensor
    actual = student_loss(t([-10.0]), t([-20.0]), t([-12.0]), t([-15.0]), beta=0.5)
    assert isinstance(actual, torch.Tensor), "NB0 loss is not implemented"
    assert actual.item() == pytest.approx(math.log1p(math.exp(-3.5)), abs=1e-6)


def test_student_averages_batch_and_preserves_gradient(student_loss):
    pc = torch.tensor([1.0, -1.0], requires_grad=True)
    zero = torch.zeros(2)
    actual = student_loss(pc, zero, zero, zero, beta=1.0)
    assert isinstance(actual, torch.Tensor), "NB0 loss is not implemented"
    assert actual.item() == pytest.approx(0.8132616875, abs=1e-6)
    actual.backward()
    assert pc.grad.tolist() == pytest.approx([-0.1344707107, -0.3655292893], abs=1e-6)


def test_student_displacement_can_have_same_loss(student_loss):
    rc, rr = torch.tensor([-20.0]), torch.tensor([-22.0])
    intended = student_loss(rc + 1, rr - 1, rc, rr, beta=1.0)
    displaced = student_loss(rc - 3, rr - 5, rc, rr, beta=1.0)
    assert isinstance(intended, torch.Tensor) and isinstance(displaced, torch.Tensor)
    assert intended.item() == pytest.approx(0.126928011, abs=1e-6)
    assert torch.allclose(intended, displaced)


def test_student_stays_finite_for_extreme_margins(student_loss):
    zero = torch.zeros(2)
    actual = student_loss(torch.tensor([1000.0, -1000.0]), zero, zero, zero, beta=1.0)
    assert isinstance(actual, torch.Tensor)
    assert actual.item() == pytest.approx(500.0)
