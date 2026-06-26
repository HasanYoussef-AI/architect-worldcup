"""Independence gate: Prediction B must remain blind to Prediction A.

This gate is what keeps the A-versus-B headline meaningful. The whole comparison is
that two systems forecast the same fixture independently; if B could read A, B's
number would be contaminated by A's and the comparison would be worthless. So B's
loader imports no A module, and B's build, given only a dossier, never opens A's
file. C is allowed to read both A and B; that asymmetry is the point, and it is why
A's drivers explainer lives only in A's output.
"""

from __future__ import annotations

import ast
import inspect

from architect_wc.llm import prediction_b


def _imported_names(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def test_b_loader_imports_no_a_module() -> None:
    imported = _imported_names(prediction_b)
    a_or_c = {
        name
        for name in imported
        if name.split(".")[-1] in {"prediction_a", "reconcile_c"}
    }
    assert not a_or_c, f"B's loader imports A or C: {a_or_c}"


def test_b_build_takes_a_dossier_not_an_a_document() -> None:
    # The signature can take a dossier and a model response, never an A document or an
    # A artifact reference, so B cannot read A by construction.
    for func in (prediction_b.build_prediction_b, prediction_b.predict_b):
        params = set(inspect.signature(func).parameters)
        assert "prediction_a" not in params
        assert "a_ref" not in params
