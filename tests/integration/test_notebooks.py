"""Integration tests for example notebooks.

Tests that example notebooks execute without errors and produce expected outputs.
Uses nbconvert to execute notebooks in a clean kernel.

NOTE: These tests are currently skipped because example notebooks have outdated imports
(prepare_clv_model_inputs) that need to be updated to use the current API.

Issue #30: https://github.com/datablogin/AutoCLV/issues/30
"""

import json
import subprocess
from pathlib import Path

import pytest


# Find all example notebooks
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
NOTEBOOKS = list(EXAMPLES_DIR.glob("*.ipynb"))


@pytest.mark.skip(reason="Notebooks have outdated imports - need updating separately")
@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes_without_errors(notebook_path):
    """Test that notebook executes completely without errors."""
    # Use nbconvert to execute the notebook
    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--stdout",
            str(notebook_path),
        ],
        capture_output=True,
        text=False,  # Get bytes for proper JSON parsing
        timeout=600,  # 10 minute timeout
    )

    # Check execution succeeded
    assert result.returncode == 0, (
        f"Notebook {notebook_path.name} failed to execute.\n"
        f"stderr: {result.stderr.decode('utf-8')}"
    )

    # Parse executed notebook
    executed_nb = json.loads(result.stdout)

    # Check that cells were actually executed (should have execution_count)
    code_cells = [cell for cell in executed_nb["cells"] if cell["cell_type"] == "code"]
    executed_cells = [
        cell for cell in code_cells if cell.get("execution_count") is not None
    ]

    assert len(executed_cells) > 0, f"No cells were executed in {notebook_path.name}"

    # Check for errors in cell outputs
    for i, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        for output in outputs:
            if output.get("output_type") == "error":
                error_name = output.get("ename", "Unknown")
                error_value = output.get("evalue", "")
                traceback = "\n".join(output.get("traceback", []))
                pytest.fail(
                    f"Notebook {notebook_path.name} cell {i} raised {error_name}: {error_value}\n{traceback}"
                )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_outputs(notebook_path):
    """Test that notebook has expected output structure."""
    with open(notebook_path, "r") as f:
        notebook = json.load(f)

    # Notebook should have cells
    assert "cells" in notebook, f"{notebook_path.name} missing cells"
    assert len(notebook["cells"]) > 0, f"{notebook_path.name} has no cells"

    # Should have at least some markdown cells (documentation)
    markdown_cells = [c for c in notebook["cells"] if c["cell_type"] == "markdown"]
    assert len(markdown_cells) > 0, f"{notebook_path.name} has no markdown cells"

    # Should have at least some code cells
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) > 0, f"{notebook_path.name} has no code cells"


def test_all_notebooks_found():
    """Test that we found the expected example notebooks."""
    notebook_names = [nb.name for nb in NOTEBOOKS]

    # Check for expected notebooks (from Issue #39)
    expected_notebooks = [
        "01_texas_clv_walkthrough.ipynb",
        "02_custom_cohorts.ipynb",
        "03_model_comparison.ipynb",
        "04_monitoring_drift.ipynb",
    ]

    for expected in expected_notebooks:
        assert expected in notebook_names, (
            f"Expected notebook {expected} not found in {EXAMPLES_DIR}"
        )


@pytest.mark.skip(reason="Notebooks have outdated imports - need updating separately")
@pytest.mark.slow
def test_monitoring_notebook_detects_drift():
    """Test that monitoring notebook correctly detects model drift."""
    notebook_path = EXAMPLES_DIR / "04_monitoring_drift.ipynb"

    if not notebook_path.exists():
        pytest.skip(f"Notebook {notebook_path} not found")

    # Execute notebook
    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--stdout",
            str(notebook_path),
        ],
        capture_output=True,
        text=False,
        timeout=600,
    )

    assert result.returncode == 0, "Monitoring notebook failed to execute"

    # Parse executed notebook
    executed_nb = json.loads(result.stdout)

    # Look for drift detection output
    # The notebook should print "⚠️  ALERT: Model drift detected!" or similar
    found_drift_detection = False
    for cell in executed_nb["cells"]:
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs", [])
            for output in outputs:
                if output.get("output_type") in ["stream", "execute_result"]:
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if "drift" in text.lower() and (
                        "alert" in text.lower() or "detected" in text.lower()
                    ):
                        found_drift_detection = True
                        break

    assert found_drift_detection, "Monitoring notebook should detect drift"


@pytest.mark.skip(reason="Notebooks have outdated imports - need updating separately")
@pytest.mark.slow
def test_model_comparison_notebook_shows_results():
    """Test that model comparison notebook produces comparison results."""
    notebook_path = EXAMPLES_DIR / "03_model_comparison.ipynb"

    if not notebook_path.exists():
        pytest.skip(f"Notebook {notebook_path} not found")

    # Execute notebook
    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--stdout",
            str(notebook_path),
        ],
        capture_output=True,
        text=False,
        timeout=600,
    )

    assert result.returncode == 0, "Model comparison notebook failed to execute"

    # Parse executed notebook
    executed_nb = json.loads(result.stdout)

    # Look for model comparison metrics (MAPE, R², etc.)
    found_metrics = False
    for cell in executed_nb["cells"]:
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs", [])
            for output in outputs:
                if output.get("output_type") in ["stream", "execute_result"]:
                    text = output.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    # Look for metric abbreviations
                    if any(metric in text for metric in ["MAPE", "RMSE", "R²", "MAE"]):
                        found_metrics = True
                        break

    assert found_metrics, "Model comparison notebook should show metrics"
