print("Running test_demos.py")

import pytest
import subprocess
import os
import sys
import numpy as np
import re

# Discover all demo scripts and notebooks
DEMO_FILES = []
for dirpath, _, filenames in os.walk('demos'):
    for f in filenames:
        if f.endswith('.py') or f.endswith('.ipynb'):
            DEMO_FILES.append(os.path.join(dirpath, f))

# Separate notebooks from scripts
NOTEBOOK_FILES = [f for f in DEMO_FILES if f.endswith('.ipynb')]
PYTHON_SCRIPTS = [f for f in DEMO_FILES if f.endswith('.py')]

@pytest.mark.parametrize("script_path", PYTHON_SCRIPTS)
def test_demo_script_execution(script_path):
    """
    Executes a demo script and checks for successful completion and small numerical error.
    """
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script {script_path} failed with exit code {result.returncode}.\nStderr:\n{result.stderr}"

    # Check for numerical errors in the output
    numerical_errors = re.findall(r"Error:\s*([\d.eE+-]+)", result.stdout)
    if numerical_errors:
        for error in numerical_errors:
            assert np.isclose(float(error), 0.0), f"Script {script_path} produced a significant numerical error: {error}"


@pytest.mark.parametrize("notebook_path", NOTEBOOK_FILES)
def test_notebook_execution(notebook_path):
    """
    Executes a notebook and checks for successful completion.
    """
    result = subprocess.run(
        ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', notebook_path],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Notebook {notebook_path} failed to execute.\nStderr:\n{result.stderr}"
