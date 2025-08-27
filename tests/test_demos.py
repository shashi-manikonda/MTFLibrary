import pytest
import subprocess
import os
import sys

# Get the single demo script to test
DEMO_SCRIPT = 'demos/em/Magnetic_Field_Calculation_Demo.py'

@pytest.mark.skip(reason="Demo script is too slow to run as part of the regular test suite.")
@pytest.mark.parametrize("script_path", [DEMO_SCRIPT])
def test_demo_script_execution(script_path):
    """
    Executes a demo script and checks for successful completion.
    """
    # Run the demo script
    # We use sys.executable to ensure we're using the same python interpreter
    # that's running pytest.
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script {script_path} failed with exit code {result.returncode}.\nStderr:\n{result.stderr}"
