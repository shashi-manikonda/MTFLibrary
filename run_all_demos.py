import os
import sys
import subprocess
import platform

def run_demos():
    """
    Finds and runs all .py and .ipynb demos in the 'demos' directory.
    For .ipynb files, it converts them to .py using jupytext first.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')

    print(f"Ensuring '{src_path}' is in Python path for subprocesses.")

    demos_dir = os.path.join(project_root, 'demos')
    python_executable = "python" if platform.system() == "Windows" else "python3"

    for root, _, files in os.walk(demos_dir):
        for file in files:
            file_path = os.path.join(root, file)
            script_path = None # Initialize script_path

            try:
                if file.endswith('.py'):
                    print(f"\n--- Running Python Demo: {file_path} ---")
                    script_to_run = file_path

                elif file.endswith('.ipynb'):
                    print(f"\n--- Converting and Running Notebook Demo: {file_path} ---")
                    # Convert notebook to .py script using jupytext
                    subprocess.run(['jupytext', '--to', 'py', file_path], check=True, capture_output=True, text=True)
                    script_path = os.path.splitext(file_path)[0] + '.py'
                    script_to_run = script_path
                    print(f"--- Executing converted script: {script_to_run} ---")

                else:
                    continue # Skip non-python/notebook files

                # We need to prepend the src path to PYTHONPATH for the subprocess
                env = os.environ.copy()
                env['PYTHONPATH'] = src_path + os.pathsep + env.get('PYTHONPATH', '')

                subprocess.run([python_executable, script_to_run], check=True, env=env)

            except FileNotFoundError:
                print("Error: 'jupytext' command not found.")
                print("Please install it with: pip install jupytext")
                return # Exit if jupytext is not available
            except subprocess.CalledProcessError as e:
                print(f"!!! ERROR running {file} !!!")
                print(f"Return Code: {e.returncode}")
                if e.stdout:
                    print(f"STDOUT:\n{e.stdout}")
                if e.stderr:
                    print(f"STDERR:\n{e.stderr}")
            finally:
                # Clean up the generated .py file from a notebook
                if script_path and os.path.exists(script_path):
                    os.remove(script_path)

if __name__ == "__main__":
    run_demos()
