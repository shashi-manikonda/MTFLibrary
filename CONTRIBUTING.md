# Contributing to MTFLibrary

First off, thank you for considering contributing to `MTFLibrary`! It's people like you that make this tool better for everyone.

This document guides you through the contribution process, from setting up your development environment to submitting a Pull Request.

## Code of Conduct

Please be respectful and considerate of others when interacting with this repository.

## Getting Started

### 1. Fork and Clone
Fork the repository on GitHub, then clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/MTFLibrary.git
cd MTFLibrary
```

### 2. Set Up Development Environment
We recommend using a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode with development dependencies
pip install -e .[dev]
```

### 3. Install Pre-commit Hooks
This project uses [pre-commit](https://pre-commit.com/) to automatically check code style and quality before every commit.

```bash
pre-commit install
```

Now, every time you commit, `black`, `flake8`, and other checks will run automatically.

## Development Workflow

1.  **Create a Branch**: Always create a new branch for your work.
    ```bash
    git checkout -b feat/my-new-feature
    ```
2.  **Make Changes**: Implement your feature or fix.
3.  **Run Tests**: Ensure that your changes don't break existing functionality.
    ```bash
    pytest
    ```
4.  **Linting**: You can run the pre-commit hooks manually on all files to check for style issues.
    ```bash
    pre-commit run --all-files
    ```

## Coding Standards

### Code Style
We enforce a consistent code style using automated tools:
*   **Black**: The uncompromising code formatter (configured with line-length 79).
*   **Flake8**: For style guide enforcement.
*   **Ruff**: An extremely fast Python linter.

Please ensure your code passes all `pre-commit` checks.

### Docstrings
We use **NumPy-style docstrings**. Every public class and function should have a docstring that describes its purpose, parameters, and return values.

**Example:**
```python
def my_function(param1, param2):
    """
    Brief description of the function.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of the return value.
    """
    return True
```

### Commit Messages
We follow the **Conventional Commits** specification.
*   `feat: ...` for new features
*   `fix: ...` for bug fixes
*   `docs: ...` for documentation updates
*   `style: ...` for formatting changes
*   `refactor: ...` for code refactoring
*   `test: ...` for adding missing tests

## Submitting a Pull Request

1.  **Push your branch** to your fork:
    ```bash
    git push origin feat/my-new-feature
    ```
2.  **Open a Pull Request** against the `main` branch of the `MTFLibrary` repository.
3.  **Description**: detailed description of your changes, referencing any related issues.
4.  **Checklist**:
    *   [ ] Tests passed locally (`pytest`).
    *   [ ] Pre-commit hooks passed.
    *   [ ] Documentation added/updated (if applicable).

## Reporting Issues

If you find a bug or have a feature request, please use the [Issue Tracker](https://github.com/shashi-manikonda/MTFLibrary/issues). Provide as much detail as possible, including steps to reproduce bugs.
