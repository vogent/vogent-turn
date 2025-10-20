# UV Development Guide

This project is set up to use [UV](https://github.com/astral-sh/uv), a fast Python package manager.

## Installation

### Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew
brew install uv
```

## Development Workflow

### 1. Set Up Environment

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# Or: .venv\Scripts\activate  # Windows
```

### 2. Install Package

```bash
# Install in editable mode with all dependencies
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Test import
python -c "from vogent_turn import TurnDetector; print('Success!')"

# Test CLI tool
vogent-turn-predict --help
```

## Common Tasks

### Adding Dependencies

```bash
# Add a runtime dependency
# 1. Edit pyproject.toml under [project.dependencies]
# 2. Reinstall: uv pip install -e .

# Add a dev dependency
# 1. Edit pyproject.toml under [project.optional-dependencies.dev]
# 2. Install: uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Install dev dependencies first
uv pip install -e ".[dev]"

# Run tests
pytest
```

### Code Formatting

```bash
# Install ruff
uv pip install -e ".[dev]"

# Format code
ruff format .

# Lint code
ruff check .
```

### Building for Distribution

```bash
# Install build tools
uv pip install build twine

# Build distributions
python -m build
# Or: uv build

# Check the build
ls dist/
# Should see: vogent-turn-0.1.0.tar.gz and vogent_turn-0.1.0-py3-none-any.whl
```

### Publishing to PyPI

```bash
# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    vogent-turn

# Upload to main PyPI
twine upload dist/*
```

## Why UV?

- **10-100x faster** than pip for package installation
- **Better dependency resolution** than pip
- **Modern standard** using `pyproject.toml` (PEP 621)
- **Single tool** for environment management and package installation

## Comparison with Traditional Workflow

### Traditional (pip)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### With UV
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

The UV approach is significantly faster and more reliable.

## Tips

1. **Activating environment**: Always run `source .venv/bin/activate` before development
2. **Deactivating**: Run `deactivate` to exit the virtual environment
3. **Cleaning**: Remove `.venv/` and `uv.lock` to start fresh
4. **Lock files**: UV can generate lock files for reproducible builds (optional)

## Troubleshooting

### "Command not found: uv"
Make sure UV is in your PATH. Try restarting your shell or run:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### "Cannot find package vogent_turn"
Make sure you've installed in editable mode:
```bash
uv pip install -e .
```

### Dependencies not resolving
Try clearing the cache:
```bash
rm -rf .venv
uv venv
uv pip install -e .
```

