COMPLETED

# PR-001: Project Setup

## Dependencies
None (this is a foundation PR)

## Overview
Initialize the Python project with proper structure, dependencies, and development tooling.

## Tech Choices
- **Python Version:** 3.10+
- **Package Manager:** pip with `pyproject.toml`
- **Linting:** ruff
- **Formatting:** black
- **Type Checking:** mypy

## Tasks

### 1. Create project structure
```
poke/
├── pyproject.toml
├── README.md
├── src/
│   └── poke/
│       ├── __init__.py
│       ├── data/           # Data processing
│       ├── agents/         # Agent implementations
│       ├── models/         # Neural network models
│       ├── training/       # Training loops
│       ├── evaluation/     # Evaluation harness
│       └── utils/          # Shared utilities
├── tests/
├── scripts/
├── configs/
└── teams/
    └── gen9ou/
        └── v1/
```

### 2. Create `pyproject.toml`
```toml
[project]
name = "poke"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "poke-env>=0.7",
    "pandas>=2.0",
    "pydantic>=2.0",
    "tqdm",
    "requests",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff",
    "black",
    "mypy",
]
tracking = [
    "wandb",
]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W"]

[tool.black]
line-length = 100

[tool.mypy]
python_version = "3.10"
warn_return_any = true
```

### 3. Create initial `__init__.py` files
Add empty `__init__.py` to all package directories.

### 4. Create `.gitignore`
```
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.mypy_cache/
.ruff_cache/
wandb/
data/raw/
data/processed/
*.pt
*.pth
checkpoints/
```

### 5. Set up virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 6. Verify installation
```bash
python -c "import poke; print('OK')"
pytest --version
ruff --version
```

## Acceptance Criteria
- [ ] All directories created
- [ ] `pip install -e .` succeeds
- [ ] `import poke` works
- [ ] Linting tools run without error on empty project
- [ ] `.gitignore` prevents committing generated files

## Estimated Complexity
Low - Standard Python project setup
