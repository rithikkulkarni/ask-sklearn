Linting commands (black/ruff):
```bash
# Check for lint issues (bugs, unused imports, style problems) without changing anything
ruff check .

# Auto-fix what ruff can safely fix
ruff check . --fix

# Check formatting without changing files (this is what CI runs)
black --check .

# Actually reformat files to match black's style
black .
```