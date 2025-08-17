# Answer Key CV


Freezing env:
```bash
uv pip freeze > requirements.txt
```

Recreating env:
```bash
uv venv

# Linux or macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

uv pip install -r requirements.txt
```