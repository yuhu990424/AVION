# Install

Docker is intentionally not used.

Use the existing conda environment named `AVION`:

```bash
conda activate AVION
pip install -r requirements.txt
python verify_all.py --stage environment
```

Do not store API keys in configuration files. For Gemini generation, export:

```bash
export GEMINI_API_KEY=...
```

