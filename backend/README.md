## EchoSync AI Offline Backend

This backend ingests Urdu speech locally, translates it to English, extracts intent using a quantized LLM, and stores resulting meetings, tasks, and alarms without relying on cloud services.

### 1. Environment Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The default database is SQLite (`data.db`). Override with `DATABASE_URL` if you already run PostgreSQL.

### 2. Local Model Layout

Download/convert models into the `backend/models` folder (not tracked by git):

```
backend/models/
├── whisper/               # Faster-Whisper model folder (e.g., medium-int8)
├── translation/           # HuggingFace seq2seq checkpoint (e.g., NLLB distilled)
└── llm/
    └── model.gguf         # Quantized Llama/Qwen/Phi GGUF for llama.cpp
```

Recommended starting points:

| Purpose      | Model                                           | Notes                                             |
|--------------|--------------------------------------------------|---------------------------------------------------|
| ASR          | `whisper-small` → CTranslate2 INT8               | Convert with `ct2-transformers-converter`         |
| Translation  | `facebook/nllb-200-distilled-600M` (offline)     | Cache via `transformers` with `local_files_only`  |
| Intent LLM   | `Meta-Llama-3.1-8B-Instruct` → GGUF Q4_K_M       | Quantize using `llama.cpp` tools                  |

Environment variables:

```
WHISPER_MODEL_PATH=backend/models/whisper
WHISPER_DEVICE=cuda            # or cpu
TRANSLATOR_MODEL_PATH=backend/models/translation
TRANSLATOR_DEVICE=cuda         # optional
LLM_MODEL_PATH=backend/models/llm/model.gguf
LLM_THREADS=8
LLM_GPU_LAYERS=40              # tune per GPU/CPU
```

### 3. Quantization Cheatsheet

- **Whisper (CTranslate2)**  
  ```bash
  ct2-transformers-converter --model openai/whisper-small --output_dir backend/models/whisper --quantization int8
  ```

- **Translation (ONNX + INT8)**  
  ```bash
  optimum-cli onnxruntime quantize \
    -m facebook/nllb-200-distilled-600M \
    --output backend/models/translation --per-channel
  ```

- **LLM (llama.cpp GGUF)**  
  ```bash
  python convert.py --outfile model.gguf /path/to/model
  ./quantize model.gguf model.q4_k_m.gguf q4_k_m
  ```

Copy the resulting folders/files under `backend/models/`.

### 4. Run the API

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Use `/health` to confirm all models are discoverable. `/process_audio` accepts Urdu speech and returns summaries, agenda text, tasks, meetings, and alarms created from that session.

### 5. Testing

- `curl -F "file=@samples/meeting.wav" http://localhost:8000/process_audio`
- `GET /tasks`, `/meetings`, `/alarms` to verify persistence.
- Use the Flutter app (see `frontend/`) to capture speech from an on-device microphone.
