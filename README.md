
## Setup & Execution

**Prerequisites:**
- Python 3.8+
- A C++ compiler (`g++`, `clang`, or MSVC on Windows)

### Step 1: Install `uv`

python bootstrap.py

### step 2:
uv run python server.py
uv run jsonprobe.py --logits

## llayout
llama_server_template/
├── models/
├── .gitignore
├── bootstrap.py    <-- 1.
├── download_model.py   
├── pyproject.toml
├── jsonprobe.py    <-- 3.
├── README.md
├── server.py       <-- 2.
└── uv.lock

