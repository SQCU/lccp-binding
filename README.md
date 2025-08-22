
## Setup & Execution

**Prerequisites:**
- Python 3.8+
- A C++ compiler (`g++`, `clang`, or MSVC on Windows)

### Step 1: Install.
`python bootstrap.py`

### step 2:
`uv run api_wrapper.py`
`uv run jsonprobe.py`
or
`uv run concurrent_client.py`

## llayout
```
lcpp-binding/
├── models/
├── .gitignore
├── api_wrapper.py      <-- 2.
├── bootstrap.py        <-- 1.
├── concurrent_clientpy <-- 3.
├── download_model.py   
├── pyproject.toml
├── jsonprobe.py        <-- 3.
├── README.md
└── uv.lock
```