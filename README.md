
## Setup & Execution

**Prerequisites:**
- Python 3.8+
- A C++ compiler (`g++`, `clang`, or MSVC on Windows)

### Step 1: Install.
`python bootstrap.py`

### step 2:
write an api wrapper around llamacpp to use ur newly provisioned llamacpp 
(w/ a programming language where u spend more time writing code than 'building')


## llayout
```
lcpp-binding/
├── models/
├── .gitignore
├── bootstrap.py    <-- 1.
├── download_model.py   
├── pyproject.toml
├── jsonprobe.py    <-- 3.
├── README.md
├── server.py       <-- 2.
└── uv.lock
```