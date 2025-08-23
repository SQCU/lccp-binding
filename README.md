
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

### step 3 (fancy):
`uv run api_wrapper.py`
`navigate to http://127.0.0.1:8080 in ur browser`
`script new behaviors in web client, sample code operating over logits attached for reference` 

### step 4:
```
slap some text in the prototype web client
hit 'probe context slices' to drag the probability mass filter around
look at low-probability items with large differences in log-odds from a full-context query
hit [PROCESS LOGIT] to mutate the logit distribution with this difference
enable [SAMPLE LOGIT] and [AUTOREGRESS] if you want probed context slices 
to continually resample a decoding chain biased by your MARKOV MOMENTUM MULTIPLIER
```


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