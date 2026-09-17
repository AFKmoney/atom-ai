# Legacy quarantine

HF / GPT-2 tokenizer loaders (`tokenizer.py`, `legacy_gpt2_data.py`, etc.) are
intentionally **not** present in this atom-ai export. Live path uses
`src/io/atomizer.py` only. Do not restore AutoTokenizer / gpt2 on the train or
chat path. See `test/test_no_hf_tokenizer.py`.
