# Amendment 15 component A: fresh-domain texts

Built by `scripts/battery-data.py`; 8 documents x 4,096 tokens per domain (the model's tokenizer.json).

## python

`data/battery/A-python.txt`: 32,775 tokens, sha256 1a779d13e87d9e4416c2b85b2b1e053b74917963164f7ec801951e345fcde1e4

1. {"source": "github.com/psf/requests", "commit": "ec553c275fe9", "files": ["requests/models.py"], "license": "Apache-2.0"}
2. {"source": "github.com/pallets/flask", "commit": "74e5263c88e5", "files": ["src/flask/app.py"], "license": "BSD-3-Clause"}
3. {"source": "github.com/pallets/click", "commit": "5e75fe7f9c1b", "files": ["src/click/core.py"], "license": "BSD-3-Clause"}
4. {"source": "github.com/python-attrs/attrs", "commit": "88d2a377487e", "files": ["src/attr/_make.py"], "license": "MIT"}
5. {"source": "github.com/pydantic/pydantic", "commit": "1aaef01805ae", "files": ["release/prepare.py", "release/push.py"], "license": "MIT"}
6. {"source": "github.com/fastapi/fastapi", "commit": "05e6df7e906c", "files": ["fastapi/_compat/v2.py"], "license": "MIT"}
7. {"source": "github.com/Textualize/textual", "commit": "5e5b7ef58b8c", "files": ["src/textual/style.py"], "license": "MIT"}
8. {"source": "github.com/huggingface/huggingface_hub", "commit": "905aee877a2e", "files": ["src/huggingface_hub/_sandbox.py"], "license": "Apache-2.0"}

## rust

`data/battery/A-rust.txt`: 32,776 tokens, sha256 e9b119cc09f6b804249beb80faa39087da719fc9b7762be679c5e01a12e16848

1. {"source": "github.com/BurntSushi/ripgrep", "commit": "61101289fabc", "files": ["crates/printer/src/standard.rs"], "license": "MIT/Unlicense"}
2. {"source": "github.com/serde-rs/serde", "commit": "6b948111ca56", "files": ["serde_derive/src/de.rs"], "license": "MIT/Apache-2.0"}
3. {"source": "github.com/tokio-rs/tokio", "commit": "c6552c5680fa", "files": ["tokio/src/net/windows/named_pipe.rs"], "license": "MIT"}
4. {"source": "github.com/clap-rs/clap", "commit": "ce1634e3b08d", "files": ["src/builder/command.rs"], "license": "MIT/Apache-2.0"}
5. {"source": "github.com/astral-sh/uv", "commit": "264d8b4a5e10", "files": ["crates/uv-environment-operations/src/lib.rs"], "license": "MIT/Apache-2.0"}
6. {"source": "github.com/astral-sh/ruff", "commit": "3c06ae6c84b5", "files": ["crates/ty_python_semantic/src/types/infer/builder.rs"], "license": "MIT"}
7. {"source": "github.com/tokio-rs/axum", "commit": "bfa66864176f", "files": ["axum/src/extract/path/inner_path.rs"], "license": "MIT"}
8. {"source": "github.com/bevyengine/bevy", "commit": "ea0681319821", "files": ["crates/bevy_inspector/src/details_panel.rs"], "license": "MIT/Apache-2.0"}

## german

`data/battery/A-german.txt`: 32,775 tokens, sha256 b38fee0f781e212960db365776b51a8c24c7ddd7fd038fc1073a699b4ab2f4d1

1. {"source": "https://www.bundestag.de/resource/blob/1224896/21101.xml", "protocol": "21/101", "date": "14.10.2026"}
2. {"source": "https://www.bundestag.de/resource/blob/1225180/21102.xml", "protocol": "21/102", "date": "15.10.2026"}
3. {"source": "https://www.bundestag.de/resource/blob/1225450/21103.xml", "protocol": "21/103", "date": "16.10.2026"}
4. {"source": "https://www.bundestag.de/resource/blob/1231278/21104.xml", "protocol": "21/104", "date": "04.11.2026"}
5. {"source": "https://www.bundestag.de/resource/blob/1231576/21105.xml", "protocol": "21/105", "date": "05.11.2026"}
6. {"source": "https://www.bundestag.de/resource/blob/1232358/21106.xml", "protocol": "21/106", "date": "06.11.2026"}
7. {"source": "https://www.bundestag.de/resource/blob/1232962/21107.xml", "protocol": "21/107", "date": "11.11.2026"}
8. {"source": "https://www.bundestag.de/resource/blob/1233202/21108.xml", "protocol": "21/108", "date": "12.11.2026"}

## math

`data/battery/A-math.txt`: 32,776 tokens, sha256 2a994f1b7a8e777cc32edb094079490aa722a3f3d9f1fa08222e4a2211036251

1. {"source": "arxiv.org/abs/2611.00095", "category": "math.GM", "license": "CC BY 4.0"}
2. {"source": "arxiv.org/abs/2611.00101", "category": "math.DG", "license": "CC BY 4.0"}
3. {"source": "arxiv.org/abs/2611.00109", "category": "math.RA", "license": "CC BY 4.0"}
4. {"source": "arxiv.org/abs/2611.00123", "category": "math.PR", "license": "CC BY 4.0"}
5. {"source": "arxiv.org/abs/2611.00125", "category": "math.HO", "license": "CC BY 4.0"}
6. {"source": "arxiv.org/abs/2611.00148", "category": "math.CO", "license": "CC BY 4.0"}
7. {"source": "arxiv.org/abs/2611.00149", "category": "math.CO", "license": "CC BY 4.0"}
8. {"source": "arxiv.org/abs/2611.00217", "category": "math.CO", "license": "CC BY 4.0"}

## Freshness check (share of non-empty lines; code and LaTeX: one pattern match, German: 3+ stopwords)

| file | code | latex | german |
|---|---|---|---|
| calib-mix.txt | 1.86% | 4.76% | 0.26% |
| calib-wt.txt | 0.08% | 1.14% | 0.00% |
