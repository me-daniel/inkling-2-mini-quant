# Amendment 16 data: code calibration and fresh set A2

Built by `scripts/a16-data.py`; every document is 4,096 tokens of the model's tokenizer.json.

## calib-code

`data/calib-code.txt`: 65,545 tokens, sha256 4d5c08b73cfabf74ce15634bbc9d477309304fec86d9044bcde7cbbbf766536b

1. {"source": "github.com/django/django", "commit": "c179ad9fe7e8", "files": ["django/db/models/sql/query.py"], "license": "BSD-3-Clause"}
2. {"source": "github.com/numpy/numpy", "commit": "922c47721324", "files": ["numpy/ma/core.py"], "license": "BSD-3-Clause"}
3. {"source": "github.com/scikit-learn/scikit-learn", "commit": "96a0bc861ad2", "files": ["sklearn/utils/estimator_checks.py"], "license": "BSD-3-Clause"}
4. {"source": "github.com/psf/black", "commit": "37542e64855c", "files": ["profiling/mix_huge.py"], "license": "MIT"}
5. {"source": "github.com/pytest-dev/pytest", "commit": "326ae0cd88f5", "files": ["testing/python/fixtures.py"], "license": "MIT"}
6. {"source": "github.com/sqlalchemy/sqlalchemy", "commit": "f74de7bc0b88", "files": ["lib/sqlalchemy/sql/compiler.py"], "license": "MIT"}
7. {"source": "github.com/encode/httpx", "commit": "4cbf13ece2e5", "files": ["httpx/_client.py"], "license": "BSD-3-Clause"}
8. {"source": "github.com/pallets/jinja", "commit": "45c23ea56dbb", "files": ["src/jinja2/compiler.py"], "license": "BSD-3-Clause"}
9. {"source": "github.com/rust-lang/cargo", "commit": "dd9900585a6c", "files": ["src/cargo/util/toml/mod.rs"], "license": "MIT/Apache-2.0"}
10. {"source": "github.com/rayon-rs/rayon", "commit": "496b931b1b4f", "files": ["src/iter/mod.rs"], "license": "MIT/Apache-2.0"}
11. {"source": "github.com/hyperium/hyper", "commit": "bb609b227236", "files": ["src/proto/h1/role.rs"], "license": "MIT"}
12. {"source": "github.com/sharkdp/fd", "commit": "686318c00509", "files": ["src/cli.rs"], "license": "MIT/Apache-2.0"}
13. {"source": "github.com/alacritty/alacritty", "commit": "2bd26fbeb0c5", "files": ["alacritty_terminal/src/term/mod.rs"], "license": "Apache-2.0"}
14. {"source": "github.com/nushell/nushell", "commit": "614bc2a94332", "files": ["crates/nu-parser/src/parser.rs"], "license": "MIT"}
15. {"source": "github.com/tauri-apps/tauri", "commit": "ce76d95ab186", "files": ["core/tauri-utils/src/config.rs"], "license": "MIT/Apache-2.0"}
16. {"source": "github.com/rust-lang/rust-analyzer", "commit": "34bc240e9405", "files": ["crates/hir/src/lib.rs"], "license": "MIT/Apache-2.0"}

## python

`data/battery/A2-python.txt`: 32,775 tokens, sha256 f1c58a292cb9e2ad88347a7f3b80131d7f23d52b06bad88383748535a0a2598f

1. {"source": "github.com/tqdm/tqdm", "commit": "6791e8c5b3d6", "files": ["tqdm/std.py"], "license": "MIT/MPL-2.0"}
2. {"source": "github.com/Delgan/loguru", "commit": "6196ab6debaa", "files": ["loguru/_logger.py"], "license": "MIT"}
3. {"source": "github.com/marshmallow-code/marshmallow", "commit": "0b0bdd36ff1f", "files": ["src/marshmallow/fields.py"], "license": "MIT"}
4. {"source": "github.com/python-trio/trio", "commit": "f37d2bd67b94", "files": ["trio/_core/_run.py"], "license": "MIT/Apache-2.0"}
5. {"source": "github.com/pypa/pip", "commit": "4b71f36fb4df", "files": ["src/pip/_vendor/packaging/ranges.py"], "license": "MIT"}
6. {"source": "github.com/python-poetry/poetry", "commit": "3cc663a73852", "files": ["src/poetry/installation/uninstaller.py"], "license": "MIT"}
7. {"source": "github.com/aio-libs/aiohttp", "commit": "dc37bb0a964f", "files": ["aiohttp/client_middleware_digest_auth.py"], "license": "Apache-2.0"}
8. {"source": "github.com/pandas-dev/pandas", "commit": "d68ece3856df", "files": ["pandas/io/sql/api.py"], "license": "BSD-3-Clause"}

## rust

`data/battery/A2-rust.txt`: 32,774 tokens, sha256 f7810a90568132500eb1d74719a1ef422167ccfa67fc31884f31b781349a880a

1. {"source": "github.com/rust-random/rand", "commit": "e97b5b68877a", "files": ["src/distributions/uniform.rs"], "license": "MIT/Apache-2.0"}
2. {"source": "github.com/dtolnay/anyhow", "commit": "0a45d7665c32", "files": ["src/ensure.rs"], "license": "MIT/Apache-2.0"}
3. {"source": "github.com/crossbeam-rs/crossbeam", "commit": "960c08abad54", "files": ["crossbeam-deque/src/deque.rs"], "license": "MIT/Apache-2.0"}
4. {"source": "github.com/image-rs/image", "commit": "b34c9c979400", "files": ["src/codecs/webp/vp8.rs"], "license": "MIT/Apache-2.0"}
5. {"source": "github.com/denoland/deno", "commit": "d8c551b5a5b4", "files": ["libs/npm/resolution/graph.rs"], "license": "MIT"}
6. {"source": "github.com/biomejs/biome", "commit": "d965df8d6920", "files": ["crates/biome_css_analyze/src/baseline_data.rs"], "license": "MIT/Apache-2.0"}
7. {"source": "github.com/gfx-rs/wgpu", "commit": "1edb79d1bdc1", "files": ["naga/src/back/glsl/writer.rs"], "license": "MIT/Apache-2.0"}
8. {"source": "github.com/rust-lang/rustfmt", "commit": "90b9723de5cc", "files": ["ci/src/integration.rs", "src/header.rs", "src/range.rs", "ci/src/common.rs"], "license": "MIT/Apache-2.0"}

## go

`data/battery/A2-go.txt`: 32,774 tokens, sha256 e254d7a8e1fa68ec94036e9cae1cdd4c54c2c308c8b6e531c5c732caf7cb66d9

1. {"source": "github.com/spf13/cobra", "commit": "bf11ab6321f2", "files": ["command.go"], "license": "Apache-2.0"}
2. {"source": "github.com/gin-gonic/gin", "commit": "8659ab573cf7", "files": ["context.go"], "license": "MIT"}
3. {"source": "github.com/sirupsen/logrus", "commit": "54357fb25f10", "files": ["entry.go", "logger.go"], "license": "MIT"}
4. {"source": "github.com/gorilla/mux", "commit": "eb99d7a67714", "files": ["route.go"], "license": "BSD-3-Clause"}
5. {"source": "github.com/prometheus/prometheus", "commit": "a5022ef622b4", "files": ["web/api/v1/openapi_schemas.go"], "license": "Apache-2.0"}
6. {"source": "github.com/caddyserver/caddy", "commit": "edcdfa0ed88f", "files": ["modules/caddytls/ech.go"], "license": "Apache-2.0"}
7. {"source": "github.com/junegunn/fzf", "commit": "9a0705815cc6", "files": ["src/algo/runeindex_x86.go", "src/zellij.go", "src/algo/runeindex_ref.go", "src/util/concurrent_set.go", "src/algo/runeindex_others.go", "src/algo/indexbyte2_other.go", "src/algo/indexbyte2_amd64.go"], "license": "MIT"}
8. {"source": "github.com/cli/cli", "commit": "761ea6d6915b", "files": ["pkg/cmd/skills/install/install.go"], "license": "MIT"}

## typescript

`data/battery/A2-typescript.txt`: 32,775 tokens, sha256 d8e2fdb2b048136943235c36df5ed7043e981edbe1353c2c46bf80181907a5e1

1. {"source": "github.com/date-fns/date-fns", "commit": "fadbd4eb7920", "files": ["src/parse/test.ts"], "license": "MIT"}
2. {"source": "github.com/colinhacks/zod", "commit": "6252c757f897", "files": ["deno/lib/types.ts"], "license": "MIT"}
3. {"source": "github.com/trpc/trpc", "commit": "1bc95c837dfb", "files": ["www/src/components/TwitterWall/script.output.ts"], "license": "MIT"}
4. {"source": "github.com/vitejs/vite", "commit": "f12a1ab9fe6d", "files": ["packages/vite/src/node/plugins/css.ts"], "license": "MIT"}
5. {"source": "github.com/excalidraw/excalidraw", "commit": "9c2e462ec23e", "files": ["packages/element/src/bucketFill.ts"], "license": "MIT"}
6. {"source": "github.com/withastro/astro", "commit": "28a925e016fa", "files": ["packages/astro/src/core/fetch/fetch-state.ts"], "license": "MIT"}
7. {"source": "github.com/honojs/hono", "commit": "9075031ea465", "files": ["adapters/aws-lambda/src/handler.ts"], "license": "MIT"}
8. {"source": "github.com/drizzle-team/drizzle-orm", "commit": "3bcb6a27ec1f", "files": ["drizzle-kit/src/serializer/gelSerializer.ts"], "license": "Apache-2.0"}

## Overlap with the calibration texts (share of 13-token windows also in calib-mix or calib-code)

| domain | share |
|---|---|
| python | 0.097% |
| rust | 0.169% |
| go | 0.000% |
| typescript | 0.000% |
