# Amendment 23 data: development set DV

Built by `scripts/a23-dev-data.py`; 12 documents x 4,096 tokens per domain (the model's tokenizer.json).

Replacements by rule:

- python: urllib3/urllib3 replaced by PyCQA/isort (urllib3/urllib3: fewer than 4096 tokens in eligible files)
- python: pypa/pip-tools replaced by pallets/quart (pypa/pip-tools: fewer than 4096 tokens in eligible files)
- python: pallets/quart replaced by python-hyper/h11 (pallets/quart: fewer than 4096 tokens in eligible files)
- python: python-hyper/h11 replaced by modelcontextprotocol/python-sdk (python-hyper/h11: fewer than 4096 tokens in eligible files)
- javascript: highlightjs/highlight.js replaced by chalk/chalk (highlightjs/highlight.js: fewer than 4096 tokens in eligible files)
- javascript: chalk/chalk replaced by pinojs/pino (chalk/chalk: fewer than 4096 tokens in eligible files)
- javascript: pinojs/pino replaced by expressjs/morgan (pinojs/pino: fewer than 4096 tokens in eligible files)
- javascript: expressjs/morgan replaced by openlayers/openlayers (expressjs/morgan: fewer than 4096 tokens in eligible files)
- javascript: koajs/koa replaced by Automattic/mongoose (koajs/koa: fewer than 4096 tokens in eligible files)
- javascript: yargs/yargs replaced by TryGhost/Ghost (yargs/yargs: fewer than 4096 tokens in eligible files)

## python

`data/battery/DV-python.txt`: 49,162 tokens, sha256 9e50ad3c6bee7d4c5f5fa30983936e212948c8c87ccbbb2b8e5cde4e03c5888d

1. {"source": "github.com/bottlepy/bottle", "commit": "99341ff3791b", "files": ["bottle.py"], "license": "MIT"}
2. {"source": "github.com/pyparsing/pyparsing", "commit": "318ec7e3b945", "files": ["pyparsing_archive.py"], "license": "MIT"}
3. {"source": "github.com/python-jsonschema/jsonschema", "commit": "cb1f8af3487e", "files": ["jsonschema/validators.py"], "license": "MIT"}
4. {"source": "github.com/more-itertools/more-itertools", "commit": "401a2c11a3cd", "files": ["more_itertools/more.py"], "license": "MIT"}
5. {"source": "github.com/pallets/itsdangerous", "commit": "931237a806f4", "files": ["src/itsdangerous/serializer.py", "src/itsdangerous/signer.py"], "license": "BSD-3-Clause"}
6. {"source": "github.com/pallets/markupsafe", "commit": "96dd4afbcba4", "files": ["src/markupsafe/__init__.py", "setup.py", "src/markupsafe/_native.py"], "license": "BSD-3-Clause"}
7. {"source": "github.com/Kludex/uvicorn", "commit": "27ec37f3d35d", "files": ["uvicorn/protocols/websockets/websockets_sansio_impl.py"], "license": "BSD-3-Clause"}
8. {"source": "github.com/agronholm/anyio", "commit": "c81f3462c9f6", "files": ["src/anyio/_core/_tempfile.py"], "license": "MIT"}
9. {"source": "github.com/fastapi/typer", "commit": "6d6383fca42f", "files": ["typer/_click/core.py"], "license": "MIT"}
10. {"source": "github.com/Textualize/rich", "commit": "3a109ccdffe5", "files": ["rich/_unicode_data/unicode17-0-0.py"], "license": "MIT"}
11. {"source": "github.com/PyCQA/isort", "commit": "5b4633e874f4", "files": ["isort/_parse_utils.py", "isort/stdlibs/py314.py"], "license": "MIT"}
12. {"source": "github.com/modelcontextprotocol/python-sdk", "commit": "77001c96f950", "files": ["src/mcp/client/client.py"], "license": "MIT"}

## javascript

`data/battery/DV-javascript.txt`: 49,164 tokens, sha256 466f90d4dba538bb3091b88dace82a43343915f7324a75135e32c4750dfb0bc0

1. {"source": "github.com/jquery/jquery", "commit": "c66d4700dcf9", "files": ["src/selector.js"], "license": "MIT"}
2. {"source": "github.com/jashkenas/underscore", "commit": "d12221366eac", "files": ["underscore-umd.js"], "license": "MIT"}
3. {"source": "github.com/jashkenas/backbone", "commit": "71ce3af790e7", "files": ["backbone.js"], "license": "MIT"}
4. {"source": "github.com/caolan/async", "commit": "086485292d80", "files": ["support/jsdoc/theme/publish.js"], "license": "MIT"}
5. {"source": "github.com/request/request", "commit": "3c0cddc7c8eb", "files": ["request.js"], "license": "Apache-2.0"}
6. {"source": "github.com/handlebars-lang/handlebars.js", "commit": "e6686963ba11", "files": ["spec/helpers.js"], "license": "MIT"}
7. {"source": "github.com/fastify/fastify", "commit": "a94dc370a089", "files": ["lib/log-controller.js"], "license": "MIT"}
8. {"source": "github.com/mochajs/mocha", "commit": "04eb516fe73e", "files": ["lib/cli/parse-args.js", "lib/nodejs/worker-core.js"], "license": "MIT"}
9. {"source": "github.com/openlayers/openlayers", "commit": "6c1db0002cde", "files": ["src/ol/source/GeoZarr.js"], "license": "BSD-2-Clause"}
10. {"source": "github.com/Leaflet/Leaflet", "commit": "3701e81adcc1", "files": ["src/control/FullscreenControl.js", "spec/suites/map/WorldCopyJumpSpec.js"], "license": "BSD-2-Clause"}
11. {"source": "github.com/Automattic/mongoose", "commit": "bcdfae6833d3", "files": ["benchmarks/findOneAndUpdateSimple.js", "lib/schema/union.js"], "license": "MIT"}
12. {"source": "github.com/TryGhost/Ghost", "commit": "cebf02eee815", "files": ["e2e/eslint.config.js", "koenig/kg-unsplash-selector/eslint.config.js", "ghost/core/shared/i18n/index.js", "ghost/core/server/api/endpoints/presence.js", ".github/actions/report-boot-benchmark/calibration.js"], "license": "MIT"}

## german

`data/battery/DV-german.txt`: 49,164 tokens, sha256 2925ab2ce69cdf7005c13b87e251ff458127a758a626ae07c5ed5285f5be350d

1. {"source": "de.wikipedia.org/?curid=14028471", "title": "La Flor", "created": "2026-11-13T19:28:09Z", "revid": 272325033, "license": "CC BY-SA 4.0"}
2. {"source": "de.wikipedia.org/?curid=14028575", "title": "Odenwald bei Hirschhorn", "created": "2026-11-13T21:28:36Z", "revid": 272533447, "license": "CC BY-SA 4.0"}
3. {"source": "de.wikipedia.org/?curid=14029311", "title": "Antifaschistische Wirtschaftspolitik (Buch)", "created": "2026-11-14T21:01:25Z", "revid": 272376343, "license": "CC BY-SA 4.0"}
4. {"source": "de.wikipedia.org/?curid=14029544", "title": "Haplogruppe R-S1194", "created": "2026-11-15T07:34:05Z", "revid": 272514576, "license": "CC BY-SA 4.0"}
5. {"source": "de.wikipedia.org/?curid=14030746", "title": "Antoniterpräzeptorei Alzey", "created": "2026-11-16T13:30:33Z", "revid": 272371878, "license": "CC BY-SA 4.0"}
6. {"source": "de.wikipedia.org/?curid=14031482", "title": "Erstöffnungsschutz", "created": "2026-11-17T11:30:13Z", "revid": 272459167, "license": "CC BY-SA 4.0"}
7. {"source": "de.wikipedia.org/?curid=14032047", "title": "Decennatherium", "created": "2026-11-17T22:17:31Z", "revid": 272344832, "license": "CC BY-SA 4.0"}
8. {"source": "de.wikipedia.org/?curid=14032880", "title": "Lithium-Metall-Anode", "created": "2026-11-18T22:14:00Z", "revid": 272401951, "license": "CC BY-SA 4.0"}
9. {"source": "de.wikipedia.org/?curid=14033556", "title": "Miskal (Panflöte)", "created": "2026-11-19T16:54:56Z", "revid": 272490616, "license": "CC BY-SA 4.0"}
10. {"source": "de.wikipedia.org/?curid=14033658", "title": "Ryomgård-Gjerrild-Grenaa Jernbane", "created": "2026-11-19T18:36:44Z", "revid": 272449584, "license": "CC BY-SA 4.0"}
11. {"source": "de.wikipedia.org/?curid=14035239", "title": "Emili Godes", "created": "2026-11-21T17:04:14Z", "revid": 272359342, "license": "CC BY-SA 4.0"}
12. {"source": "de.wikipedia.org/?curid=14035858", "title": "HAL Light Utillity Helicopter", "created": "2026-11-22T11:51:41Z", "revid": 272430482, "license": "CC BY-SA 4.0"}

## english

`data/battery/DV-english.txt`: 49,160 tokens, sha256 d8a68bc4100c35b7dee5d58967aac131c5744a19be381803a5b1316b3d70489d

1. {"source": "en.wikipedia.org/?curid=84687510", "title": "Guillaume de la Rivière", "created": "2026-11-02T22:35:54Z", "revid": 1383481476, "license": "CC BY-SA 4.0"}
2. {"source": "en.wikipedia.org/?curid=84689338", "title": "2026 New York State Assembly election (districts 1–50)", "created": "2026-11-03T04:59:28Z", "revid": 1387375124, "license": "CC BY-SA 4.0"}
3. {"source": "en.wikipedia.org/?curid=84689340", "title": "2026 New York State Assembly election (districts 51–100)", "created": "2026-11-03T04:59:33Z", "revid": 1386240837, "license": "CC BY-SA 4.0"}
4. {"source": "en.wikipedia.org/?curid=84689341", "title": "2026 New York State Assembly election (districts 101–150)", "created": "2026-11-03T04:59:41Z", "revid": 1386240823, "license": "CC BY-SA 4.0"}
5. {"source": "en.wikipedia.org/?curid=84692122", "title": "Timeline of the 2024-2026 Serbian anti-corruption protests (January–June 2025)", "created": "2026-11-03T15:33:21Z", "revid": 1387292614, "license": "CC BY-SA 4.0"}
6. {"source": "en.wikipedia.org/?curid=84695850", "title": "Amplification of antiquity", "created": "2026-11-04T01:14:20Z", "revid": 1387103299, "license": "CC BY-SA 4.0"}
7. {"source": "en.wikipedia.org/?curid=84702401", "title": "Magruder murders", "created": "2026-11-04T21:31:07Z", "revid": 1385212110, "license": "CC BY-SA 4.0"}
8. {"source": "en.wikipedia.org/?curid=84714267", "title": "Papacito", "created": "2026-11-06T02:58:49Z", "revid": 1387363595, "license": "CC BY-SA 4.0"}
9. {"source": "en.wikipedia.org/?curid=84717657", "title": "Germany and the Russo-Ukrainian war", "created": "2026-11-06T16:49:51Z", "revid": 1386162884, "license": "CC BY-SA 4.0"}
10. {"source": "en.wikipedia.org/?curid=84717673", "title": "Reorganization of the Brazilian Armed Forces during the military dictatorship", "created": "2026-11-06T16:51:33Z", "revid": 1384237651, "license": "CC BY-SA 4.0"}
11. {"source": "en.wikipedia.org/?curid=84722406", "title": "Poland and the Russo-Ukrainian war", "created": "2026-11-07T06:48:13Z", "revid": 1387426727, "license": "CC BY-SA 4.0"}
12. {"source": "en.wikipedia.org/?curid=84727491", "title": "Bust of Girolamo Benivieni", "created": "2026-11-07T23:06:43Z", "revid": 1384372765, "license": "CC BY-SA 4.0"}

## math

`data/battery/DV-math.txt`: 49,164 tokens, sha256 20e4591e175ecb28833684a875dac37675f1f899db23a486a9a77ed46e53347d

1. {"source": "arxiv.org/abs/2611.14490", "category": "math.RA", "license": "CC BY 4.0"}
2. {"source": "arxiv.org/abs/2611.14523", "category": "math.RT", "license": "CC BY 4.0"}
3. {"source": "arxiv.org/abs/2611.14543", "category": "math.RA", "license": "CC BY 4.0"}
4. {"source": "arxiv.org/abs/2611.14546", "category": "math.CO", "license": "CC BY 4.0"}
5. {"source": "arxiv.org/abs/2611.14547", "category": "math.RA", "license": "CC BY 4.0"}
6. {"source": "arxiv.org/abs/2611.14552", "category": "math.LO", "license": "CC BY 4.0"}
7. {"source": "arxiv.org/abs/2611.14555", "category": "math.CV", "license": "CC BY 4.0"}
8. {"source": "arxiv.org/abs/2611.14566", "category": "math.GR", "license": "CC BY 4.0"}
9. {"source": "arxiv.org/abs/2611.14579", "category": "math.NT", "license": "CC BY 4.0"}
10. {"source": "arxiv.org/abs/2611.14645", "category": "math.ST", "license": "CC BY 4.0"}
11. {"source": "arxiv.org/abs/2611.14667", "category": "math.CO", "license": "CC BY 4.0"}
12. {"source": "arxiv.org/abs/2611.14689", "category": "math.AG", "license": "CC BY 4.0"}

## Overlap (share of 13-token windows also in the calibration texts / in A3)

| domain | calibration | A3 |
|---|---|---|
| python | 0.069% | 0.028% |
| javascript | 0.002% | 0.032% |
| german | 0.000% | 0.000% |
| english | 0.000% | 0.000% |
| math | 0.010% | 0.067% |
