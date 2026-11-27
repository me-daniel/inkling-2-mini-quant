# Amendment 18 data: fresh set A3

Built by `scripts/a18-data.py`; 12 documents x 4,096 tokens per domain (the model's tokenizer.json).

Replacements by rule:

- python: boto/boto3 replaced by pytorch/vision (boto/boto3: fewer than 4096 tokens in eligible files)

## python

`data/battery/A3-python.txt`: 49,163 tokens, sha256 675e5c25b4f479358beb5c94860994722de3f348cd80f110f662ca41d137078b

1. {"source": "github.com/httpie/cli", "commit": "621042a0486c", "files": ["httpie/cli/definition.py"], "license": "BSD-3-Clause"}
2. {"source": "github.com/tornadoweb/tornado", "commit": "711868dddb27", "files": ["tornado/web.py"], "license": "Apache-2.0"}
3. {"source": "github.com/pallets/werkzeug", "commit": "e7ff5d28d0a1", "files": ["src/werkzeug/datastructures.py"], "license": "BSD-3-Clause"}
4. {"source": "github.com/sphinx-doc/sphinx", "commit": "ff202240abeb", "files": ["sphinx/domains/cpp.py"], "license": "BSD-2-Clause"}
5. {"source": "github.com/scrapy/scrapy", "commit": "deaf1fb6cf51", "files": ["scrapy/core/downloader/handlers/http11.py"], "license": "BSD-3-Clause"}
6. {"source": "github.com/celery/celery", "commit": "029262fabe50", "files": ["celery/canvas.py"], "license": "BSD-3-Clause"}
7. {"source": "github.com/streamlit/streamlit", "commit": "1819c568449c", "files": ["lib/streamlit/web/skills.py"], "license": "Apache-2.0"}
8. {"source": "github.com/sympy/sympy", "commit": "e5bace11b60a", "files": ["sympy/polys/zippel.py"], "license": "BSD-3-Clause"}
9. {"source": "github.com/networkx/networkx", "commit": "c220ac4fa9ee", "files": ["networkx/algorithms/community/infomap.py"], "license": "BSD-3-Clause"}
10. {"source": "github.com/mitmproxy/mitmproxy", "commit": "046618526875", "files": ["mitmproxy/contentviews/_api.py", "mitmproxy/contentviews/_utils.py", "mitmproxy/utils/htpasswd.py", "mitmproxy/contentviews/_registry.py", "mitmproxy/contentviews/_compat.py", "mitmproxy/contentviews/_view_dns.py"], "license": "MIT"}
11. {"source": "github.com/pyca/cryptography", "commit": "173d0955f2c7", "files": ["src/cryptography/hazmat/asn1/asn1.py"], "license": "Apache-2.0/BSD-3-Clause"}
12. {"source": "github.com/pytorch/vision", "commit": "4174d08e42fb", "files": ["torchvision/_autograd_registrations.py"], "license": "BSD-3-Clause"}

## javascript

`data/battery/A3-javascript.txt`: 49,163 tokens, sha256 161c49257a209b20af5250b8855a8d9733b595b6728c5be2c762be10e58688bf

1. {"source": "github.com/expressjs/express", "commit": "442fd4679929", "files": ["lib/response.js"], "license": "MIT"}
2. {"source": "github.com/lodash/lodash", "commit": "f299b52f3948", "files": ["lodash.js"], "license": "MIT"}
3. {"source": "github.com/axios/axios", "commit": "abd2cb83a3bc", "files": ["lib/adapters/http.js"], "license": "MIT"}
4. {"source": "github.com/chartjs/Chart.js", "commit": "fd505accedc1", "files": ["src/core/core.scale.js"], "license": "MIT"}
5. {"source": "github.com/mrdoob/three.js", "commit": "2a4451c430b6", "files": ["manual/3rdparty/monaco-editor/min/vs/language/typescript/tsWorker.js"], "license": "MIT"}
6. {"source": "github.com/moment/moment", "commit": "000ac1800e62", "files": ["min/tests.js"], "license": "MIT"}
7. {"source": "github.com/nodejs/node", "commit": "ae8312791da0", "files": ["deps/amaro/lib/wasm.js"], "license": "MIT"}
8. {"source": "github.com/eslint/eslint", "commit": "1429546afcaa", "files": ["lib/rules/preserve-caught-error.js"], "license": "MIT"}
9. {"source": "github.com/facebook/react", "commit": "edae846a9a17", "files": ["flow-typed/environments/node.js"], "license": "MIT"}
10. {"source": "github.com/mozilla/pdf.js", "commit": "e473982703af", "files": ["external/openjpeg/openjpeg_nowasm_fallback.js"], "license": "Apache-2.0"}
11. {"source": "github.com/prettier/prettier", "commit": "da84d5342bf0", "files": ["src/language-markdown/parse/micromark/micromark-extension-html-text.js"], "license": "MIT"}
12. {"source": "github.com/webpack/webpack", "commit": "aebe51c0d176", "files": ["lib/javascript/syntax-printer.js"], "license": "MIT"}

## german

`data/battery/A3-german.txt`: 49,163 tokens, sha256 a7bd72bf6dec9526bab934ce375699601336ce504d9d9ce3ac0b07fb7f909120

1. {"source": "de.wikipedia.org/?curid=14016095", "title": "Geschichte des Weserberglands um Hameln bis zum Mittelalter", "created": "2026-10-30T13:14:31Z", "revid": 272508363, "license": "CC BY-SA 4.0"}
2. {"source": "de.wikipedia.org/?curid=14016182", "title": "Leopard von Rudraprayag", "created": "2026-10-30T15:31:18Z", "revid": 272247305, "license": "CC BY-SA 4.0"}
3. {"source": "de.wikipedia.org/?curid=14016456", "title": "Lavta", "created": "2026-10-30T21:26:42Z", "revid": 272301893, "license": "CC BY-SA 4.0"}
4. {"source": "de.wikipedia.org/?curid=14017171", "title": "Hans-Joachim", "created": "2026-10-31T20:07:39Z", "revid": 272256838, "license": "CC BY-SA 4.0"}
5. {"source": "de.wikipedia.org/?curid=14018268", "title": "Glister", "created": "2026-11-02T06:05:30Z", "revid": 272152403, "license": "CC BY-SA 4.0"}
6. {"source": "de.wikipedia.org/?curid=14018671", "title": "Stortingsgata 16", "created": "2026-11-02T16:23:04Z", "revid": 271989469, "license": "CC BY-SA 4.0"}
7. {"source": "de.wikipedia.org/?curid=14021078", "title": "Russische hybride Kriegsführung", "created": "2026-11-05T15:02:53Z", "revid": 272311574, "license": "CC BY-SA 4.0"}
8. {"source": "de.wikipedia.org/?curid=14023113", "title": "Am Wege (Roman)", "created": "2026-11-07T20:30:00Z", "revid": 272152714, "license": "CC BY-SA 4.0"}
9. {"source": "de.wikipedia.org/?curid=14024632", "title": "Principality of ezidkhan", "created": "2026-11-09T15:27:52Z", "revid": 272153185, "license": "CC BY-SA 4.0"}
10. {"source": "de.wikipedia.org/?curid=14024912", "title": "Einsturz des Hyatt Regency Hotel in Kansas City", "created": "2026-11-09T20:46:01Z", "revid": 272271765, "license": "CC BY-SA 4.0"}
11. {"source": "de.wikipedia.org/?curid=14027073", "title": "Tove Pihl", "created": "2026-11-12T10:38:10Z", "revid": 272269791, "license": "CC BY-SA 4.0"}
12. {"source": "de.wikipedia.org/?curid=14027386", "title": "Sanborns", "created": "2026-11-12T17:53:24Z", "revid": 272147757, "license": "CC BY-SA 4.0"}

## english

`data/battery/A3-english.txt`: 49,163 tokens, sha256 9775a7ae0b94eded6ce9a2dca98224a5a13fc7c1dc9a0e3a4cef27833cfac4e3

1. {"source": "en.wikipedia.org/?curid=84643483", "title": "List of Hazara leaders 1892-1893", "created": "2026-10-29T00:20:17Z", "revid": 1384264633, "license": "CC BY-SA 4.0"}
2. {"source": "en.wikipedia.org/?curid=84643544", "title": "Video games in Puerto Rico", "created": "2026-10-29T00:34:04Z", "revid": 1387592801, "license": "CC BY-SA 4.0"}
3. {"source": "en.wikipedia.org/?curid=84643709", "title": "Amazonas Film Festival", "created": "2026-10-29T01:02:30Z", "revid": 1383334374, "license": "CC BY-SA 4.0"}
4. {"source": "en.wikipedia.org/?curid=84652552", "title": "List of mental disorders in the DSM-5 and DSM-5-TR", "created": "2026-10-30T02:11:11Z", "revid": 1382746813, "license": "CC BY-SA 4.0"}
5. {"source": "en.wikipedia.org/?curid=84653266", "title": "Trams in Wanganui", "created": "2026-10-30T05:37:10Z", "revid": 1387745846, "license": "CC BY-SA 4.0"}
6. {"source": "en.wikipedia.org/?curid=84653267", "title": "October 2026 Indonesian protests", "created": "2026-10-30T05:37:19Z", "revid": 1386724677, "license": "CC BY-SA 4.0"}
7. {"source": "en.wikipedia.org/?curid=84660713", "title": "Recurring elements in the Resident Evil series", "created": "2026-10-31T03:52:12Z", "revid": 1386023625, "license": "CC BY-SA 4.0"}
8. {"source": "en.wikipedia.org/?curid=84660739", "title": "Murder of Luis Vigoreaux", "created": "2026-10-31T04:00:26Z", "revid": 1386969383, "license": "CC BY-SA 4.0"}
9. {"source": "en.wikipedia.org/?curid=84660926", "title": "List of mental disorders in the DSM-III and DSM-III-R", "created": "2026-10-31T05:00:31Z", "revid": 1382955069, "license": "CC BY-SA 4.0"}
10. {"source": "en.wikipedia.org/?curid=84667365", "title": "Mostra do Filme Livre", "created": "2026-11-01T00:54:03Z", "revid": 1384671873, "license": "CC BY-SA 4.0"}
11. {"source": "en.wikipedia.org/?curid=84676207", "title": "Dillie", "created": "2026-11-02T05:00:55Z", "revid": 1383790557, "license": "CC BY-SA 4.0"}
12. {"source": "en.wikipedia.org/?curid=84681519", "title": "Neryosang", "created": "2026-11-02T16:45:01Z", "revid": 1387562424, "license": "CC BY-SA 4.0"}

## math

`data/battery/A3-math.txt`: 49,163 tokens, sha256 592d386cc8fc54e9ede8d9693a6552d3ab73b5d99a6d44c5ebeb822480a9149f

1. {"source": "arxiv.org/abs/2611.01477", "category": "math.DG", "license": "CC BY 4.0"}
2. {"source": "arxiv.org/abs/2611.01871", "category": "math.CO", "license": "CC BY 4.0"}
3. {"source": "arxiv.org/abs/2611.02247", "category": "math.LO", "license": "CC BY 4.0"}
4. {"source": "arxiv.org/abs/2611.04013", "category": "math.AT", "license": "CC BY 4.0"}
5. {"source": "arxiv.org/abs/2611.08208", "category": "math.PR", "license": "CC BY 4.0"}
6. {"source": "arxiv.org/abs/2611.08901", "category": "math.LO", "license": "CC BY 4.0"}
7. {"source": "arxiv.org/abs/2611.09753", "category": "math.NT", "license": "CC BY 4.0"}
8. {"source": "arxiv.org/abs/2611.10788", "category": "math.RA", "license": "CC BY 4.0"}
9. {"source": "arxiv.org/abs/2611.11269", "category": "math.AG", "license": "CC BY 4.0"}
10. {"source": "arxiv.org/abs/2611.12467", "category": "math.NT", "license": "CC BY 4.0"}
11. {"source": "arxiv.org/abs/2611.12680", "category": "math.OC", "license": "CC BY 4.0"}
12. {"source": "arxiv.org/abs/2611.13809", "category": "math.PR", "license": "CC BY 4.0"}

## Overlap with the calibration texts (13-token windows also in calib-mix, calib-mixc or calib-code)

| domain | share |
|---|---|
| python | 0.026% |
| javascript | 0.037% |
| german | 0.000% |
| english | 0.000% |
| math | 0.000% |
