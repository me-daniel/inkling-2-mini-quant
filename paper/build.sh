#!/bin/sh
# Build the paper: figures from results/pilot/, the PDF with tectonic, paper.md from the same source, then the number check.
#   sh paper/build.sh [LOGS_DIR]     (LOGS_DIR: the run logs, default logs/; not in git)
set -eu
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
"$PY" -I figures.py
tectonic -X compile --keep-intermediates main.tex
"$PY" -I tex2md.py
"$PY" -I check-numbers.py --logs "${1:-../logs}"
rm -f main.aux main.log main.out main.blg
