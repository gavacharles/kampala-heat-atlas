#!/bin/zsh
# Remaining pipeline. Stage 1 (Earth Engine) runs in parallel; stage 2 (local analysis) in order.
# Sample batches and downloads are cached, so the script can be re-run safely.
setopt no_nomatch
cd "${0:A:h}/.."
PY=~/.venvs/kampala/bin/python
LOGS=~/kampala-thesis-data/logs; mkdir -p $LOGS   # outside OneDrive: OneDrive breaks long-running log writes
log() { echo "$LOGS/$(basename $1 .py)${2:+_$2}.log"; }
step() { echo "=== $(date +%H:%M) start $*"; if $PY "$@" > "$(log $@)" 2>&1; then echo "ok $*"; else echo "FAILED $*"; tail -4 "$(log $@)"; fi }
have_map() { ls ~/kampala-thesis-data/processed/m1/landcover_$1_*_10m.tif >/dev/null 2>&1 }

# stage 1: Earth Engine (parallel)
for e in t1 t2 t3; do have_map $e || step scripts/07_m1_classify.py $e & done
step scripts/05_export_lst.py &
wait

# stage 2: local analysis (needs stage 1 outputs)
step scripts/08_m2_thermal_classes.py
step scripts/09_m3_informal.py
step scripts/10_m4_trees.py
step scripts/11_m5_solar.py
echo "=== pipeline finished $(date +%H:%M)"
