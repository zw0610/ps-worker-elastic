#!/bin/zsh

BASEDIR=$(dirname "$0")

# ps
python3 "${BASEDIR}"/worker.py --role=none --idx=0 --port 2001 2>&1| tee "${BASEDIR}"/ps_0.log &

python3 "${BASEDIR}"/worker.py --role=none --idx=0 --port 2002 2>&1 | tee "${BASEDIR}"/ps_1.log &

# worker
python3 "${BASEDIR}"/worker.py --role=none --idx=0 --port 2101 2>&1 | tee "${BASEDIR}"/worker_0.log &

python3 "${BASEDIR}"/worker.py --role=none --idx=0 --port 2102 2>&1 | tee "${BASEDIR}"/worker_1.log &

python3 "${BASEDIR}"/worker.py --role=none --idx=0 --port 2103 2>&1 | tee "${BASEDIR}"/worker_2.log &

