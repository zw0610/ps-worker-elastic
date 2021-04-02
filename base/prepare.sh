#!/bin/zsh

BASEDIR=$(dirname "$0")

PYTHONPATH=${PYTHONPATH}:"${BASEDIR}" python "${BASEDIR}"/ps.py --role=ps --idx=0 2>&1 | tee "${BASEDIR}"/ps_0.log &

PYTHONPATH=${PYTHONPATH}:"${BASEDIR}" python "${BASEDIR}"/ps.py --role=ps --idx=1 2>&1 | tee "${BASEDIR}"/ps_1.log &

PYTHONPATH=${PYTHONPATH}:"${BASEDIR}" python "${BASEDIR}"/worker.py --role=worker --idx=0 2>&1 | tee "${BASEDIR}"/worker_0.log &

PYTHONPATH=${PYTHONPATH}:"${BASEDIR}" python "${BASEDIR}"/worker.py --role=worker --idx=1 2>&1 | tee "${BASEDIR}"/worker_1.log &

PYTHONPATH=${PYTHONPATH}:"${BASEDIR}" python "${BASEDIR}"/worker.py --role=worker --idx=2 2>&1 | tee "${BASEDIR}"/worker_2.log &

