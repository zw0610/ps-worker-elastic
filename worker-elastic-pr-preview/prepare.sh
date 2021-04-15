#!/bin/zsh

BASEDIR=$(dirname "$0")

# ps
python3 ./worker.py --role=ps --idx=0 --port 2001 2>&1| tee ./ps_0.log &

# worker
python3 ./worker.py --role=worker --idx=0 --port 2101 2>&1 | tee ./worker_0.log &

python3 ./worker.py --role=worker --idx=1 --port 2102 2>&1 | tee ./worker_1.log &

python3 ./worker.py --role=worker --idx=2 --port 2103 2>&1 | tee ./worker_2.log &

python3 ./worker.py --role=worker --idx=3 --port 2104 2>&1 | tee ./worker_3.log &

python3 ./worker.py --role=worker --idx=4 --port 2105 2>&1 | tee ./worker_4.log &

python3 ./worker.py --role=worker --idx=5 --port 2106 2>&1 | tee ./worker_5.log &
