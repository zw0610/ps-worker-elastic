#!/bin/zsh

BASEDIR=$(dirname "$0")

python3 "${BASEDIR}"/coordinator.py 2>&1 | tee "${BASEDIR}"/coordinator.log