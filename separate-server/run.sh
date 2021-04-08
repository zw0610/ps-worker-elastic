#!/bin/zsh

BASEDIR=$(dirname "$0")

"${BASEDIR}"/prepare.sh

python3 "${BASEDIR}"/coordinator.py
