#!/bin/zsh

BASEDIR=$(dirname "$0")

"${BASEDIR}"/prepare.sh

PYTHONPATH=${PYTHONPATH}:"${BASEDIR}" python "${BASEDIR}"/coordinator.py
