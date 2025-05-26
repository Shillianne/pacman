#!/bin/bash

# Usage:
# ./run_examples <DEPTH> <LAYOUT>
# If this doesn't work, try changing permissions on the file or running:
# bash run_examples.sh <DEPTH> <LAYOUT>
#
# Examples:
# ./run_examples 3 mediumClassic

rm -r logs
mkdir logs
python pacman.py -p SearchAgent -q --layout=$2 -a depth=$1,alphabeta=False,transposition=False,ordering=False,layout=$2
python pacman.py -p SearchAgent -q --layout=$2 -a depth=$1,alphabeta=True,transposition=False,ordering=False,layout=$2
python pacman.py -p SearchAgent -q --layout=$2 -a depth=$1,alphabeta=True,transposition=True,ordering=False,layout=$2
python pacman.py -p SearchAgent -q --layout=$2 -a depth=$1,alphabeta=True,transposition=False,ordering=True,layout=$2
python pacman.py -p SearchAgent -q --layout=$2 -a depth=$1,alphabeta=True,transposition=True,ordering=True,layout=$2
