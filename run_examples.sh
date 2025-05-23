#!/bin/bash

DEPTH=3
rm -r logs
mkdir logs
python pacman.py -p SearchAgent -a depth=$DEPTH,alphabeta=False,transposition=False,ordering=False
python pacman.py -p SearchAgent -a depth=$DEPTH,alphabeta=True,transposition=False,ordering=False
python pacman.py -p SearchAgent -a depth=$DEPTH,alphabeta=True,transposition=True,ordering=False
python pacman.py -p SearchAgent -a depth=$DEPTH,alphabeta=True,transposition=False,ordering=True
python pacman.py -p SearchAgent -a depth=$DEPTH,alphabeta=True,transposition=True,ordering=True
