#!/bin/bash

if [ -d ./test ]; then
	rm test
fi

mkdir test



python pacman.py -n 100 -p SearchAgent -d test -q --layout=capsuleClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=capsuleClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=contestClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=contestClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=mediumClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=mediumClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=minimaxClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=minimaxClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=openClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=openClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=originalClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=originalClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=powerClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=powerClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=smallClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=smallClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=testClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=testClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=trappedClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=trappedClassic
python pacman.py -n 100 -p SearchAgent -d test -q --layout=trickyClassic -a depth=6,alphabeta=True,transposition=True,ordering=False,layout=trickyClassic
