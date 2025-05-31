# pacman

## Play Pacman
### Install

```bash
git clone https://github.com/AhmedBegggaUA/pacman.git
cd pacman
pip install numpy
pip install matplotlib
pip install pandas
pip install torch torchvision torchaudio
```
Or preferably
```bash
git clone https://github.com/Shillianne/pacman.git
cd pacman
sudo apt-get graphviz graphviz-dev swig
pip install -r requirements.txt
```
### Run

```bash
python pacman.py # to play
python pacman.py -p RandomAgent # to play with random agent
python pacman.py -p NeuralAgent # to play with neural agent
python pacman.py --seed=42 -p SearchAgent --layout=mediumClassic -a depth=5,alphabeta=True,transposition=True,ordering=False,layout=mediumClassic # for search agent
python net.py # to train the neural agent
```