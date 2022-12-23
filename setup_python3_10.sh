sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository --yes ppa:deadsnakes/ppa
sudo apt install -y python3.10 python3.10-dev python3.10-venv
python3.10 --version
python3.10 -m venv env
./env/bin/pip install --upgrade pip setuptools wheel
./env/bin/pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
./env/bin/pip install -r requirements.txt
