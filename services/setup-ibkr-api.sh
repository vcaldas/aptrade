# bin/sh

wget https://interactivebrokers.github.io/downloads/twsapi_macunix.1019.01.zip
unzip twsapi_macunix.1019.01.zip
cd IBJts/source/pythonclient
python3 setup.py install
rm -rf twsapi_macunix.1019.01.zip