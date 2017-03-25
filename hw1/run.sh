#!bin/bash

wget -O model.tgz https://www.dropbox.com/s/p0neayp18nfoajc/model.tgz?dl=0
tar zxvf model.tgz
python3 inference.py --save_dir=./model/  --test_file=$1 --output=$2
