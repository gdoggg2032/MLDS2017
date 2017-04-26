#!/bin/bash
cat ./model/s_model.* > ./model/model-2290.data-00000-of-00001
python3 caption_gen.py --eval 1 --hidden 128 --checkpoint_file="./model/model-2290" --test_id=$1 --test_dir=$2