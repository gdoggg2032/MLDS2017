TEST_TEXT=$1

unzip model.zip
python3 image_generation.py\
 --mode 1 --batch_size 32\
  --z_type normal --log model\
   --seed 123 --vocab ./vocab\
    --model_type dclsgan --test_text $TEST_TEXT