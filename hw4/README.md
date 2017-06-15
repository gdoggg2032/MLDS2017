Reinforcement Learning
====
reinforce policy gradient for open domain single-turn chatbot


## experiments 
compare SELU and tanh activation<br />
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw4/assets/activation_exp.png)
<br />

- [SELU](https://github.com/bioinf-jku/SNNs): Klambauer, Günter, et al. "Self-Normalizing Neural Networks." a  rXiv preprint arXiv:1706.02515 (2017).

compare different models<br />
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw4/assets/leanring_curve.png)
<br />

- notice that there's something wrong with actor-critic model
- now class PolicyActorCritic is actually a policy gradient reinforce model


## Environment
python3 <br />
tensorflow 1.0 <br />
scipy <br />
nltk <br />




## Usage 
1. Download some corpus and format it to a pair of sentence of two lines
```
how are you?
i'm fine
```



2. Simple usage
```
./run.sh [S2S/RL/BEST] [input_file_path] [output_file_path]
```

- S2S: sequence-to-sequence model
- RL: reinforce policy gradient model
- BEST: reinforce policy gradient model

## Train
```
$ python3 chatbot.py --mode 0 --train_file_path [train_file_path]
```

## Model
- single layer LSTM sequence-to-seqeunce model (encoder-decoder model)
- flag "--h" for more usage 

## Test 

```
$ python3 chatbot.py --mode 1 --test_file_path [test_input_file_path] --test_output_path [test_output_file_path] --vocab [vocab_path]
```


## Testing input data format
```
sentence 1
sentence 2
```

## Testing output data format
```
response of sentence 1
response of sentence 2
```









