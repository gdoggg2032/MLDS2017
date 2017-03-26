Convolution Neural Network
===
This is a Sentence Completition Challenge solved by a RNN-based sequence-to-sequence language model. <br />
Implemented by tensorflow 1.0
<br/>
<br/>
## Requirement
```
python-3.5
tensorflow-1.0
progressbar-3.12.0
pandas-0.19.2
numpy-1.12.0
nltk-3.2.1
``` 

## Usage

Download Microsoft Sentence Completition Challenge Sherlock Holmes novels dataset from [here](https://www.microsoft.com/en-us/research/project/msr-sentence-completion-challenge/) <br/>

Convert all text to a file[raw_train_data_path]

```
$ python3 preprocessing2.py --raw_tr_data [raw_train_data_path] --tr_data [train_data_path] --val_data [val_data_path] --vocab [vocab_file_path] --mode 2
$ python3 rnnlm.py --tr_data [train_data_path] --val_data [val_data_path] --vocab [vocab_file_path] --mode 2 --model [model_path]
```
For more details, use:

```
$ python3 preprocessing2.py --help
$ python3 rnnlm.py --help
```

##Performance

Test: 1040 sentence with missing word and 5 choice words for each.

| Data | Accuarrcy |
| :---: |:---:|
| test | 0.45 |

## Prediction
```
$ python3 preprocessing2.py --raw_te_data [raw_test_data_path] --te_data [test_data_path] --vocab [vocab_file_path] --mode 1
$ python3 rnnlm.py --te_data [test_data_path] --vocab [vocab_file_path] --mode 1 --model [model_path]
```
<br/>
<br/>
<br/>

