Convolution Neural Network
===
This is a MNIST number recognition problem solved by a convolution neural network model. <br />
Implement by tensorflow 0.11
![MNIST](https://github.com/m516825/CNN-MNIST/blob/master/img.png)
<br/>
<br/>
## Requirement
```
python2.7
tensorflow 0.11
progressbar
numpy
``` 

## Usage

Download MNIST training data and its labels from [here](http://yann.lecun.com/exdb/mnist/) <br/>

train-images-idx3-ubyte.gz:  training set images (9912422 bytes) <br/>
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) <br/>


```
$ python cnn_mnist.py --tr_data [train_data_path] --tr_labels [train_labels_path]
```

##Performance

Train/Dev: 58000/2000

| Data | Accuarrcy |
| :---: |:---:|
| train | 0.995 |
| dev | 0.993 |

## Prediction
```
$ python cnn_mnist.py --te_data [test_data_path] --mode 1 --model [model_path]
```
<br/>
<br/>
<br/>
TODO: image distortion
