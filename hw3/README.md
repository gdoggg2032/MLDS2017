Conditional GAN
====
Conditional Generative Adversarial Networks for anime generation (C-AnimeGAN).


Training results dump every epoch for the following tags<br />
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/2390_white_hair_red_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/7949_blue_hair_red_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/9509_brown_hair_orange_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/11288_black_hair_blue_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/11885_blonde_hair_bicolored_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/13838_black_hair_bicolored_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/20146_blonde_hair_brown_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/23144_white_hair_gray_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/28322_gray_hair_orange_eyes.gif)
![image](https://github.com/gdoggg2032/MLDS2017/blob/gdog/hw3/assets/30847_gray_hair_gray_eyes.gif)
<br />

- white hair red eyes
- blue hair red eyes
- brown hair orange eyes
- black hair blue eyes
- blonde hair bicolored eyes
- black hair bicolored eyes
- blonde hair brown eyes
- white hair gray eyes
- gray hair orange eyes
- gray hair gray eyes

## Environment
python3 <br />
tensorflow 1.0 <br />
scipy <br />
skimage <br />

## Data
[source link](https://drive.google.com/open?id=0BwJmB7alR-AvMHEtczZZN0EtdzQ) <br />


## Usage 
1. Download hw3 data from data link, place the MLDS_HW3_dataset/(rename it as data) in the same directory and unzip the face.zip in data/
2. Start training !

## Train
```
$ python3 image_generation.py --mode 2
```

## Model
- dcgan structure for Generator, Discriminator
- model_type: [dcgan, dc-lsgan, dc-wgan, mydcgan], mydcgan designed for conditional label
- use normal(0, 1) as z_sampler
- use one hot encoding for condition tags(only hair and eyes colors)
- all support colors can be found in vocab (dumped with pickle)
- flag "--h" for more usage 

## Test 
This code will automatically dump the results for the tags specified in data/sample_testing_text.txt to the samples/ folder. <br />

## Testing tags format
```
1,<Color> hair <Color> eyes 
2,<Color> hair <Color> eyes
3,<Color> hair <Color> eyes
4,<Color> hair <Color> eyes
.
.
.
```
- Possible colors for eyes
```
['<UNK>', 'yellow', 'gray', 'blue', 'brown', 'red', 'green', 'purple', 'orange',
 'black', 'aqua', 'pink', 'bicolored']
```
- Possible colors for hair
```
['<UNK>', 'gray', 'blue', 'brown', 'red', 'blonde', 'green', 'purple', 'orange',
 'black', 'aqua', 'pink', 'white']
```









