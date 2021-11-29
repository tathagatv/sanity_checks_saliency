
Sanity Checks for Saliency Maps
=====================
This repository provides code to replicate partial experiments of the paper
**Sanity Checks for Saliency Maps** by<br/>
*Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, & Been Kim* and an additionally proposed Model Noise Test.



### Overview


#### Model Randomization Test

For the model randomization test, we randomize the weights of a
model starting from the top layer, successively, all the way to
the bottom layer. This procedure destroys the learned
weights from the top layers to the bottom ones. We compare the resulting explanation from a network with random weights to the one obtained with the model’s original weights. 

#### Model Noise Test

The model noise test is inspired from the model randomization test. The only difference from the randomization test is that instead of randomizing, we add noise to the model weights. In this repository, we specifically add Gaussian noise with zero mean and varying standard deviations. 

### Data

See /doc/data/ for the demo images and the ImageNet image ids used in this
work.  

### Instructions

We have added scripts for training simple MLPs and CNNs on MNIST. To run any of the MNIST notebooks, use these scripts to quickly train either an MLP on MNIST (or Fashion MNIST) or a CNN on MNIST (or Fashion MNIST). The scripts are relatively straight forward. To run the inception v3 notebooks, you will also need to grab pre-trained weights and put them models folder as described in the instructions below.

We use the [saliency python package](https://github.com/pair-code/saliency) to obtain saliency masks. Please see that package for a quick overview. Overall, this replication is mostly for illustration purposes. There are now other packages in PyTorch that provide similar capabilities.

You can use the instructions below to setup an environment with the right dependencies.

```
python -m venv pathtovirtualvenv
source pathtovirtualvenv/bin/activate
pip install -r requirements.txt
```

### Train simple CNNs/MLPs on MNIST/Fashion MNIST
You can train a CNN on MNIST using *src/train_cnn_models.py* as follows:
```
python train_cnn_models.py --data mnist --savemodelpath ../models/ --reg --log
```

You can toggle the data with the --data option. You can also train MLPs with an analogous command:  

```
python train_mlp_models.py --data mnist --savemodelpath ../models/ --reg --log
```

To run the CNN and MLP on MNIST notebooks, you will need to train quick models with the commands above.

### Inception V3 Checkpoint (Important!)
To run any of the incetion_v3 notebooks, you will need inception pretrained weights. These are available from [tensorflow](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz). Alternatively, the weights can be obtained and decompressed as follows:

```
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
```

At the end of this, you should have the file *inception_v3.ckpt* in the folder *models/inceptionv3*. With this, you can run the inception notebooks.


#### Notebooks
In the notebook folder, you will find replication of the key experiments in the paper. Here is a quick summary of the notebooks provided:

- *cnn_mnist_cascading_randomization.ipynb*: shows the cascading randomization on a CNN trained on MNIST.

- *inceptionv3_cascading_randomization.ipynb*: shows the cascading randomization on an Inception v3 model trained on ImageNet for a single bird image. We also show how to compute similarity metrics.

- *mlp_mnist_cascading_noisification.ipynb*: shows the cascading noise addition on a MLP trained on MNIST.


This work was done as the course project for *Machine Learning: Theory and Methods* at IIT Bombay. 