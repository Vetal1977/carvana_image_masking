## Introduction
The repository contains an implementation of the [U-Net model](https://arxiv.org/abs/1505.04597) for the Kaggle competition [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge).  
You need to download train and test images and extract them into _train_ and _test_ directories respectively. Then just start 
```
python main.py
```
to train the model and make predictions. Be aware, this is a long-time process - it took several hours at my GPU-powered PC. It could be optimized though.
