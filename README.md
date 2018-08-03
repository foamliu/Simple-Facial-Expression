# Simple Facial Expression

Bare minimal code to run [Facial Expression Prediction](https://github.com/foamliu/Facial-Expression-Prediction) demo.

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## ImageNet Pretrained Models

Download [ResNet-101](https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294) into imagenet_models folder.

## Demo
Download pre-trained [model](https://github.com/foamliu/Facial-Expression-Prediction/releases/download/v1.0/model.best.hdf5) into "models" folder then run:

```bash
$ python demo.py
```

![image](https://github.com/foamliu/Simple-Facial-Expression/raw/master/images/sample.png)
 
![image](https://github.com/foamliu/Simple-Facial-Expression/raw/master/images/output.png)