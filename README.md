# CS_T0828_HW2

## Reference from Github
 detectron2: https://github.com/facebookresearch/detectron2
## Environment
 - OS win10
 - Python=3.6.5
 - torch=1.6.0 and torchvision=0.7.0
 - CUDA=10.1
 - CuDNN=7.6
 - GCC=9.2
 
 ## Note
 The uploaded files are without Detectron2. If you wnat to run, you need to build dtectron2 first.
  - **hw2_demo.py**: After you build detectron2, you can run this file to check whether you build it successfully.This is modified from the original detectron2 provided demo.py.
  - **hw2_dataset.py**:This file converts the origin input annotation format(.mat) into a format detectron2 accepted(.json).
  - **hw2_main.py**:This is main file for HW2, it contains training and testing part. Some hyperparameters can be modified here,like epoch.batch size.learning rate etc.
  - **hw2_speed_benchmark.ipynb**:You can run this file on Google Colab. Before you run the file,you should upload the model weights and testing image. And you can see the inference time result.
  - **speed benchmark.jpg**:It's a screenshot of inference time result.
  ![image]https://github.com/kyliao426/CS_T0828_HW2/blob/main/speed_benchmark.jpg
