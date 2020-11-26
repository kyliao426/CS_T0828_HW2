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
  - **speed_benchmark.jpg**:It's a screenshot of inference time result.
  ![image](https://github.com/kyliao426/CS_T0828_HW2/blob/main/speed_benchmark.jpg)
  
  ## Brief Introduction
  The major task of this assignment is to detect and recognize the numbers in the image. I use Detectron2 to train the faster R-CNN with ResNet50+FPN backbone. Finally, I got 0.46 on mAP and speed benchmark with 56.6ms. 
  
  ## Methodology
  I used Detectron2 on this task. Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. There are many models can be chosen in Detectron2 Model Zoo, and I chose the faster R-CNN with ResNet+FPN backbone. I also tried to train a model with Retina Net, the mAP is roughly equal to first one, but the speed is much slower. To compromise between precision and speed, I choose the first one for my assignment.
  
  ## Findings and Summary
  First, I didn’t do any data augmentation, because the task is digits detection, and it also needs to recognize the digits. So I believe we can’t rotation too much or even flip it. I thought if we do too much pre-processing, it won’t get a better performance. But later I thought that maybe we can try to do image shear, however I don’t have enough time to train a model again. Though I didn’t do any augmentation, I think there must be some ways to improve performance. And this is what I need to learn.
  
I spent most of my time on building environment. Because detectron2 doesn't support on Microsoft Windows officially, and I spent much of time on solving it. I found that there are many state-of-the-art toolboxes on github, but most of them is only support Linux or macOS. If we want to build them on Windows, we will spend a lot of extra time. I think if I want to continue to study in this field, I need to try to install Linux on my computer. It can help me save a lot of time.
 
There is another thing I want to discuss. Though we can obtain a good result in most cases(like fig.2 and fig.3), in some cases there will be ridiculous error. In Fig.4, the model recognized a street lamp as a number ‘4’. And in fig.5, it recognized a disk as a number ‘0’, but ignore the real number on the disk. 

![image](https://github.com/kyliao426/CS_T0828_HW2/blob/main/example%20image/2.jpg)
![image](https://github.com/kyliao426/CS_T0828_HW2/blob/main/example%20image/3.jpg)
Fig.2 and Fig.3  success examples

![image](https://github.com/kyliao426/CS_T0828_HW2/blob/main/example%20image/4.jpg)

Fig.4 error example (recognize the street lamp as number ‘4’)

![image](https://github.com/kyliao426/CS_T0828_HW2/blob/main/example%20image/5.jpg)

Fig.5 error example (recognize the disk as number ‘0’ but ignore the real number on the disk)

