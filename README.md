# YOLOv3_OpenCV

An implementation of real-time object detection with a web camera using YOLOv3 and OpenCV.

**Use [this link](https://pjreddie.com/media/files/yolov3.weights)
to download *yolov3.weights* and place the file in the project folder.**

This project was built following the instructions in
[this video tutorial.](https://youtu.be/h56M5iUVgGs)
Slight improvements were made and detailed comments were added to show understanding of the code.

The script will run on the CPU, but [this guide](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) 
can be followed to enable OpenCV's "dnn" module to use an NVIDIA GPU instead, resulting in a significant speed improvement.
