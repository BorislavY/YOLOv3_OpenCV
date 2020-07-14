# YOLOv3_OpenCV

An implementation of real-time object detection using YOLOv3 and OpenCV. 
The script can work either with the web camera or with a video file.

**Use [this link](https://pjreddie.com/media/files/yolov3.weights)
to download *yolov3.weights* and place the file in the project folder.**

This project is based on
[this video tutorial.](https://youtu.be/h56M5iUVgGs)
Some improvements were made and detailed comments were added to show understanding of the code.

The script will run on the CPU, but [this guide](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) 
can be followed to enable OpenCV's "dnn" module to use an NVIDIA GPU instead, resulting in a significant speed improvement.

### Example of a processed video file:
![GIF missing :(](results.gif)