This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

Preliminary submission to check behavior on Carla. (It is already working pretty well in simulator and test images.)

This README is not completed yet. Description will be coming soon.

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Description

In most of the nodes except tl_detector, I mostly followed how it is implemented in the video.

I will describe how I implemented tl_detector, especially how to classify traffic lights color and use the result to the vehicle movement.

One thing I changed from the reference implentation is to avoid using map to decide the place to stop because I think the car should not rely heavily on map and should use the result of recognition as much as possible.

Therefore, a new parameter, "remoteness" which means how far (small) the traffic signal sees in the camera image, is introduced and the car tries to decelerate if RED or YELLOW signals are coming and stop if it is close (big) in the images.

#### Traffic Light recognition in Simulator

For simulator environment, I noticed that simple template matching was sufficient because the shape and color of traffic signals never changes.

Therefore, I prepared a template image for each color as below and use it for [cv2.matchTemplate()](https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be).

![template_red](https://raw.githubusercontent.com/ngard/CarND-Capstone/master/ros/src/tl_detector/templates/simulator/red.bmp)

![template_red](https://raw.githubusercontent.com/ngard/CarND-Capstone/master/ros/src/tl_detector/templates/simulator/yellow.bmp)

![template_red](https://raw.githubusercontent.com/ngard/CarND-Capstone/master/ros/src/tl_detector/templates/simulator/green.bmp)

Actually, I implemented this just for debugging other nodes, however,
even with this very simple implementation, it almost never fails to recognize, surprisingly,
and I decided to keep it. That also allowed me to avoid making dataset.

I wanted to use [cv::cuda::TemplateMatching](https://docs.opencv.org/3.3.1/d2/d58/classcv_1_1cuda_1_1TemplateMatching.html) to utilize GPU, however, it seems like OpenCV installed in Carla is not built with CUDA and could not use it.

#### Traffic Light recognition in Test Site

However, in real world, the thing is not so easy, I tried to finish this project just using template matcing but the result was so poor.

Therefore, I decided to use deep learning based technique.

YOLO and SSD are two majoy architectures to realize real-time object detection in the world. This time, I decided to use normal SSD mainly because it requires less training dataset than YOLO or other similar ones if using pretrained VGG16 weights as we only have ~2000 images which is extracted from bags file provided.

I extracted images from the bags and annotate them with VOTT annotation tool.

Then customize [ssd_keras](https://github.com/rykov8/ssd_keras) to use my dataset.
My dataset and customized jupyter notebook to train SSD will be uploaded soon.

Surprisingly, only 2 epochs were required to train model properly thanks to pretrained VGG16 weights and the trained model performed surprisingly well to recognize traffic signals in the test images.