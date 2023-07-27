#ROAD OBJECT DETECTION - POTHOLE DETECTION
##INTRODUCTION
Road object detection plays a crucial role in various applications. In our project, we specifically focus on the subdomain of pothole detection. Potholes pose a significant challenge for road maintenance and can lead to accidents and vehicle damage. Therefore, efficient and accurate detection of potholes is of utmost importance. To accomplish this, we employ the YOLOv4_tiny algorithm, which stands for You Only Look Once. YOLOv4 is a state-of-the-art real-time object detection algorithm known for its exceptional speed and accuracy. By leveraging YOLOv4_tiny, we aim to detect potholes in road images and videos effectively. Our dataset, sourced from roboflow, provides a diverse collection of annotated images and videos, allowing us to train our model. By employing the YOLOv4_tiny algorithm and utilizing both images and videos, our project aims to contribute to the field of road object detection, specifically in the context of pothole detection.

DATASET
This dataset is a collection of 665 images of roads with the potholes labeled. The dataset was created and shared by undergraduate student as part of his undergraduate thesis and was originally shared on Kaggle. We can access the dataset by using delow link

https://public.roboflow.com/object-detection/pothole/1/download/yolov4scaled
OUR APPROACH
Data Collection:
Gather a dataset consisting of images and videos capturing road scenes with potholes. This dataset should be annotated with bounding boxes around the potholes.

YOLOv4_tiny Model Training:
Train the YOLOv4_tiny object detection model using the annotated dataset. This involves feeding the images and videos into the model and adjusting the model's parameters to optimize pothole detection performance.

Detecting:
Apply the trained YOLOv4_tiny model to new images or videos containing road scenes to detect potholes. The model will analyse the input data and generate bounding boxes around the detected potholes.
