# Objective: 
Detection of a electrical panel door using a Intel Realsense camera and Opencv and Scikit learn libraries. 

Project is focused towards incorporating this technology into a drone for assisting it to open the door in air autonomously. 
Drones are being used extensively for surveillance and reconnaissance tasks.
But in the recent past they are also being employed for interacting physically with the environment.
This can involve tasks like picking up packages or boxes, mounting some sensor on a wall or opening doors while hovering.
Several sophisticated end effectors have also been designed for this purpose. 
But for all of them to work the drone needs to identify the target object in front of it.
So the focus of this project is to identify an electrical panel door and its door handle and measure the distance of the drone from it.
This information will be later used (in a separate project) by the drone for controlling its position and the movement of its end effector to grab the door handle and pull open the door.

This project is only about the image processing part to identify and electrical panel door. 
The camera to be used should be small and light, so that it can be put on a drone. So we selected the Intel Realsense Depth camera for this purpose.

# Requirements: 
* Algorithm should be able to run in real time on a laptop as well as on a sigle board computer (without any dependence on GPUs).
* Algorithm should be able to detect the electrical panel door and show the position of the door handle.
* The distance of the door handle from the camera should also be calculated continuously in real time.
* All software should be open source. 
* Overall setup should be battery operated and should be small and light enough to be mounted on a drone. 

# Current Framework: 
* Opencv, SciKit Learn and SciKit Image libraries, Ubuntu 16.04. 
* Intel Realsense R200 Depth Camera.
* Odroid XU4 single board computer, Laptop computer.

#### Intel Realsense R200 Depth Camera and Odroid XU4:
![intel_realsense](images/intel_realsense_r200.png)
![Odroid_XU4](images/odroid_XU4.jpg)

#### Overall Setup mounted on the test Drone:
**[ Odroid is inside the white case ]**

![setup_on_drone_1_marked](images/setup_on_drone_1_marked.jpg)
![setup_on_drone_2_marked](images/setup_on_drone_2_marked.jpg)
![setup_on_drone_3](images/setup_on_drone_3.jpg)
![setup_on_drone_4](images/setup_on_drone_4.jpg)

The next figure shows the **electrical panel door** mounted on a dummy wall in the lab. 
The figure also shows the **yellow claws with fingers** to grab the door handle. 
This will be later attached to the drone. For now it is only mounted on a stand so that the overall setup looks like a real image as seen by the realsense camera.
The claws will be visible from one side of the frame, as it is supposed to be mounted on one arm of the drone.

![original_rgb_frame](images/original_rgb_frame.png)

# Algorithm Description: 
The algorithm goes through several stages for detecting the box.

### Stage 1:
The videos from the realsense is read as a numpy array. This included both the **rgb frame** as well as the depth frame.
The **depth frame** is of the same dimension as the rgb frame (640 x 480 pixels) but each pixel in the frame has a value equal to the distance of the object represented by that pixel from the camera in mm.

### Stage 2:
When the box is nearer to the camera, the claws will be obstructing parts of the box. But it is already known at which pixels the claws are visible.
So the pixels for the claws are replaced by the other surrounding background pixels to that the overall frame has no visible claws and only the box and the background are visible. 
The following figure shows the original frame and the processed frame with the claws removed. This frame will be used for further processing. This frame will be referred to as **modified rgb frame** from now onwards.

![original_rgb_frame](images/original_rgb_frame.png)
![modified_rgb_frame](images/modified_rgb_frame.png)

### Stage 3:
The modified rgb frames from stage 2, (claws removed) are then subjected to edge detection and all the contours from these edges are found out. This **contour frame** is shown below.
These contours are then filtered based on their size and area so that most of the unwanted contours get removed. This will be called **filtered contour frame**.
This filtered contour frame is also shown which includes the contour of the box and some other similar contours. These contours are then drawn as filled blocks in a new frame to create a mask.
This will be called **contour mask**.

![modified_rgb_frame](images/modified_rgb_frame.png)
![contour_frame](images/contour_frame.png)
![filtered_contour_frame](images/filtered_contour_frame.png)
![contour_mask](images/contour_mask.png)

However, the size of this box contour will not stay fixed. It will change depending on the distance of the box from the camera.
So the size and area filters applied to the contour frame cannot have a fixed threshold, but has to be a range.

### Stage 4:
After this, the modified rgb frame is filtered with a color filter, so that only the objects which are within a range of color, remains visible.
This **color mask** is shown below. This color mask is then combined with the contour mask created earlier in stage 3 to remove the last remaining unwanted contours. 
This creates the **final contour frame** shown below.

![color_mask](images/color_mask.png)
![final_contour_frame](images/final_contour_frame.png)

### Stage 5:
This final contour frame is used as a mask to crop out parts of the original rgb frame. The **Histogram of Oriented Gradient (HOG)** features are then extracted from all of these cropped out parts.
These HOG features are then used to create a positive training set for a **Support Vector Machine (SVM)**. 
Negative examples are generated from some of the other unwanted contour HOG features. Then model is also trained with other negetive examples collected from environments outside the lab.
A **Hard Negative Mining** is also performed with some of the difficult negetive examples to make the overall model more robust.
The images below shows the **cropped out portions** and the modified rgb frame and the **HOG features** extracted from them, which will be used to train the SVM.
The HOG features helps to identify the texture of the box. It gives an idea of its shape.

![cropped_image](images/cropped_image.png)
![hog_feature](images/hog_feature.png)




 





