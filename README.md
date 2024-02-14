PLEASE NOTE: THIS CODE IS MODIFIED FOR TESTING ON LAPTOP, PC ONLY. THERE'S ANOTHER DIFFERENT ONE IN THE DEVICE
# FabricScan 360

## Overview
FabricScan 360 is an innovative smart scanning device designed to revolutionize the process of fabric digitization. Leveraging advanced computer vision technology with OpenCV, Raspberry Pi, and Aruco markers, FabricScan 360 guides users to accurately position their cameras for optimal fabric imaging. This tool aims to enhance the accuracy, efficiency, and accessibility of fabric digitalization, making high-quality digital twins of fabrics easily achievable for users of all skill levels.

## Features
- **High Accuracy:** Ensures detailed and precise digital replicas of physical fabrics.
- **User Accessibility:** Simplifies the digitization process, suitable for non-technical users.
- **Efficiency:** Speeds up the digitalization workflow, enabling rapid scaling of digital fabric collections.

## Prerequisites
- Raspberry Pi (Model 3B+ or newer recommended)
- Compatible camera module for Raspberry Pi
- Bandicoot fabric scanning chart with the aruco marker

## Before you started 
- Check on the Raspberry Pi if the camera is enable: 
+ Open Terminal and Enter: 
sudo raspi-config
+ Choose Interface Options - Legacy Camera - Select Yes that you’d like the camera interface enabled
+ Go back to Terminal and Enter: 
vcgencmd get_camera 
+ You should get back supported=1, detected=1, indicating that the camera is detected and supported by the operating system. If you get detected=0, then the camera is not being seen by the operating system.

- You can check the camera function via: testpicamera.py

- Check the camera calibration: If you are planning to use another camera module or use another chart with a different aruco markers, please do the camera calibration again. You can do that by: 
+ Delete the datas in calib_data and images (delete all .npz and png)
+ Print out a Camera Calibration checkerboard. You can make one here:
https://markhedleyjones.com/projects/calibration-checkerboard-collection
NOTE: The size of the of checkerboard square need to match the aruco marker size in order to get the correct measurement.
+ Once you have the checkerboard printed out. Run "captureImages.py" and take picture of the checkboard with the current or new camera module (press Y). Min: 30 shoots of different angle of the checkerboard. All pictures will be stored in "images" folder for calibration.
+ Finally, run "cameraCalibration" to process the new images and a new .npz file will be created in "calib_data" folder

## Contact
- Project Lead: Huynh Phuong Anh Nguyen - phnganhnguyn0323@gmail.com
- Project Link: https://github.com/yourusername/FabricScan360

## Run the code
Once you have everything set up you can run the code "pointTracker.py" via "pointTracker" folder


## Acknowledgements
- [OpenCV](https://opencv.org/)
- [Raspberry Pi Foundation](https://www.raspberrypi.org/)
- [Aruco Marker Library](https://www.uco.es/investiga/grupos/ava/node/26)
