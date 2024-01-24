import cv2
import cv2.aruco as aruco
import numpy as np

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to find the middle point between two points
def calculate_middle_point(point1, point2):
    return (point1 + point2) / 2

# Function to calculate a point at a specified distance and direction
def calculate_point_away_from_middle(middle_point, distance, direction):
    if direction == 'top':
        return middle_point - np.array([0, distance])
    elif direction == 'down':
        return middle_point + np.array([0, distance])
    elif direction == 'left':
        return middle_point - np.array([distance, 0])
    elif direction == 'right':
        return middle_point + np.array([distance, 0])
    elif direction == 'top-left':
        return middle_point - np.array([distance, distance])
    elif direction == 'top-right':
        return middle_point - np.array([-distance, distance])
    elif direction == 'down-left':
        return middle_point - np.array([distance, -distance])
    elif direction == 'down-right':
        return middle_point - np.array([-distance, -distance])
    else:
        return middle_point

# Replace these values with your camera calibration parameters
fx, fy, cx, cy = 1000, 1000, 320, 240
k1, k2, p1, p2, k3 = 0, 0, 0, 0, 0

# Load the camera matrix and distortion coefficients
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

# Create ArUco parameters
parameters = aruco.DetectorParameters()

# Capture video from the camera (replace 0 with the camera index)
cap = cv2.VideoCapture(0)

# Dots and directions
dots = ['top', 'down', 'left', 'right', 'top-left', 'top-right', 'down-left', 'down-right']
current_dot = 0

while True:
    ret, frame = cap.read()

    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) >= 2:
        marker716_index = np.where(ids == 716)[0]
        marker592_index = np.where(ids == 592)[0]

        if len(marker716_index) > 0 and len(marker592_index) > 0:
            marker716_center = np.mean(corners[marker716_index[0]][0], axis=0)
            marker592_center = np.mean(corners[marker592_index[0]][0], axis=0)

            distance_716_592 = calculate_distance(marker716_center, marker592_center)
            middle_point = calculate_middle_point(marker716_center, marker592_center)

            # Draw lines connecting the markers
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw line and distance between markers
            frame = cv2.line(frame, tuple(marker716_center.astype(int)), tuple(marker592_center.astype(int)), (0, 255, 0), 2)
            frame = cv2.putText(frame, f"Distance 716-592: {distance_716_592:.2f} units", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw dot at the middle point
            frame = cv2.circle(frame, tuple(middle_point.astype(int)), 5, (0, 0, 255), -1)

            # Calculate and draw the current dot
            current_direction = dots[current_dot]
            distance_from_middle = 200  # Specify the distance in centimeters
            dot_point = calculate_point_away_from_middle(middle_point, distance_from_middle, current_direction)
            frame = cv2.circle(frame, tuple(dot_point.astype(int)), 5, (255, 0, 0), -1)

            # Display the current dot direction
            frame = cv2.putText(frame, f"Dot Direction: {current_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("ArUco Marker Distance Measurement", frame)

    # Check for user input to switch to the next dot
    key = cv2.waitKey(1)
    if key == ord('y') or key == ord('Y'):
        current_dot = (current_dot + 1) % len(dots)

    # Exit the program on 'q' key press
    elif key == ord('q') or key == 27:  # 27 is the ASCII code for the Escape key
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()