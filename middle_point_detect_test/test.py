import cv2
import numpy as np

def find_large_rectangle(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and help edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_rectangles = []
    
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Set a threshold for the minimum area to consider as a large rectangle
        if area > 1500:  # Adjust this threshold based on your specific case
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if the polygon has 4 corners (rectangle)
            if len(approx) == 4:
                large_rectangles.append(approx)
    
    return large_rectangles

def draw_middle_point(frame, rectangle):
    # Calculate the middle point of the rectangle
    middle_point = np.mean(rectangle, axis=0, dtype=int)
    middle_point = tuple(middle_point.flatten())
    
    # Draw a circle at the middle point
    cv2.circle(frame, middle_point, 5, (0, 255, 0), -1)
    
    return frame

# Open a video capture object for the webcam (use index 0)
cap = cv2.VideoCapture(1)

# Initialize frame_with_point before the loop
frame_with_point = None

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    large_rectangles = find_large_rectangle(frame)
    
    for rect in large_rectangles:
        frame_with_point = draw_middle_point(frame.copy(), rect)
    
    # Check if frame_with_point is not None before displaying
    if frame_with_point is not None:
        cv2.imshow('Large Rectangle with Middle Point', frame_with_point)
    
    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
