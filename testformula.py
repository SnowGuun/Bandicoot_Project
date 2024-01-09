import math

p = 95  # height from the middle point to the camera in centimeters
d1 = 10  # distance from the middle point to the corner aruco marker
r = 20   # horizontal distance from the middle point in centimeters
phi = math.asin(r / p)  # calculating phi as arcsin(r/p)
delta = math.pi*7 / 6  # delta value (30 degrees)

# List of theta values
theta_values = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]

# Calculate the distance for each theta and label as points
for i, theta in enumerate(theta_values, start=1):
    dist = math.sqrt(p**2 + d1**2 - 2*p*d1*math.sin(phi)*math.cos(theta - delta))
    print(f"Point {i} Distance: {dist} cm")