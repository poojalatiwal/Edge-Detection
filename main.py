import cv2
import numpy as np

image = cv2.imread('image.png')

# Check if the image is loaded correctly
if image is None:
    print(" Error: Failed to load image. Check the path or file name.")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute gradients using Sobel operator
gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

# Compute magnitude and direction of the gradient
magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Gradient X', gradient_x)
cv2.imshow('Gradient Y', gradient_y)
cv2.imshow('Magnitude', magnitude.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
