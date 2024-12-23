import cv2
import numpy as np

# Load an image and convert to grayscale
image = cv2.imread('../source/OIP.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel Filter (X and Y direction)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges

# Convert back to uint8
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Show Sobel X and Sobel Y
cv2.imshow('original', image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
