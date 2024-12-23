import cv2
import numpy as np

# Load an image and convert to grayscale
image = cv2.imread('../source/OIP.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply Laplacian Filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Convert back to uint8
laplacian = cv2.convertScaleAbs(laplacian)

# Show original and Laplacian image
cv2.imshow('Original', image)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()
