import cv2
import numpy as np

# Load an image
image = cv2.imread('../source/noicyImg.jpg')

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
gaussian_blur1 = cv2.GaussianBlur(gaussian_blur, (5, 5), 0)

# Show original and blurred image
cv2.imshow('Original', image)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Gaussian Blur1', gaussian_blur1)
cv2.waitKey(0)
cv2.destroyAllWindows()
