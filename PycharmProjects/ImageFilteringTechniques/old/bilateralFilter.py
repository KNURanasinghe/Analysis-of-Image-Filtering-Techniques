import cv2

# Load an image
image = cv2.imread('../source/noicyImg.jpg')

# Apply Bilateral Filter
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)
bilateral_filter1 = cv2.bilateralFilter(bilateral_filter, 9, 75, 75)

# Show original and filtered image
cv2.imshow('Original', image)
cv2.imshow('Bilateral Filter', bilateral_filter)
cv2.imshow('Bilateral Filter1', bilateral_filter1)
cv2.waitKey(0)
cv2.destroyAllWindows()
