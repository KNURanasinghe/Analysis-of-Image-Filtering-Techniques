import cv2

# Load an image
image = cv2.imread('../source/saltANDpepper.jpeg')

# Apply Median Filter
median_blur = cv2.medianBlur(image, 5)
median_blur1 = cv2.medianBlur(median_blur, 5)

# Show original and filtered image
cv2.imshow('Original', image)
cv2.imshow('Median Filter', median_blur)
cv2.imshow('Median Filter1', median_blur1)
cv2.waitKey(0)
cv2.destroyAllWindows()
