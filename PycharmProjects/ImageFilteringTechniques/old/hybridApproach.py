import cv2
import numpy as np

# Load the image in color
image = cv2.imread('../source/noicyImg.jpg')

#result = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)










# Display the result
cv2.imshow(' Image', image)
cv2.imshow('Denoised Image', dft)
cv2.waitKey(0)
cv2.destroyAllWindows()
