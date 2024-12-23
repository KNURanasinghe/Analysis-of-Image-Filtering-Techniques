import cv2
import numpy as np

# Load an image and convert to grayscale
image = cv2.imread('../source/OIP.jpeg', cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Create a mask with a low-pass filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# Apply the mask and inverse DFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Show original and filtered image
cv2.imshow('Original', image)
cv2.imshow('Low-pass Filtered (Fourier)', img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()
