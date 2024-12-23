import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration

# Load the blurred image
image_path = '../source/image.jpg'  # Change this to the path of your image
blurred_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Wiener filter for deblurring
def wiener_deblur(image):
    # Create a simple PSF (Point Spread Function)
    psf = np.ones((5, 5)) / 25  # A simple averaging kernel
    deblurred_image = restoration.wiener(image, psf, 1)  # 1 is a placeholder for the noise variance
    return deblurred_image

# Process the image
deblurred_image = wiener_deblur(blurred_image)

# Display original and deblurred images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(blurred_image, cmap='gray')
plt.title('Original Blurred Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(deblurred_image, cmap='gray')
plt.title('Deblurred Image (Wiener Filter)')
plt.axis('off')

plt.tight_layout()
plt.show()
