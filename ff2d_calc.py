import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, ndimage
import cv2
import os

# Params
image_size = 64 #px
PATH = "Data/SpiralGalaxy/original"


def main():
    """
    Python script that calculates the 2D fourier transformation of all the spiral galaxy images, because these will be
    the input images for the NN.

    Bug: Some images are somehow wrongly labeled as .gif (by APOD). This seems to break mpimg.imread
    """

    counter = 0
    # Loop through all images in the desired folder:
    for filename in os.listdir(PATH):

        file = os.path.join(PATH, filename)
        image_name = filename[:-4]

        image_org = mpimg.imread(file)

        # Convert RGB to 1-channel grayscale and rescale
        # try is necessary because some images might already be grayscale
        try:
            img_gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        except:
            print(f"Image {filename} not RGB, so conversion failed.")

        img_gray_resized = cv2.resize(img_gray, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

        # Calculate 2D fourier transform (including fft-shifts)
        img_fft = fft.fftshift(fft.fft2(fft.fftshift(img_gray_resized)))
        argument = np.log(np.abs(img_fft))
        image_fft_norm = argument/np.max(argument)


        # Save both images in different folders
        mpimg.imsave(f"Data/SpiralGalaxy/rescaled/{image_name}_gray_resized.png", img_gray_resized, cmap=plt.get_cmap('gray'))
        mpimg.imsave(f"Data/SpiralGalaxy/ff2d/{image_name}_fft_norm.png", image_fft_norm, cmap=plt.get_cmap('gray'))
        #print(f"Data/SpiralGalaxy/ff2d/{image_name}_fft_norm.png")

        counter += 1

    print(f"{counter} images successfully converted.")


if __name__ == "__main__":
    main()
