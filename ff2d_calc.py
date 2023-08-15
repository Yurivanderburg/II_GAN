import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, ndimage
import cv2
import os

# Params
image_size = 64 #px
PATH = "Data/SpiralGalaxy/original"
grayscale = False


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def main():
    """
    Python script that calculates the 2D fourier transformation of all the spiral galaxy images, because these will be
    the input images for the NN.

    Bug 1: Grayscale does not reduce to 1 channel, but to 4.
    Bug 2: Non-grayscale image cannot be saved, because imsave requires normalized arrays. -> image distorted
    """

    counter = 0
    # Loop through all images in the desired folder:
    for filename in os.listdir(PATH):

        file = os.path.join(PATH, filename)
        image_name = filename[:-4]

        image_org = mpimg.imread(file)

        # Convert RGB to 1-channel grayscale and rescale
        # try is necessary because some images might already be grayscale
        if grayscale:
            try:
                img_gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
                img_processed = cv2.resize(img_gray, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
            except:
                print(f"Image {filename} not RGB, so conversion failed.")

        else:
            img_processed = cv2.resize(image_org, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

        # Calculate 2D fourier transform (including fft-shifts)
        img_fft = fft.fftshift(fft.fft2(fft.fftshift(img_processed)))

        argument = np.log(np.abs(img_fft))
        #argument[np.isneginf(argument)] = 0
        #argument[np.isposinf(argument)] = 0
        image_fft_norm = normalize(argument/np.max(argument))



        # Save both images in different folders
        if grayscale:
            mpimg.imsave(f"Data/SpiralGalaxy/gray_rescaled/{image_name}_gray_resized.png", img_processed)
            mpimg.imsave(f"Data/SpiralGalaxy/gray_ff2d/{image_name}_gray_fft_norm.png", image_fft_norm)
                      #   cmap=plt.get_cmap('gray'))

        else:
            mpimg.imsave(f"Data/SpiralGalaxy/rescaled/{image_name}_resized.png", img_processed)
            mpimg.imsave(f"Data/SpiralGalaxy/ff2d/{image_name}_fft_norm.png", image_fft_norm)


        counter += 1
        if counter == 10:
            break

    print(f"{counter} images successfully converted.")


if __name__ == "__main__":
    main()