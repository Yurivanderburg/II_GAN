import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, ndimage
import cv2
import os

# Params
image_size = 64 #px
PATH = "Data/SpiralGalaxy/original"
save_images = True


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


# Try to concat images
def concat_images(img_a, img_b):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width)) #, 1))
    new_img[:ha, :wa] = img_a
    new_img[:hb, wa:wa+wb] = img_b
    return new_img


def main():
    """
    Python script that calculates the 2D fourier transformation of all the spiral galaxy images, because these will be
    the input images for the NN.

    Bug 1: Grayscale does not reduce to 1 channel, but to 4.
    Bug 2: Non-grayscale image cannot be saved, because imsave requires normalized arrays. -> image distorted
    """
    counter = 0

    for filename in os.listdir(PATH):

        file = os.path.join(PATH, filename)
        image_name = filename[:-4]

        # Convert to 1-channel grayscale and resize to 64px
        img_org = mpimg.imread(file)

        # Skip images that are already greyscale
        try:
            img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

        except:
            print(f"Image {image_name} not converted to grayscale.")

        img_org_resized = cv2.resize(img_org, dsize=(64,64), interpolation=cv2.INTER_AREA)

        # Calculate FF2D incl. Shift
        img_fft = fft.fftshift(fft.fft2(fft.fftshift(img_org_resized)))
        fft_argument = np.log(np.abs(img_fft))
        img_fft_norm = fft_argument/np.max(fft_argument)

        # Seems to be required, otherwise images are saved as black -> Unclear why ?
        # img_org_resized = cv2.convertScaleAbs(img_org_resized, alpha=(255.0))
        img_fft_norm = cv2.convertScaleAbs(img_fft_norm, alpha=(255.0))

        # Need to combine image to deal with tensorflow
        combined_image = concat_images(img_org_resized, img_fft_norm)

        # Save or display images > use cv2 instead of matplotlib, as this always saves as (64,64,4)
        # Already save them in test & train & validation datasets; seems random TODO: Improve/Cross-Check
        if save_images:
            if counter <= 150: # Save to val
                cv2.imwrite(f"Data/SpiralGalaxy/val/{image_name}.jpg", combined_image)
            elif (counter > 150) and (counter <= 550):
                cv2.imwrite(f"Data/SpiralGalaxy/test/{image_name}.jpg", combined_image)
            else:
                cv2.imwrite(f"Data/SpiralGalaxy/train/{image_name}.jpg", combined_image)

        else:
            plt.imshow(img_org_resized)
            print(img_org_resized.shape)
            plt.show()
            plt.imshow(img_fft_norm)
            print(img_fft_norm.shape)
            plt.show()

            if counter == 10:
                break

        counter += 1

    if save_images:
        print(f"{counter} images successfully converted.")
    else:
        print(f"{counter} images not saved")


if __name__ == "__main__":
    main()
