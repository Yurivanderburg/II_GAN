import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, ndimage
import cv2
import os

# Params
image_size = 512 #px
PATH = "Data/Shapes/original2"
save_images = True


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

    # Create folders if necessary
    if save_images:
        folders = ["val", "test", "train"]
        for folder in folders:
            if not os.path.exists(f"Data/Shapes/{folder}"):
                os.makedirs(f"Data/Shapes/{folder}")
                print(f"Directory Data/Shapes/{folder} created.")

    for filename in os.listdir(PATH):

        file = os.path.join(PATH, filename)
        image_name = filename[:-4]
        img_org = mpimg.imread(file)

        ## Manipulate original images:
        # Convert to 1-channel grayscale
        # Resize images to image_size (skip images that are already grayscale)
        # subtract mean > better contrast
        '''
        try:
            img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        except:
            print(f"Image {image_name} not converted to grayscale.")

        img_org_ed = cv2.resize(img_org, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
        
        '''
        img_org_ed = img_org - np.mean(img_org)


        ## Generate power spectrum from images
        # Calculate 2D fourier transform (including shift)
        # Take absolute value and normalize
        img_fft = fft.fftshift(fft.fft2(fft.fftshift(img_org_ed)))
        fft_argument = np.abs(img_fft)
        img_fft_norm = fft_argument/np.max(fft_argument)

        # Seems to be required, otherwise images are saved as black -> Unclear why ?
        #img_org_ed = cv2.convertScaleAbs(img_org_ed, alpha=(255.0))
        img_fft_norm = cv2.convertScaleAbs(img_fft_norm, alpha=(255.0))

        # Need to combine image to deal with tensorflow
        combined_image = concat_images(img_org_ed, img_fft_norm)

        ## Create directories and save images:
        # Save or display images > use cv2 instead of matplotlib, as this always saves as (64,64,4)
        # Already save them in test & train & validation datasets; seems random TODO: Improve/Cross-Check
        if save_images:
            if counter < 5000: # Save to val
                cv2.imwrite(f"Data/Shapes/val/{image_name}.jpg", combined_image)
            elif (counter >= 5000) and (counter < 10000):
                cv2.imwrite(f"Data/Shapes/test/{image_name}.jpg", combined_image)
            else:
                cv2.imwrite(f"Data/Shapes/train/{image_name}.jpg", combined_image)

        else:
            plt.imshow(img_org_ed)
            print("Original image: ", np.mean(img_org_ed))
            plt.show()
            plt.imshow(img_fft_norm)
            print("FF2D image: ", np.mean(img_fft_norm))
            plt.show()

            if counter == 2:
                break

        counter += 1

    if save_images:
        print(f"{counter} images successfully converted.")
    else:
        print(f"{counter} images not saved")


if __name__ == "__main__":
    main()
