import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, ndimage
import cv2
import os

# Params
image_size = 64 #px
PATH = "Data/original"
save_images = True
grayscale = True
SAP_noise = True
alpha = 0.05 # Salt and Pepper Noise probability


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


def saltandpepper(image, prob):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white

    return output


def main():
    """
    Python script that calculates the 2D fourier transformation of all the ellipsoids, because these will be
    the input images for the NN.
    """
    counter = 0

    # Create folders if necessary
    if save_images:
        folders = ["val", "test", "train"]
        for folder in folders:
            if not os.path.exists(f"Data/Ellipsoids/SAP/{folder}"):
                os.makedirs(f"Data/Ellipsoids/SAP/{folder}")
                print(f"Directory Data/Ellipsoids/SAP/{folder} created.")

    for filename in os.listdir(PATH):

        # Load image
        file = os.path.join(PATH, filename)

        # Ignore .directory
        if filename == ".directory":
            continue

        image_name = filename[:-4]
        img_org = mpimg.imread(file)

        # (Optional) Convert to grayscale
        if grayscale:
            try:
                img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
            except:
                print(f"Image {image_name} not converted to grayscale.")

        # (Optional) Calculate Salt and Pepper noise
        if SAP_noise:
            noise = saltandpepper(img_org, alpha)
            img_org = cv2.add(img_org, noise)
            label = "_noise"
        else:
            label = ""

        # Resize to image_size and subtract the mean
        img_org_ed = cv2.resize(img_org, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
        img_org_ed = img_org_ed - np.mean(img_org_ed)

        # Calculate 2D FFT and normalize
        img_fft = fft.fftshift(fft.fft2(fft.fftshift(img_org_ed)))
        fft_argument = np.abs(img_fft)
        img_fft_norm = fft_argument/np.max(fft_argument)

        # Seems to be required, otherwise images are saved as black -> Unclear why ?
        img_fft_norm = cv2.convertScaleAbs(img_fft_norm, alpha=(255.0))

        # Need to combine image to deal with tensorflow
        combined_image = concat_images(img_org_ed, img_fft_norm)

        # If noise is on, we need to combine it also with the original image (w/o noise):
        if SAP_noise:
            img_nonoise = mpimg.imread(file)
            img_nonoise = cv2.cvtColor(img_nonoise, cv2.COLOR_BGR2GRAY)
            img_nonoise = cv2.resize(img_nonoise, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
            img_nonoise = img_nonoise - np.mean(img_nonoise)
            combined_image = concat_images(img_nonoise, combined_image)

        ## Create directories and save images:
        # Save or display images > use cv2 instead of matplotlib, as this always saves as (64,64,4)
        # Already save them in test & train & validation datasets; seems random TODO: Improve/Cross-Check
        if save_images:
            if counter < 300: # Save to val
                cv2.imwrite(f"Data/Ellipsoids/SAP/val/{image_name}{label}.jpg", combined_image)
            elif (counter >= 300) and (counter < 600):
                cv2.imwrite(f"Data/Ellipsoids/SAP/test/{image_name}{label}.jpg", combined_image)
            else:
                cv2.imwrite(f"Data/Ellipsoids/SAP/train/{image_name}{label}.jpg", combined_image)

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
