import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
import cv2
import os
from functions import sap_noise, concat_images


# Params
image_size = 128 #px
N_tele = 3 # Number of telescopes
PATH = "Data/original/"
PATH_out = f"Data/Ellipsoids_{image_size}px_{N_tele}tele/"
PATH_to_mask = f"Data/masks/{image_size}px/"
save_images = True
SAP_noise = True # Salt and pepper noise
sampling = True # Sparse sampling
alpha = 0.005 # Salt and Pepper Noise probabilitye


def main():
    """
    Script that calculates the Power spectrum (2D-FFT) of an image (i.e. a stellar source). Then automatically creates
    training, testing and validation datasets.
    Includes optional sparse sampling (mask needs to be provided) and addition of Salt and Pepper noise.
    """
    counter = 0

    # Create folders if necessary
    if save_images:
        folders = ["val", "test", "train"]
        for folder in folders:
            if not os.path.exists(f"{PATH_out}{folder}"):
                os.makedirs(f"{PATH_out}{folder}")
                print(f"{PATH_out}{folder} created.")

    # The mask for the sparse sampling can be outside the loop (more efficient)
    if sampling:
        #sampling_mask = sparse_sampling(n_ellipses=N_ellip, image_size=image_size, scaling=8)
        sampling_mask = np.load(f"{PATH_to_mask}MASK_{N_tele}_telescopes.npy")

    for filename in os.listdir(PATH):

        # Load image
        file = os.path.join(PATH, filename)

        # Ignore .directory
        if (filename == ".directory") or (filename == "images"):
            continue

        image_name = filename[:-4]
        image_original = np.load(file) # > input image
        img_org = image_original.copy() # > ground truth

        # (Optional) Calculate Salt and Pepper noise
        if SAP_noise:
            image_original = sap_noise(img_org, alpha)

        # Resize to image_size and subtract the mean (not for ground truth though!!!)
        img_ed = cv2.resize(image_original, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
        img_ed = img_ed - np.mean(img_ed)
        ground_truth = cv2.resize(img_org, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

        # Calculate 2D FFT
        img_fft = fft.fftshift(fft.fft2(fft.fftshift(img_ed)))
        fft_argument = np.abs(img_fft)

        # (Optional) sparse sampling
        if sampling:
            fft_argument = np.multiply(fft_argument, sampling_mask)

        # Normalization
        img_fft_norm = fft_argument / np.max(fft_argument)

        # The images are on float ([0,1]), but cv2 requires integer ([0, 255]):
        img_fft_norm = cv2.convertScaleAbs(img_fft_norm, alpha=(255.0))
        ground_truth = cv2.convertScaleAbs(ground_truth, alpha=(255.0))

        # Combine ground truth and input image for NN
        combined_image = concat_images(ground_truth, img_fft_norm)

        # Create directories and save images:
        # Save or display images > use cv2 instead of matplotlib, as this always saves as (64,64,4)
        # Already save them in test & train & validation datasets; seems random TODO: Improve/Cross-Check
        if save_images:
            if counter < 150: # Save to val
                cv2.imwrite(f"{PATH_out}val/{image_name}.jpg", combined_image)
            elif (counter >= 150) and (counter < 300):
                cv2.imwrite(f"{PATH_out}test/{image_name}.jpg", combined_image)
            else:
                cv2.imwrite(f"{PATH_out}train/{image_name}.jpg", combined_image)

        else:
            plt.imshow(ground_truth)
            print("Original image: ", np.mean(ground_truth))
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
