import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
import cv2
import os

# Params
image_size = 64 #px
PATH = "Data/original_new"
PATH_out = "Data/Ellipsoids_sampled_5ellip/"
save_images = True
SAP_noise = True # Salt and pepper noise
sampling = True # Sparse sampling

alpha = 0.005 # Salt and Pepper Noise probability
N_ellip = 5 # Creates one centered ellipse


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
        white = 1
    else:
        raise ValueError('This image has multiple channels, which is not supported.')

    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white

    return output


def make_grid(N, grid_size):
    xlist = []
    positions = []

    for i in range(N+1):
        if i != 0:
            xlist.append(np.round((grid_size*(i/(N+1)))))

    for i in xlist:
        for j in xlist:
            positions.append([i,j])

    return np.array(positions)


def sparse_sampling(n_ellipses, scaling=8):
    """
    Adds a mask (0 or 1s) where the image is sampled. Can be either one centered ellipse, or a grid of ellipses
    n_ellipses > 1: creates a nxn grid, so n_ellipses^2 samples
    scaling: scales the image, such that the borders are smoother (default: 64 > 512px)
    """
    img_size = image_size * scaling
    mask = np.zeros(shape=(img_size, img_size))
    angle = 135
    startAngle = 0
    endAngle = 360
    color = 1
    thickness = -1

    if n_ellipses == 1:
        center = (int(img_size / 2), int(img_size / 2))
        axesLength = (int(img_size / 8), int(img_size / 16))

        mask = cv2.ellipse(mask, center, axesLength, angle, startAngle, endAngle, color, thickness)
        mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

        return mask

    else:
        mask_raw = np.zeros(shape=(img_size, img_size))
        mask_final = np.zeros(shape=(img_size, img_size))
        axesLength = (int(img_size / (3 * n_ellipses)), int(img_size / (5 * n_ellipses)))

        points = make_grid(n_ellipses, img_size)

        for i in range(len(points)):
            center = (int(points[i, 0]), int(points[i, 1]))
            # Add current ellipse
            mask_final += cv2.ellipse(mask_raw, center, axesLength, angle, startAngle, endAngle, color, thickness)
            mask_raw = np.zeros(shape=(img_size, img_size))

        mask_final = cv2.resize(mask_final, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

        return mask_final


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
            if not os.path.exists(f"{PATH_out}{folder}"):
                os.makedirs(f"{PATH_out}{folder}")
                print(f"{PATH_out}{folder} created.")

    # The mask for the sparse sampling can be outside the loop (more efficient)
    if sampling:
        sampling_mask = sparse_sampling(n_ellipses=N_ellip, scaling=8)

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
            image_original = saltandpepper(img_org, alpha)

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
            if counter < 300: # Save to val
                cv2.imwrite(f"{PATH_out}val/{image_name}.jpg", combined_image)
            elif (counter >= 300) and (counter < 600):
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
