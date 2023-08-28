import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


PATH = "Data/Ellipses/Tests"
color_gradient = True


def main():
    """
    Function that generates ellipses of different size, angle and shape, and saves the images as .jpg images.
    The ellipses
    """

    # Fixed parameters
    image_size = 64
    startAngle = 0
    endAngle = 360
    center = (int(image_size / 2), int(image_size / 2))
    color = (255, 0, 0)
    thickness = -1

    # Variable parameters:
    axesLength1 = np.arange(6, 32, 2)  # 15
    axesLength2 = np.arange(6, 32, 2)  # 15
    angles = np.arange(0, 360, 20)  # 15

    for axis in [axesLength1, axesLength2]:
        if np.max(axis) > (image_size / 2):
            raise ValueError("Ellipse should not be larger than the image")

    # Create new directory (if it not already exists)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print(f"Directory {PATH} created.")

    # Estimate number of iterations
    iterations = (len(axesLength1) * len(axesLength2) * len(angles))
    print(f"Generate almost {iterations} images...")

    # Loop over variable parameters (center, axesLength, angle)
    counter = 0
    counter_failed = 0
    for axis1 in axesLength1:
        for axis2 in axesLength2:
            axesLength = (axis1, axis2)
            for angle in angles:

                # Condition: ellipse not too eccentric
                if abs(axis1 - axis2) <= 18:

                    # Create ellipse
                    new_image = np.zeros(shape=(image_size, image_size))

                    image_ellipse = cv2.ellipse(new_image, center, axesLength, angle, startAngle, endAngle,
                                                color, thickness)

                    if color_gradient:
                        # Create mask for color gradient
                        upper_side = center[1] - axesLength[1]
                        middle = center[1]
                        lower_side = center[1] + axesLength[1]
                        difference = (axesLength[1] + 1)
                        col_grad = np.linspace(1, 2, difference)
                        mask = np.ones((image_size, image_size))

                        # Rotation
                        if angle != 0:

                            # Define an offset, because of issues with the rotation
                            offset = int(5 * np.sin(2*angle * np.pi / 180) ** 2)
                            length = int(np.sqrt((upper_side / 2) ** 2 + (lower_side / 2) ** 2))+offset
                            col_grad_rot = np.linspace(1, 2, length)

                            try:
                                for i in range(image_size):
                                    mask[middle:middle + length, i] = col_grad_rot
                                    mask[middle - length:middle, i] = np.flip(col_grad_rot)

                                # Rotate the mask and merge with image
                                mask_rot = rotate(mask, angle=angle, reshape=False)
                                image_ell_mask = np.multiply(mask_rot, image_ellipse)

                            except:
                                print(f"Image No. {counter} failed.")

                        # No rotation
                        else:
                            for i in range(image_size):
                                mask[middle:(lower_side + 1), i] = col_grad
                                mask[upper_side:(middle + 1), i] = np.flip(col_grad)

                            # Merge with the mask
                            image_ell_mask = np.multiply(mask, image_ellipse)

                        # Save image
                        image_ell_mask = image_ell_mask * (255 / np.max(image_ell_mask))
                        image = Image.fromarray(image_ell_mask.astype(np.uint8))

                        # print("Image stat:", np.max(image), np.min(image))
                        image.save(f"{PATH}/ellipse_{counter}.jpg")


                    # If not color_gradient:
                    else:
                        image = Image.fromarray(image_ellipse.astype(np.uint8))
                        image.save(f"{PATH}/ellipse_{counter}.jpg")

                    # Print progress every 1'000 steps
                    if (counter % 1000) == 0:
                        print(f"Progress: Step {counter} ___________________________")
                    counter += 1

    print(f"{counter} images were created")

    return None


if __name__ == "__main__":
    main()
