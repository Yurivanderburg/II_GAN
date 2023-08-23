import numpy as np
import cv2
from PIL import Image
import os


PATH = "Data/Ellipses/original"


def main():
    """
    Function that generates ellipses of different size, angle and shape, and saves the images as .jpg images.
    """

    # Fixed parameters
    image_size = 64
    startAngle = 0
    endAngle = 360
    color = (255, 0, 0)
    thickness = -1

    # Variable parameters:
    center_x = np.arange(12, 52, 12)  # 5
    center_y = np.arange(12, 52, 12)  # 5
    axesLength1 = np.arange(3, 35, 7)  # 5
    axesLength2 = np.arange(3, 35, 7)  # 5
    angles = np.arange(0, 360, 20)  # 15

    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print(f"Directory {PATH} created.")



    iterations = (len(center_x) * len(center_y) * len(axesLength1) * len(axesLength2) * len(angles))
    print(f"Generate almost {iterations} images...")

    counter = 0

    # Loop over variable parameters (center, axesLength, angle)
    for x_pos in center_x:
        for y_pos in center_y:
            center = (x_pos, y_pos)
            for axis1 in axesLength1:
                for axis2 in axesLength2:
                    axesLength = (axis1, axis2)
                    for angle in angles:

                        # Further conditions:
                        # 1) Only rotate non-spherical shapes
                        # 2) Not too eccentric

                        if axis1 != axis2:
                            if abs(axis1 - axis2) <= 150:

                                new_image = np.zeros(shape=(image_size, image_size))
                                image_ellipse = cv2.ellipse(new_image, center, axesLength, angle, startAngle, endAngle,
                                                            color, thickness)
                                # Save image
                                image = Image.fromarray(image_ellipse.astype(np.uint8))
                                image.save(f"{PATH}/ellipse_{counter}.jpg")

                                # Print progress every 10'000 steps
                                if (counter % 10000) == 0:
                                    print(f"Progress: Step {counter}...")
                                counter += 1

    print(f"{counter} images were created")

    return None


if __name__ == "__main__":
    main()
