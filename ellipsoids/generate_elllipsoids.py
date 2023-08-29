import numpy as np
import os
import matplotlib.pyplot as pl
from functions import grids, ellip



PATH = "Data/Tests"
color_gradient = True


def main():
    """
    Function that generates ellipsoids of different size, angle and shape, and saves the images as .jpg images.
    """

    # Fixed parameters
    image_size = 64
    sx, sy, x, y = grids(1e-10, 512, 1e-6)


    # Variable parameters:
    # Ellipse inputs: (sx,sy,rad,inc,pa,sq)
    # Rad: between 3e-9 and 1.5e-8
    # Inclination: between 0 and 2pi? (artifacts for e.g. 1.57!!!)
    # pa: Rotation around (x/y axis (not z axis!!); between 0 and 2pi
    # sq: "thickness" in z-direction of the ellipsoid (pole-pole distance). Best between 0.5 and 1.2
    rad_ = np.arange(3e-9, 1.6e-8, 2e-9) # 7
    inclination_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    pa_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    sq_ = np.arange(0.5, 1.2, 0.1)  # 7

    # Create new directory (if it not already exists)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print(f"Directory {PATH} created.")

    # Loop over variable parameters (center, axesLength, angle)
    counter = 0
    for rad in rad_:
        for inclination in inclination_:
            for pa in pa_:
                for sq in sq_:

                    # Draw ellipse

                    ellipse = ellip(sx,sy,rad,inclination,pa,sq)
                    #image = Image.fromarray(ellipse.astype(np.uint8))
                    pl.imsave(f"{PATH}/ellipsoid_{counter}.jpg", ellipse)

                    # Print progress every 1'000 steps
                    if (counter % 100) == 0:
                        print(f"Progress: Step {counter} ___________________________")
                    counter += 1


    print(f"{counter} images were created")

    return None


if __name__ == "__main__":
    main()
