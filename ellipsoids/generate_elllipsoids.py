import numpy as np
import os
import matplotlib.pyplot as pl
import time
from functions import grids, ellip


PATH = "Data/original_new2"
color_gradient = True


def main():
    """
    Function that generates ellipsoids of different size, angle and shape, and saves the images as .jpg images, as well
    as pure numpy arrays
    """

    # Fixed parameters
    sx, sy, x, y = grids(1e-10, 512, 1e-6)
    epsilon = 0.07


    # Variable parameters:
    ## First version
    #     rad_ = np.arange(3e-9, 1.6e-8, 2e-9) # 7
    #     inclination_ = np.arange(0,2*np.pi, np.pi/4)  # 8
    #     pa_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    #     sq_ = np.arange(0.6, 1.6, 0.1)  # 10

    ## Second version
    #   rad_ = np.arange(3e-9, 1.6e-8, 2e-9) # 7
    #   inclination_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    #   pa_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    #   sq_ = np.arange(0.6, 1.6, 0.15)  # 7

    # Ellipse inputs: (sx,sy,rad,inc,pa,sq)
    # Rad: between 3e-9 and 1.5e-8
    # Inclination: between 0 and 2pi? (artifacts for e.g. 1.57!!!)
    # pa: Rotation around (x/y axis (not z axis!!); between 0 and 2pi
    # sq: "thickness" in z-direction of the ellipsoid (pole-pole distance). Best between 0.5 and 1.2

    rad_ = np.arange(1.1e-8, 1.6e-8, 2e-9) # 3
    inclination_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    pa_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    sq_ = np.arange(0.6, 1.6, 0.15)  # 7



    # Create new directory (if it not already exists)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print(f"Directory {PATH} created.")

    # Estimate number of images:
    print(f"Generating {len(rad_)*len(inclination_)*len(pa_)*len(sq_)} images...")

    start = time.time()

    # Loop over variable parameters (center, axesLength, angle)
    counter = 1792
    for rad in rad_:
        for inclination in inclination_:

            # Add a small deviation, to dodge the singularities
            if (inclination == (1 / 2) * np.pi) or (inclination == (3 / 2) * np.pi):
                inclination += epsilon

            for pa in pa_:
                for sq in sq_:

                    # Draw ellipse and save both image and pure numpy array
                    ellipse = ellip(sx,sy,rad,inclination,pa,sq)
                    pl.imsave(f"{PATH}/images/ellipsoid_{counter}.jpg", ellipse)
                    np.save(f"{PATH}/ellipsoid_{counter}", ellipse)




                    # Print progress every 50 steps
                    if (counter % 50) == 0:
                        if counter != 0:
                            print(f"Progress: Step {counter}. Last 50 steps took {time.time()-start:.2f} seconds.")
                            start = time.time()
                    counter += 1

    print(f"{counter} images were created")

    return None


if __name__ == "__main__":
    main()
