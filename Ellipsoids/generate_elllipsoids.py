import numpy as np
import os
import matplotlib.pyplot as pl
import time
from functions import grids, ellip


PATH = "TEST"
image_size = 512

def main():
    """
    Python script that generates ellipsoids of different size, angle and shape, and saves the images as .jpg images
    and as pure numpy arrays. Might take a long time run!
    Parameters:
    rad: size of the ellipsoid [3e-9 and 1.5e-8]
    inclination: Inclination of the ellipsoid [0, 2pi]
    pa: Rotation around x/y-axis [0, 2pi]
    sq: Thickness of the ellipsoid along z-axis [0.6, 1]
    Output: .jpg image & .npx array
    """

    # Fixed parameters
    sx, sy, x, y = grids(1e-10, image_size, 1e-6)
    epsilon = 0.07

    # Variable parameters (sx,sy,rad,inc,pa,sq):
    #   rad_ = np.arange(3e-9, 1.6e-8, 2e-9) # 7
    #   inclination_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    #   pa_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    #   sq_ = np.arange(0.6, 1.6, 0.15)  # 7
    """
    rad_ = np.arange(3e-9, 1.6e-8, 2e-9) # 7
    inclination_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    pa_ = np.arange(0, 2*np.pi, np.pi/4)  # 8
    sq_ = np.arange(0.5, 1, 0.1)  # 5
    """
    rad_ = [1.4e-8]
    inclination_ = [3*np.pi/4]
    pa_ = [np.pi]
    sq_ = [0.7]
    # Create new directory (if it not already exists)

    paths = [PATH, f"{PATH}/images"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created.")

    # Estimate number of images:
    total = len(rad_)*len(pa_)*len(sq_) + 2*(len(rad_))
    print(f"Generating {total} images...")

    start = time.time()

    # Loop over variable parameters (center, axesLength, angle)
    counter = 0
    for rad in rad_:
        for inclination in inclination_:

            # Add a small deviation, to dodge the singularities
            if (inclination == (1 / 2) * np.pi) or (inclination == (3 / 2) * np.pi):
                inclination += epsilon

            # Run only once in this case
            if (inclination == 0) or (inclination == np.pi):
                ellipse = ellip(sx, sy, rad, 0, 0, 1)
                pl.imsave(f"{PATH}/images/ellipsoid_{counter}.jpg", ellipse)
                np.save(f"{PATH}/ellipsoid_{counter}", ellipse)
                counter += 1

            else:
                for pa in pa_:
                    for sq in sq_:

                        # Draw ellipse and save both image and pure numpy array
                        ellipse = ellip(sx,sy,rad,inclination,pa,sq)
                        pl.imsave(f"{PATH}/images/ellipsoid_{counter}.jpg", ellipse)
                        np.save(f"{PATH}/ellipsoid_{counter}", ellipse)
                        counter += 1

                        # Print progress every 50 steps
                        if (counter % 50) == 0:
                            if counter != 0:
                                print(f"Progress: Step {counter}. Last 50 steps took {time.time()-start:.2f} seconds.")
                                start = time.time()

    print(f"{counter} images were created")

    return None


if __name__ == "__main__":
    main()
