# II_Project

Code used in the Researing Intership in Intensity Interferometry.

- ```generate_ellipsoids.py ```: Generates images of ellipsoids with different sizes, shapes and angular rotations. The output is stores in a folder ```Ellipsoids_original``` and imposes the basis of the training data. 

- ```generate_sampling_mask.py ```: Generates the sparse sampling mask based on the telescope layout at MAGIC (+LST) at the Northern CTA site. 

- ```ff2d_calc_ellipsoids.py```: Creates the input data for the GAN, based on the output of 'generate_ellipsoids.py'. It is important that the input sizes of the images match. This includes the optional sparse sampling (mask based on 'generate_sampling_mask.py') and introduced Salt-and-Pepper noise. 

- ```functions.py```: All the functions used in the three scripts above. Must be in the same directory to work. 

- ```Ellipsoids_v0.X.ipynb```: Different versions of the GAN. The important two versions are Version_0.7 (64x64px images) and Version_0.8 (128x128px images). 

- ```Analysis.ipynb```: 

