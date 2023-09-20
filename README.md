# II_Project

Code used in the Researing Intership in Intensity Interferometry, where I adapted the pix2pix GAN (Generative Adversarial Network) to work on the phase retrieval problem, specifially on the power spectrum from a fast-rotating star. 


- ```generate_ellipsoids.py ```: Generates images of ellipsoids with different sizes, shapes and angular rotations. The output is stores in a folder ```Ellipsoids_original``` and imposes the basis of the training data. 

- ```generate_sampling_mask.py ```: Generates the sparse sampling mask based on the telescope layout at MAGIC (+LST) at the Northern CTA site. 

- ```ff2d_calc_ellipsoids.py```: Creates the input data for the GAN, based on the output of 'generate_ellipsoids.py'. It is important that the input sizes of the images match. This includes the optional sparse sampling (mask based on 'generate_sampling_mask.py') and introduced Salt-and-Pepper noise. 

- ```functions.py```: All the functions used in the three scripts above. Must be in the same directory to work. 

- ```Ellipsoids_v0.X.ipynb```: Different versions of the GAN. The important two versions are Version_0.7 (64x64px images) and Version_0.8 (128x128px images). When running an existing model, it is very important that the architecture machtes with the one used in the model, otherwise there might be an error (and the generated images will not work properly). 

- ```Analysis.ipynb```: Analysis of the model performance for different hyperparameters. Downloads the logs from Tensorboard.dev and creates a pandas dataframe. This can be used to plot the different losses for different parameters. 



To replicate this setup, please:

1. Install the packages, e.g. with ``` pip install requirements.txt```

2. Run all the scripts in the order: generate_ellipsoids, generate_sampling_mask, ff2d_calc_ellipsoids: ```python3 genererate_ellipsoids.py```

3. Adapt the parameters in the start of the Ellipsoids.v0.X notebook. And start the training!


If you only want to see the results:

1. Install requirements. 

2. Open the ```Ellipsoids_v0.X.ipynb``` notebook, adapt parameters and set 'run_training' to True. Set the correct path to the model in the last section of the notebook. 


