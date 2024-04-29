# ECE695: Inference and Learning in Generative Models Term Project

The following libraries are needed:
* NumPy
* Matplotlib
* PyTorch
* PIL
* Torchvision
* scikit-image
* pickle
* tqdm

The following files were used only for training, if you would like to see the training code, you can check them out but running them is not advised:
* ``gan_s1_CelebA.py``
* ``gan_s2_CelebA.py``
* ``gan_s1_MNIST.py``
* ``gan_s2_MNIST.py``
* ``vae_s1.py``
* ``vae_s2.py``

You can see all of the main results in the Jupyter notebook ``testbench.ipynb``. The notebook already has pre-run results. However, you can re-run the cells to get novel outputs.

Be advised, however, that the cells calculating the closest images in the training dataset for the VAEs have a long running time (about 30 minutes each in my case.)
