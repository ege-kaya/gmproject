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

You must download the learned model parameters from the following link and put these under the subdirectory ``models`` in the same directory as ``testbench.ipynb``: https://drive.google.com/drive/folders/1datMe52uMlDHtTKbOooFJP5esqUNZD5f?usp=share_link. I was not able to upload these to GitHub because they were too large.

The following files were used only for training, if you would like to see the training code, you can check them out but running them is not advised:
* ``gan_s1_CelebA.py``
* ``gan_s2_CelebA.py``
* ``gan_s1_MNIST.py``
* ``gan_s2_MNIST.py``
* ``vae_s1.py``
* ``vae_s2.py``

You can see all of the main results in the Jupyter notebook ``testbench.ipynb``. The notebook already has pre-run results. However, you can re-run the cells to get novel outputs.

Be advised, however, that the cells calculating the closest images in the training dataset for the VAEs have a long running time (about 30 minutes each in my case.)

The code is written so that the CelebA and MNIST datasets are downloaded automatically at the start. However, in my case, PyTorch was unable to access the CelebA dataset mirror. In this case, you would have to procure the required CelebA dataset files (``mg_align_celeba.zip, list_attr_celeba.txt, identity_CelebA.txt, list_bbox_celeba.txt, list_landmarks_align_celeba.txt, list_eval_partition.txt``) from the following link: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg.
