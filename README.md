# KV6003
 KV6003 Individual Computing Project 
>Ben Palmer 19005151
>ben.c.palmer31@outlook.com

---

## How to run on Google Colab (RECOMMENDED WAY):
1. Upload KV6003-Training.ipynb or KV6003-Main.ipynb to Google Colab
2. Go to "Runtime" > "Change runtime type" and select GPU.
3. Run all cells as needed.

---

## How to run locally (NOT RECOMMENDED):
1. Training files can all be ran locally with Tensorflow v2.7 or greater.
2. get_mask.py and image segmentation model requires TensorFlow ObjectDetection API (hard to install). Instructions on installing this can be found here: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
3. Use KV6003-Training.ipynb or KV6003-Main.ipynb in an IDE of your choosing and run

---

## File explanation:
### Notebook Files
- KV6003-Training.ipynb - Training notebook that was used during development and for the iterative development process
- KV6003-Main.ipynb - Main notebook to use all trained models simultaneously and run inference on uploaded images

### Python files
- variables.py - Used to store variables used in the system, such as hyperparameters and file locations

#### Dataset files
- ahp_slicer.py - Used to create a small "slice" of the AHP dataset with reduced size
- ds_obscurer.py - Used to obscure images within the training and validation sets of the dataset

#### Model files
- dataset.py - Loads the dataset into TensorFlow during runtime
- get_mask.py - Contains function for generating segmentation masks of an image (Requires Object Detection API)
- model.py - Contains the model functions and GAN classes used in this project
- train_image.py - Trains an instance of the image recovery network
- train_mask.py - Trains an instance of the mask recreation network
- utils.py - contains utility functions used by the project

---
## Model saves:
Model saves are too large to be kept on this git repository, so are available here: https://drive.google.com/file/d/1H21qF0Jl2yMPFPkoEZvmes-xzP1ZXo2u/view?usp=sharing