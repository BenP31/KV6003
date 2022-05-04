
################################ MODEL HYPERPARAMETERS ################################
BATCH_SIZE =  4 
GEN_LR =  1e-4 
CRITIC_LR = 1e-4
DROPOUT_RATE = 0.3 
CRITIC_ITERATIONS =   5
CLIP_WEIGHT = 0.01 
EPOCHS = 20000

MASK_OUTPUT_CHANNELS = 1
IMAGE_OUTPUT_CHANNELS = 3 
IMAGE_SIZE = 256

################################ FILE PATHS ################################

CHECKPOINT_DIR = "./training_checkpoints"

#Mask Dataset Paths
MASK_TRAIN_INPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/train/ObsAnnotations" 
MASK_TRAIN_OUTPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/train/Annotations" 
MASK_EVAL_INPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/val/ObsAnnotations" 
MASK_EVAL_OUTPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/val/Annotations" 
MASK_TEST_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/test/Annotations" 

#Image Dataset Paths
IMAGE_TRAIN_INPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/train/ObsImages" 
IMAGE_TRAIN_OUTPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/train/JPEGImages" 
IMAGE_EVAL_INPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/val/ObsImages" 
IMAGE_EVAL_OUTPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/val/JPEGImages" 
IMAGE_TEST_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/test/JPEGImages"

#Test Dataset Paths
TEST_IMAGES_OBS = "D:/code-projects/KV6003-GAN/datasets/AHP/test/RGBErased"
TEST_MASKS = "D:/code-projects/KV6003-GAN/datasets/AHP/test/ModalMasks"
TEST_IMAGES_REAL = "D:/code-projects/KV6003-GAN/datasets/AHP/test/RGBImages"

################################ MODEL SAVES ################################

IMAGE_GEN = "D:\code-projects\KV6003-GAN\TF\KV6003\model_saves\image_gan\save-20000-epochs\generator"
MASK_GEN = "D:\code-projects\KV6003-GAN\TF\KV6003\model_saves\mask_gan\save-10000-epochs\generator"
