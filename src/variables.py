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

CHECKPOINT_DIR = "./training_checkpoints"

#Mask Dataset Paths
MASK_TRAIN_INPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/train/ObsAnnotations" 
MASK_TRAIN_OUTPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/train/Annotations" 
MASK_EVAL_INPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/val/ObsAnnotations" 
MASK_EVAL_OUTPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/val/Annotations" 
MASK_TEST_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/test/Annotations" 


##Image Dataset Paths
IMAGE_TRAIN_INPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/train/ObsImages" 
IMAGE_TRAIN_OUTPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/train/JPEGImages" 
IMAGE_EVAL_INPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/val/ObsImages" 
IMAGE_EVAL_OUTPUT_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/val/JPEGImages" 
IMAGE_TEST_PATH = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice/test/JPEGImages"