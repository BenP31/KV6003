import imageio
import tensorflow as tf
import pathlib
import numpy as np
import PIL
from .variables import BATCH_SIZE, IMAGE_EVAL_INPUT_PATH, IMAGE_EVAL_OUTPUT_PATH, IMAGE_SIZE, IMAGE_TRAIN_INPUT_PATH, IMAGE_TRAIN_OUTPUT_PATH, MASK_EVAL_INPUT_PATH, MASK_EVAL_OUTPUT_PATH, MASK_TRAIN_INPUT_PATH, MASK_TRAIN_OUTPUT_PATH



def get_mtrain_ds():
    raw_train_input_ds = tf.keras.utils.image_dataset_from_directory(
        MASK_TRAIN_INPUT_PATH,
        labels=None,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False,
    )
    raw_train_input_ds = raw_train_input_ds.map(lambda x: (x/127.5))
    raw_train_input_ds = raw_train_input_ds.map(lambda x: x[:,:,:,0])

    raw_train_output_ds = tf.keras.utils.image_dataset_from_directory(
        MASK_TRAIN_OUTPUT_PATH,
        labels=None,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False
    )
    raw_train_output_ds = raw_train_output_ds.map(lambda x: (x / 225))
    raw_train_output_ds = raw_train_output_ds.map(lambda x: x[:,:,:,0])

    ds =  tf.data.Dataset.zip((raw_train_input_ds, raw_train_output_ds))
    
    return ds 

def get_mval_ds():
    raw_val_input_ds = tf.keras.utils.image_dataset_from_directory(
        MASK_EVAL_INPUT_PATH,
        labels=None,
        batch_size=1,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False,
    )
    raw_val_input_ds = raw_val_input_ds.map(lambda x: (x/225))
    raw_val_input_ds = raw_val_input_ds.map(lambda x: x[:,:,:,0])

    raw_val_output_ds = tf.keras.utils.image_dataset_from_directory(
        MASK_EVAL_OUTPUT_PATH,
        labels=None,
        batch_size=1,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False,
    )
    raw_val_output_ds = raw_val_output_ds.map(lambda x: (x/225))
    raw_val_output_ds = raw_val_output_ds.map(lambda x: x[:,:,:,0])

    ds = tf.data.Dataset.zip((raw_val_input_ds, raw_val_output_ds))
    
    return ds

def get_itrain_ds():
    raw_train_input_ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_TRAIN_INPUT_PATH,
        labels=None,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False,
    )
    raw_train_input_ds = raw_train_input_ds.map(lambda x: (x/127.5)-1)

    raw_train_mask_ds = tf.keras.utils.image_dataset_from_directory(
        MASK_TRAIN_OUTPUT_PATH,
        labels=None,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False
    )
    raw_train_mask_ds = raw_train_mask_ds.map(lambda x: (x/127.5)-1)
    raw_train_mask_ds = raw_train_mask_ds.map(lambda x: x[:,:,:,0])

    raw_train_output_ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_TRAIN_OUTPUT_PATH,
        labels=None,
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False
    )
    raw_train_output_ds = raw_train_output_ds.map(lambda x: (x/127.5)-1)

    ds =  tf.data.Dataset.zip((raw_train_input_ds, raw_train_mask_ds, raw_train_output_ds))
    
    return ds 

def get_ival_ds():
    raw_val_input_ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_EVAL_INPUT_PATH,
        labels=None,
        batch_size=1,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False,
    )
    raw_val_input_ds = raw_val_input_ds.map(lambda x: (x/127.5)-1)

    raw_val_mask_ds = tf.keras.utils.image_dataset_from_directory(
        MASK_EVAL_OUTPUT_PATH,
        labels=None,
        batch_size=1,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False,
    )
    raw_val_mask_ds = raw_val_mask_ds.map(lambda x: (x/127.5)-1)
    raw_val_mask_ds = raw_val_mask_ds.map(lambda x: x[:,:,:,0])

    raw_val_output_ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_EVAL_OUTPUT_PATH,
        labels=None,
        batch_size=1,
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        shuffle=False,
    )
    raw_val_output_ds = raw_val_output_ds.map(lambda x: (x/127.5)-1)

    ds = tf.data.Dataset.zip((raw_val_input_ds, raw_val_mask_ds, raw_val_output_ds))
    
    return ds

def get_mask_ds():
    train_ds =  get_mtrain_ds()
    val_ds = get_mval_ds()
    return train_ds, val_ds

def get_image_ds():
    train_ds = get_itrain_ds()
    val_ds = get_ival_ds()
    return train_ds, val_ds