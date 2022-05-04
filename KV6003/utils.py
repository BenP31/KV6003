import numpy as np
import tensorflow as tf
from .variables import BATCH_SIZE
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

def save_image(generated_image:tf.Tensor, save_path: str):
    """
    Saves a generated image in a tensor format to a jpg format file.
    generated_image: The generated image (tensor)
    save_path: The filepath to save the image in
    """
    generated_image = (generated_image + 1) * 127.5  
    generated_image = generated_image.numpy()
    image = tf.keras.utils.array_to_img(generated_image)
    image.save(save_path)

# def display_images(*args):
#     plt.figure(figsize=(15,15))

#     for i in range(len(args)):
#         plt.subplot(1,len(args),i+1)
#         plt.title("Image " + str(i))
#         plt.imshow(tf.squeeze(args[i]))
#         plt.axis("off")
#     plt.show()

def mask_to_image(mask_batch):
    for mask in mask_batch:
        rs_mask = tf.expand_dims(mask, axis=2)
        zeros = tf.zeros((256,256,2))
        image = tf.concat([rs_mask, zeros], 2)
    return mask_batch

def mask_iou(truth_tensor, synthetic_tensor, threshold = 0.2):
    """
    Function to find the Intersection Over Union value (Jaccard index) of two 
    masks (May not be truthful to original as masks are sometimes disjointed and
    exist in parts that are not joined together)
    """
    t = tf.reshape(truth_tensor>threshold,-1)
    s = tf.reshape(synthetic_tensor>threshold, -1)

    intersect = 0
    union = 0

    for i in range(len(t)-1):
        if t[i] == True:
            if s[i]== True:
                intersect += 1
                union += 1
        else:
            if s[i]==True:
                union += 1

    return intersect/union

def assemble_image(image, mask, size=BATCH_SIZE):
    mask = tf.expand_dims(mask, len(tf.shape(mask)))
    zeros = tf.zeros((256,256,2))
    assembled_image = tf.concat([image, mask, zeros], axis=2)
    return assembled_image

def mask_accuracy( prediction, truth, threshold=0.2):
    truth = tf.reshape(truth>threshold,-1)
    prediction = tf.reshape(prediction>threshold, -1)

    total = tf.reduce_sum(tf.cast(truth, tf.float32))
    pred = tf.reduce_sum(tf.cast(tf.math.logical_and(prediction, truth), tf.float32))

    return (pred/total) 

def get_inception():
    model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(256,256,3))
    return model

def calculate_fid(model, predicted_images, real_images, test=False):
    # Calculate the FID score for two sets of images
    # model: The model to use for the FID score (In most cases InceptionV3)
    # predicted_images: The predicted images to use for the FID score
    # real_images: The real images to use for the FID score
    # test: Whether or not to output the prediction information for InceptionV3
    pred_resized = tf.keras.preprocessing.image.smart_resize(predicted_images,(299,299))
    real_resized = tf.keras.preprocessing.image.smart_resize(real_images,(299,299))
    pred_results = model(pred_resized)
    real_results = model(real_resized)

    if test:
        print(pred_results, real_results)

    pred_mu = tf.reduce_mean(pred_results)
    real_mu = tf.reduce_mean(real_results)
    pred_sig = np.cov(pred_results, rowvar=False)
    real_sig = np.cov(real_results, rowvar=False)

    ssdif = np.sum((pred_mu-real_mu)**2)
    cov_mean = sqrtm(np.dot(pred_sig, real_sig))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = ssdif + np.trace(pred_sig + real_sig - 2 * cov_mean)
    print(f"FID: {fid:.4f}")
    return fid

def display_images(*args):
    plt.figure(figsize=(15,15))

    for i in range(len(args)):
        plt.subplot(1,len(args),i+1)
        plt.title("Image " + str(i))
        plt.imshow(tf.squeeze(args[i]))
        plt.axis("off")
    plt.show()