import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from kv6003 import get_image_ds
from object_detection.utils import ops
from object_detection.utils import visualization_utils as viz
from object_detection.utils.label_map_util import create_category_index_from_labelmap

segmen = hub.load("https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1")

def predict_mask(model, image):
    pred = model(image)
    output = {k: v.numpy() for k, v in pred.items()}
    detection_masks = tf.convert_to_tensor(output['detection_masks'][0])
    detection_boxes = tf.convert_to_tensor(output['detection_boxes'][0])
    detection_masks_reframed = ops.reframe_box_masks_to_image_masks \
    (detection_masks, detection_boxes, image.shape[1], image.shape[2])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
    output['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output

if __name__ == "__main__":
    _, val_ds = get_image_ds()
    for input_batch, mask_batch, target_batch in val_ds.take(1):
        for input, mask, target in zip(input_batch, mask_batch, target_batch):
            input = tf.cast((input+1)*127.5, tf.uint8)
            pred = predict_mask(segmen, tf.expand_dims(input,0))

    #display_images(pred['detection_masks_reframed'][pred['detection_classes'][0] == 1][2], input)
