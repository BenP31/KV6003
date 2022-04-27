import tensorflow as tf
import tensorflow.keras.layers as layers

tf.config.run_functions_eagerly(True)


import os
from datetime import datetime
from random import randint
from time import time

import imageio
import numpy as np
import PIL

from .utils import  assemble_image, mask_accuracy
from .variables import (BATCH_SIZE, CLIP_WEIGHT, CRITIC_ITERATIONS, DROPOUT_RATE, 
                       IMAGE_OUTPUT_CHANNELS, IMAGE_SIZE)



def downsample1D(filters:int, size:int, i:int, apply_batchnorm = True) -> tf.keras.Model:
    """
    1 dimension version of the downsample layer function.
    Function to assemble each downsampling 'layer' of a network. Intended to funnel into more downsampling layers until bottleneck is reached.
    
    filters (int): dimensionality of the output of the convolutional layer (and therefore what will be passed to the next layer)
    size (int): height and width of the convolutional window
    i (int): a number given to the layer, normally the position of the layer in the collection of the same layer (for representation sake)
    apply_batchnorm (bool): If true, includes a batch normalisation layer after the convolutional layer

    Returns a sequential model with all layers
    """
    model = tf.keras.Sequential(name="Downsample1D_"+str(i))
    init = tf.random_normal_initializer(0., 0.02)

    model.add( layers.Conv1D(
        filters,
        size,
        strides=2,
        padding='same',
        kernel_initializer=init,
        use_bias=False,
    ))
    
    if apply_batchnorm:
        model.add( layers.BatchNormalization())

    model.add( layers.LeakyReLU())

    return model

def upsample1D(filters:int, size:int, i:int, apply_dropout = False) -> tf.keras.Model:
    """
    1 dimensional version of the upsample layer function.
    Function to assemble each upsampling 'layer' of a network. Intended to funnel into more upsampling layers until bottleneck is reached.
    
    filters (int): dimensionality of the output of the convolutional layer (and therefore what will be passed to the next layer)
    size (int): height and width of the convolutional window
    i (int): a number given to the layer, normally the position of the layer in the collection of the same layer (for representation sake)
    apply_dropout (bool): If true, includes a dropout layer after the batch normalisation layer

    Returns a sequential model with all layers
    """
    model = tf.keras.Sequential(name="Upsample1D_"+str(i))
    init = tf.random_normal_initializer(0., 0.02)

    model.add(layers.Conv1DTranspose(
        filters,
        size,
        strides=2,
        padding='same',
        kernel_initializer=init,
        use_bias=False,
    ))
    
    model.add(layers.BatchNormalization())

    if apply_dropout:
        model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.LeakyReLU())

    return model

def upsample(filters:int, size:int, i:int, apply_dropout = False) -> tf.keras.Model:
    """
    Function to assemble each upsampling 'layer' of a network. Intended to funnel into more upsampling layers until bottleneck is reached.
    
    filters (int): dimensionality of the output of the convolutional layer (and therefore what will be passed to the next layer)
    size (int): height and width of the convolutional window
    i (int): a number given to the layer, normally the position of the layer in the collection of the same layer (for representation sake)
    apply_dropout (bool): If true, includes a dropout layer after the batch normalisation layer

    Returns a sequential model with all layers
    """
    model = tf.keras.Sequential(name="Upsample_"+str(i))
    init = tf.random_normal_initializer(0., 0.02)

    model.add(layers.Conv2DTranspose(
        filters,
        size,
        strides=2,
        padding='same',
        kernel_initializer=init,
        use_bias=False,
    ))
    
    model.add(layers.BatchNormalization())

    if apply_dropout:
        model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.LeakyReLU())

    return model

def downsample(filters:int, size:int, i:int, apply_batchnorm = True) -> tf.keras.Model:
    """
    Function to assemble each downsampling 'layer' of a network. Intended to funnel into more downsampling layers until bottleneck is reached.
    
    filters (int): dimensionality of the output of the convolutional layer (and therefore what will be passed to the next layer)
    size (int): height and width of the convolutional window
    i (int): a number given to the layer, normally the position of the layer in the collection of the same layer (for representation sake)
    apply_batchnorm (bool): If true, includes a batch normalisation layer after the convolutional layer

    Returns a sequential model with all layers
    """
    model = tf.keras.Sequential(name="Downsample_"+str(i))
    init = tf.random_normal_initializer(0., 0.02)

    model.add( layers.Conv2D(
        filters,
        size,
        strides=2,
        padding='same',
        kernel_initializer=init,
        use_bias=False,
    ))
    
    if apply_batchnorm:
        model.add( layers.BatchNormalization())

    model.add( layers.LeakyReLU())

    return model

"""##Functions and optimisers"""

def generator_loss_fn(critic_output):
        """
        Loss function for the generator
        Based on the Wasserstein generator loss from "Wasserstein GAN" https://arxiv.org/abs/1701.07875
        critic_output: the value given by the critic

        Returns the loss value for the generator
        """

        #Loss for critic output 
        return -(tf.math.reduce_sum(critic_output))

def critic_loss_fn(real_output, fake_output): 
    """
    Loss function for the critic
    Based on the Wasserstein critic loss from "Wasserstein GAN" https://arxiv.org/abs/1701.07875
    real_output: the critic output for the real image
    fake_output: the critic output for the generated image

    Returns the loss value for the critic
    """

    # #Loss of the real output value
    real_loss = tf.math.reduce_mean(real_output)

    # #Loss of the output value generated by the critic
    fake_loss = tf.math.reduce_mean(fake_output)

    #Return the distance between the two losses
    return fake_loss - real_loss

"""## Mask Completion Models

###Generator
"""

def build_mask_generator():
    inputs = layers.Input(shape=[IMAGE_SIZE,IMAGE_SIZE], batch_size=BATCH_SIZE)

    parts = [
        #Layers of the downsampling part of the hourglass module
        downsample1D(64, 4, 1, apply_batchnorm = False),
        downsample1D(128, 4, 2),
        downsample1D(256, 4, 3),
        downsample1D(512, 4, 4),
        downsample1D(512, 4, 5),
        downsample1D(512, 4, 6),
        downsample1D(512, 4, 7),
        downsample1D(512, 4, 8),

        #Layers of the upsampling part of the hourglass module
        upsample1D(512, 4, 1, apply_dropout=True),
        upsample1D(512, 4, 2, apply_dropout=True),
        upsample1D(512, 4, 3, apply_dropout=True),
        upsample1D(512, 4, 4, apply_dropout=True),
        upsample1D(256, 4, 5),
        upsample1D(128, 4, 6),
        upsample1D(64, 4, 7)
    ]

    last = layers.Conv1DTranspose(IMAGE_SIZE, 4,
                                    strides = 2,
                                    padding = 'same',
                                    kernel_initializer = tf.random_normal_initializer(0, 0.02),
                                    activation = 'tanh')
    
    model = inputs

    for part in parts:
        model = part(model)
    
    model = last(model)

    return tf.keras.Model(inputs=inputs, outputs=model, name="Mask_Generator")

"""###Critic"""

def build_mask_critic():
    input = layers.Input(name="input", shape=[IMAGE_SIZE,IMAGE_SIZE], batch_size=BATCH_SIZE)

    down1 = downsample1D(64, 4, 1, apply_batchnorm = False)(input)
    down2 = downsample1D(128, 4, 2)(down1)
    down3 = downsample1D(256, 4, 3)(down2)

    zp = layers.ZeroPadding1D()(down3)

    down4 = downsample1D(512, 4, 4)(zp)

    zp2 = layers.ZeroPadding1D()(down4)
    c1d = layers.Conv1D(1,4, strides=1)(zp2)

    flatten = layers.Flatten()(c1d)
    out = layers.Dense(1, activation = 'linear')(flatten)

    return tf.keras.Model(inputs=input, outputs=out, name="Mask_Critic")

"""## Image Recovery Models

### Generator
"""

def build_image_generator():
    input = layers.Input(shape=[IMAGE_SIZE,IMAGE_SIZE, IMAGE_OUTPUT_CHANNELS*2], batch_size=BATCH_SIZE, name='Image')


    parts = [
        #Layers of the downsampling part of the hourglass module
        downsample(64, 4, 1, apply_batchnorm = False),
        downsample(128, 4, 2),
        downsample(256, 4, 3),
        downsample(512, 4, 4),
        downsample(512, 4, 5),
        downsample(512, 4, 6),
        downsample(512, 4, 7),
        downsample(512, 4, 8),

        #Layers of the upsampling part of the hourglass module
        upsample(512, 4, 1, apply_dropout=True),
        upsample(512, 4, 2, apply_dropout=True),
        upsample(512, 4, 3, apply_dropout=True),
        upsample(512, 4, 4, apply_dropout=True),
        upsample(256, 4, 5),
        upsample(128, 4, 6),
        upsample(64, 4, 7)
    ]

    last = layers.Conv2DTranspose(IMAGE_OUTPUT_CHANNELS, 4,
                                    strides = 2,
                                    padding = 'same',
                                    kernel_initializer = tf.random_normal_initializer(0, 0.02),
                                    activation = 'tanh')
    
    model = input

    for part in parts:
        model = part(model)
    
    model = last(model)

    return tf.keras.Model(inputs=input, outputs=model, name="Image_Generator")

"""### Critic"""

def build_image_critic():
    input = layers.Input(name="input", shape=[IMAGE_SIZE,IMAGE_SIZE, IMAGE_OUTPUT_CHANNELS], batch_size=BATCH_SIZE)

    down1 = downsample(64, 4, 1, apply_batchnorm = False)(input)
    down2 = downsample(128, 4, 2)(down1)
    down3 = downsample(256, 4, 3)(down2)

    zp = layers.ZeroPadding2D()(down3)
    down4 = downsample(512, 4, 4)(zp)
    zp2 = layers.ZeroPadding2D()(down4)
    c1d = layers.Conv2D(1,4, strides=1)(zp2)

    flatten = layers.Flatten()(c1d)
    out = layers.Dense(1, activation = 'linear')(flatten)

    return tf.keras.Model(inputs=input, outputs=out, name="Image_Critic")

"""## GAN struct and train functions"""

class GAN():
    def __init__(self, generator=None, critic=None):
        self.generator = generator
        self.critic = critic
        self.g_metric = tf.keras.metrics.Mean(name='c_loss')
        self.c_metric = tf.keras.metrics.Mean(name='g_loss')

    def compile(self, g_optimizer, c_optimizer, g_loss_fn, c_loss_fn):
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.g_loss_fn = g_loss_fn
        self.c_loss_fn = c_loss_fn

    @tf.function
    def train_step_critic(self, input, target):
        with tf.GradientTape() as crit_tape:
            gen_output = self.generator(input)

            crit_real_out = self.critic(target, training=True)
            crit_synth_out = self.critic(gen_output, training=True)

            crit_loss = self.c_loss_fn(crit_real_out, crit_synth_out)
            crit_gradient = crit_tape.gradient(crit_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(crit_gradient, self.critic.trainable_variables))

            for l in self.critic.layers:
                        weight = l.get_weights()
                        weight = [np.clip(w, -CLIP_WEIGHT, CLIP_WEIGHT) for w in weight]
                        l.set_weights(weight)

            self.c_metric.update_state(crit_loss)


    @tf.function
    def train_step_gen(self, input):
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(input, training=True)
            critic_out = self.critic(gen_output, training=True)

            gen_loss = self.g_loss_fn(critic_out)
            gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
            self.g_metric.update_state(gen_loss)

    # @tf.function
    # def train_step(self, input_image, target):
    #     with tf.GradientTape() as gen_tape, tf.GradientTape(persistent=True) as crit_tape:
    #         gen_output = self.generator(input_image, training=True)

    #        # Train the critic
    #         # Cutout the critic training when optimal (Continues to train generator)
                    

    #                 # Clip the weights to keep Lipschitz continuous (required for Wasserstein loss)
    #         # Train generator
            
    #         # Update metrics with new losses
            

    def fit(self, train_data, test_data, steps):
        start = time()
        for step, (train_input, train_target) in train_data.repeat().take(steps).enumerate():
            # Every 1k epochs, test the output of the generator
            if (step % 1000) == 0:
                if (step != 0):
                    for test_input, test_out in test_data.take(1):
                        print(f"\rTime taken for 1000 steps: {time() - start:.2f} seconds\n")

                        start = time()

                        gen_out = self.generator(test_input)

                        accuracy = mask_accuracy(test_out, gen_out)
                        print(f"Accuracy to target: {accuracy:.2f}%")
                        break

            for _ in range(CRITIC_ITERATIONS):
                self.train_step_critic(train_input, train_target)

            self.train_step_gen(train_input)

            # Print the training progress
            print(f"\r{step+1}/{steps} G_Loss: {self.g_metric.result():.10f} C_Loss: {self.c_metric.result():.10f} Time taken: {(time()-start):.4f}", end='')
            start = time()
        print("Finished training!")

    def __call__(self, image):
        return self.generator(image)

    def predict(self, image):
        return self.critic(image)

    def save_model(self, path):
        """
        Saves the models in a path that is 
        """
        if not os.path.exists(path):
            os.makedirs(path)
        dt = datetime.now()
        gen_dir = os.path.join(path, dt.strftime("%y%m%d-%H%M"),"generator")
        crit_dir = os.path.join(path, dt.strftime("%y%m%d-%H%M"),"critic")
        self.generator.save(gen_dir, save_format="tf", overwrite=True)
        self.critic.save(crit_dir, save_format="tf", overwrite=True)

    def load_model(self, path):
        if not os.path.exists(path):
            raise Exception("Filepath cannot be found")
        self.generator = tf.keras.models.load_model(os.path.join(path,"generator"))
        self.critic = tf.keras.models.load_model(os.path.join(path, "critic"))

class Image_Recovery_GAN(GAN):
    def __init__(self, generator=None, critic=None):
        super().__init__(generator, critic)

    @tf.function
    def train_step_critic(self, input_image, input_mask, target):
        with tf.GradientTape() as crit_tape:
            input = assemble_image(input_image, input_mask)
            gen_output = self.generator(input)

            crit_real_out = self.critic(target, training=True)
            crit_synth_out = self.critic(gen_output, training=True)

            crit_loss = self.c_loss_fn(crit_real_out, crit_synth_out)
            crit_gradient = crit_tape.gradient(crit_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(crit_gradient, self.critic.trainable_variables))

            for l in self.critic.layers:
                        weight = l.get_weights()
                        weight = [np.clip(w, -CLIP_WEIGHT, CLIP_WEIGHT) for w in weight]
                        l.set_weights(weight)

            self.c_metric.update_state(crit_loss)

    @tf.function
    def train_step_gen(self, input_image, input_mask):
        with tf.GradientTape() as gen_tape:
            input = assemble_image(input_image, input_mask)
            
            gen_output = self.generator(input, training=True)
            critic_out = self.critic(gen_output, training=True)

            gen_loss = self.g_loss_fn(critic_out)
            gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
            self.g_metric.update_state(gen_loss)


    # @tf.function
    # def train_step(self, input_image, input_mask, target):
    #     with tf.GradientTape() as gen_tape, tf.GradientTape(persistent=True) as crit_tape:

    #         # Process input into [batch_size, height, width, 6]
    #         input_mask = tf.expand_dims(input_mask, axis=3)
    #         zeros = tf.zeros((BATCH_SIZE,256,256,2))
    #         input = tf.concat([input_image, input_mask, zeros], axis=3)

    #         # Predict for the input
    #         gen_output = self.generator(input, training=True)

    #         # Train the critic
    #         # Cutout the critic training when optimal (Continues to train generator)
    #         if self.c_metric.result() <= CRITIC_LOSS_CUTOFF:
    #             for _ in range(CRITIC_ITERATIONS):
    #                 crit_real_out = self.critic(target, training=True)
    #                 crit_synth_out = self.critic(gen_output, training=True)

    #                 crit_loss = self.c_loss_fn(crit_real_out, crit_synth_out)
    #                 crit_gradient = crit_tape.gradient(crit_loss, self.critic.trainable_variables)
    #                 self.c_optimizer.apply_gradients(zip(crit_gradient, self.critic.trainable_variables))

    #                 # Clip the weights to keep within Lipschitz continuity (required for Wasserstein loss)
    #                 for l in self.critic.layers:
    #                     weight = l.get_weights()
    #                     weight = [np.clip(w, -CLIP_WEIGHT, CLIP_WEIGHT) for w in weight]
    #                     l.set_weights(weight)

    #         # Train the generator
    #         gen_loss = self.g_loss_fn(crit_synth_out)
    #         gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    #         self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
            
    #         # Update metrics with new losses
    #         self.c_metric.update_state(crit_loss)
    #         self.g_metric.update_state(gen_loss)

    def fit(self, train_data, test_data, steps):
        start = time()

        for step, (train_input, train_mask, train_target) in train_data.repeat().take(steps).enumerate():
            for _ in range(CRITIC_ITERATIONS):
                self.train_step_critic(train_input, train_mask, train_target)

            self.train_step_gen(train_input, train_mask)

            # Print the training progress
            print(f"\r{step+1}/{steps} G_Loss: {self.g_metric.result():.10f} C_Loss: {self.c_metric.result():.10f} Time taken: {(time()-start):.4f}", end='')
            start = time()
        print("Finished training!")

    def __call__(self, image, mask):
        input = assemble_image(image, mask)
        return self.generator(input)
