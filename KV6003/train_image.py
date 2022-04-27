from .model import Image_Recovery_GAN, build_image_critic, build_image_generator, generator_loss_fn, critic_loss_fn
from .dataset import get_image_ds
from .variables import GEN_LR, CRITIC_LR, EPOCHS
import tensorflow as tf
from os import system

if __name__ == "__main__":
    system("cls")
    train_ds, val_ds = get_image_ds()
    gen = build_image_generator()
    crit = build_image_critic()
    gen_optimizer = tf.optimizers.RMSprop(GEN_LR)
    crit_optimizer = tf.optimizers.RMSprop(CRITIC_LR)

    gan = Image_Recovery_GAN(gen, crit)
    gan.compile(gen_optimizer, crit_optimizer, generator_loss_fn, critic_loss_fn)
    gan.fit(train_ds, val_ds, steps=EPOCHS)

    gan.save_model("./model_saves/image_gan")
    print("\n!!!!! DONE !!!!!!")