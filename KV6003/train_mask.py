from .model import GAN, build_mask_critic, build_mask_generator, generator_loss_fn, critic_loss_fn
from .dataset import get_mask_ds
from .variables import EPOCHS, GEN_LR, CRITIC_LR
import tensorflow as tf
from os import system


if __name__ == "__main__":
    system("cls")
    train_ds, val_ds = get_mask_ds()

    generator = build_mask_generator()
    critic = build_mask_critic()

    gen_optimizer = tf.optimizers.RMSprop(GEN_LR)
    crit_optimizer = tf.optimizers.RMSprop(CRITIC_LR)

    gan = GAN(generator, critic)
    gan.compile(gen_optimizer, crit_optimizer, generator_loss_fn, critic_loss_fn)
    gan.fit(train_ds, val_ds, steps=EPOCHS)

    gan.save_model("./model_saves/mask_gan")
    print("\n!!!!! DONE !!!!!!")

