#!/usr/bin/env python3
"""Defines the Simple_GAN class, a basic Generative Adversarial Network
built on top of keras.Model.
"""
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """A simple Generative Adversarial Network.

    The model owns a generator, a discriminator, a latent vector
    generator, and a set of real examples. Training alternates
    between several gradient descent steps on the discriminator and
    a single gradient descent step on the generator.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the GAN.

        Args:
            generator: a keras.Model mapping latent vectors to fake
                examples.
            discriminator: a keras.Model mapping examples (real or
                fake) to a scalar score.
            latent_generator: a callable that takes an integer k and
                returns a batch of k latent vectors.
            real_examples: a tensor containing the real data set.
            batch_size: the number of examples used per training
                step.
            disc_iter: the number of discriminator updates performed
                per generator update.
            learning_rate: the learning rate used by both optimizers.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # define the generator loss and optimizer
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer, loss=generator.loss)

        # define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer, loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """Generate a batch of fake examples.

        Args:
            size: the number of fake examples to generate. Defaults
                to self.batch_size.
            training: whether the generator is called in training
                mode.

        Returns:
            A tensor of fake examples produced by the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Generate a batch of real examples.

        Args:
            size: the number of real examples to sample. Defaults to
                self.batch_size.

        Returns:
            A tensor containing a random subset of self.real_examples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """Perform one training step of the GAN.

        The discriminator is updated self.disc_iter times on
        real/fake batches, then the generator is updated once on a
        fresh fake batch.

        Args:
            useless_argument: unused, required by the keras.Model
                training loop signature.

        Returns:
            A dictionary with the latest discriminator loss
            ("discr_loss") and generator loss ("gen_loss").
        """
        for _ in range(self.disc_iter):
            # compute the loss for the discriminator in a tape
            # watching the discriminator's weights
            with tf.GradientTape() as discr_tape:
                # get a real sample
                real_sample = self.get_real_sample()
                # get a fake sample
                fake_sample = self.get_fake_sample(training=False)

                # compute the loss discr_loss of the discriminator on
                # real and fake samples
                discr_loss = self.discriminator.loss(
                    self.discriminator(real_sample, training=True),
                    self.discriminator(fake_sample, training=True))

            # apply gradient descent once to the discriminator
            discr_gradients = discr_tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients,
                    self.discriminator.trainable_variables))

        # compute the loss for the generator in a tape watching the
        # generator's weights
        with tf.GradientTape() as gen_tape:
            # get a fake sample
            fake_sample = self.get_fake_sample(training=True)

            # compute the loss gen_loss of the generator on this
            # sample
            gen_loss = self.generator.loss(
                self.discriminator(fake_sample, training=False))

        # apply gradient descent to the generator
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
