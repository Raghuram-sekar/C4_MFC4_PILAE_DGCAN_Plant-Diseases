import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
from data_loader import load_dataset
from dcgan import make_generator_model, make_discriminator_model

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 15
noise_dim = 100
num_examples_to_generate = 16

# Paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_path, "Dataset")
checkpoint_dir = os.path.join(base_path, "models", "training_checkpoints")
images_dir = os.path.join(base_path, "models", "generated_images")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Checkpoints
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Seed for visualization
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    
    # Rescale to [0, 1] for plotting
    predictions = (predictions + 1) / 2.0

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')

    plt.savefig(os.path.join(images_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()

def train(dataset, epochs):
    print("Starting training...")
    for epoch in range(epochs):
        start = time.time()
        
        gen_loss_avg = 0
        disc_loss_avg = 0
        batches = 0

        for image_batch, _ in dataset:
            g_loss, d_loss = train_step(image_batch)
            gen_loss_avg += g_loss
            disc_loss_avg += d_loss
            batches += 1

        # Produce images for the GIF
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = os.path.join(checkpoint_dir, "ckpt"))
        
        # Save the current discriminator for immediate testing
        discriminator.save(os.path.join(base_path, "models", "discriminator_current.h5"))

        print ('Time for epoch {} is {} sec. Gen Loss: {:.4f}, Disc Loss: {:.4f}'.format(epoch + 1, time.time()-start, gen_loss_avg/batches, disc_loss_avg/batches))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)
    
    # Save final models
    generator.save(os.path.join(base_path, "models", "generator_final.h5"))
    discriminator.save(os.path.join(base_path, "models", "discriminator_final.h5"))
    print("Training finished. Models saved.")

if __name__ == "__main__":
    try:
        # load_dataset now returns (ds, count, class_names)
        train_dataset, count, _ = load_dataset(dataset_dir, batch_size=BATCH_SIZE)
        train(train_dataset, EPOCHS)
    except Exception as e:
        print(f"Error during training: {e}")
