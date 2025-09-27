import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



def train_triplet_model_custom(triplet_model, data_loader, epochs=10, batch_size=32, steps_per_epoch=100, learning_rate=0.01):
    """train triplet model with custom training loop"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    history = {'loss':[]}
    
    @tf.function
    def train_step(anchors, positives, negatives):
        with tf.GradientTape() as tape:
            anchor_emb, positive_emb, negative_emb = triplet_model([anchors, positives, negatives], training=True)

            pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)

            loss = tf.reduce_mean(tf.maximum(0.0, pos_dist - neg_dist + 0.5))  #  margin was .2, now .5

        gradients = tape.gradient(loss, triplet_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, triplet_model.trainable_variables))

        return loss
    
    for epoch in range(epochs):
        epoch_losses = []
        print(f"Epoch {epoch+1}/{epochs}")
        
        for step in range(steps_per_epoch):

            anchors, positives, negatives = data_loader.generate_batch(batch_size)

            anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
            positives = tf.convert_to_tensor(positives, dtype=tf.float32)
            negatives = tf.convert_to_tensor(negatives, dtype=tf.float32)

            loss = train_step(anchors, positives, negatives)

            epoch_losses.append(float(loss))

            if step % 20 == 0:
                print(f" Step {step}/{steps_per_epoch} - loss: {loss.numpy():.4f}")

                # debug: Check model weights are changing (first epoch only)
                if epoch == 0 and step == 0:
                    print("Debug - Checking if model weights change...")
                
        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)
        print(f" Epoch {epoch+1} - avg loss: {avg_loss:.4f}")
        print("-" * 50)
    return history


def compile_triplet_model(model, learning_rate=0.001):
    """compile triplet model with custom loss"""
    from src.models import triplet_loss
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=triplet_loss())
    return model
    
    
def train_triplet_model(model, data_loader, epochs=10, batch_size=32, steps_per_epoch=100):
    """train triplet model"""
    history = {'loss':[]}
    
    for epoch in range(epochs):
        # print(f"Epoch {epoch+1}/{epochs}")
        epoch_losses = []
        print(f"Epoch {epoch+1}/{epochs}")
        
        for step in range(steps_per_epoch):
            anchors, positives, negatives = data_loader.generate_batch(batch_size)
            
            loss = model.train_on_batch(
                [anchors, positives, negatives],
                [np.zeros((batch_size, 128)), np.zeros((batch_size, 128)), np.zeros((batch_size, 128))]
            )
            epoch_losses.append(loss)
            
            
            if step % 20 == 0:
                print(f" Step {step}/{steps_per_epoch} - loss: {loss:.4f}")
                
        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)
        print(f" Epoch {epoch+1} - avg loss: {avg_loss:.4f}")
        print("-" * 50)
    return history


def visualize_training(history):
    """plotting training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.title('training loss')
    plt.xlabel('epoch')
    plt.ylabel('triplet loss')
    plt.grid(True)
    plt.show()


def evaluate_embeddings(model,data_loader, num_samples=500):
    """evaluate embedding with t-SNE visualization"""
    
    base_network = model.layers[3]
    
    images, labels = [],[]
    for alphabet in data_loader.alphabet_names[:10]:
        alphabet_data = data_loader.alphabet_data[alphabet]
        samples = np.random.choice(len(alphabet_data), min(50, len(alphabet_data)), replace=False)
        for idx in samples:
            images.append(alphabet_data[idx][0])
            labels.append(alphabet)
    
    images = np.array(images)
    embeddings = base_network.predict(images, verbose=0)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    unique_labels = list(set(labels))

    for i, alphabet in enumerate(unique_labels):
        mask = np.array(labels) == alphabet
        plt.scatter(reduced_embeddings_2d[mask, 0],
                     reduced_embeddings_2d[mask, 1],
                     c=plt.cm.tab10(i / len(unique_labels)),  # Use colormap to get actual color
                     label=alphabet,
                     alpha=0.7,
                     s=30)
                    
    
    
    
    plt.legend(box_to_anchor=(1.05, 1), loc='upper left')
    plt.title('learned char embeddings (t-SNE vis)')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()
    
    return embeddings, labels