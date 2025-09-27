import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


def evaluate_embedding(model, test_data):
    pass