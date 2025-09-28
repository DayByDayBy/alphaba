import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model # type: ignore

def create_base_network(input_shape=(105, 105, 1), embedding_dim=64):
    """Create the base CNN for feature extraction."""
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Use Input layer instead of input_shape
        layers.Conv2D(64, (10, 10), activation='relu'),
        layers.BatchNormalization(),  # Add batch norm for stability
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (7, 7), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (4, 4), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (4, 4), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),  # Smaller dense layer
        layers.BatchNormalization(),
        layers.Dense(embedding_dim, activation=None),  # No activation for embeddings
        # Remove L2 normalization during training - apply it only for evaluation
    ])
    return model

def create_triplet_model(embedding_dim=64):
    """Create the full triplet network with shared weights."""
    # shared base network
    base_network = create_base_network(embedding_dim=embedding_dim)

    # 3 inputs
    anchor_input = layers.Input(shape=(105, 105, 1), name='anchor')
    positive_input = layers.Input(shape=(105, 105, 1), name='positive')
    negative_input = layers.Input(shape=(105, 105, 1), name='negative')
    
    # get embeddings using shared weights
    anchor_embedding = base_network(anchor_input)
    positive_embedding = base_network(positive_input)
    negative_embedding = base_network(negative_input)
    
    # create model
    model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding]
    )
    
    return model, base_network

def triplet_loss(margin=0.2):
    """custom triplet loss function."""
    @tf.function
    def loss(y_true, y_pred):
        
        anchor = y_pred[0]
        positive = y_pred[1]
        negative = y_pred[2]
        
        # calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1) 
        
        # triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = tf.maximum(0.0, pos_dist - neg_dist + margin)
        return tf.reduce_mean(loss)
    return loss

def normalize_embeddings(embeddings):
    """Apply L2 normalization to embeddings for evaluation."""
    return tf.nn.l2_normalize(embeddings, axis=1)




