"""
Data augmentation for character images
Improves training robustness and generalization
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import random


class CharacterAugmentation:
    """Data augmentation for character images"""
    
    def __init__(self, rotation_range=15.0, zoom_range=0.1, noise_std=0.01, 
                 elastic_alpha=1.0, elastic_sigma=8.0):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.noise_std = noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT_101)
        return rotated
    
    def zoom_image(self, image, zoom_factor):
        """Apply zoom to image"""
        h, w = image.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Resize and crop back to original size
        if zoom_factor > 1.0:
            # Zoom in - crop center
            resized = cv2.resize(image, (new_w, new_h))
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            zoomed = resized[start_y:start_y+h, start_x:start_x+w]
        else:
            # Zoom out - pad with reflection
            resized = cv2.resize(image, (new_w, new_h))
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            zoomed = cv2.copyMakeBorder(resized, pad_y, h-new_h-pad_y, 
                                       pad_x, w-new_w-pad_x, 
                                       cv2.BORDER_REFLECT_101)
        
        return zoomed
    
    def add_noise(self, image):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, self.noise_std, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def elastic_transform(self, image):
        """Apply elastic transformation to image"""
        if self.elastic_alpha == 0:
            return image
        
        h, w = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.randn(h, w) * self.elastic_sigma
        dy = np.random.randn(h, w) * self.elastic_sigma
        
        # Smooth displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), self.elastic_sigma) * self.elastic_alpha
        dy = cv2.GaussianBlur(dy, (0, 0), self.elastic_sigma) * self.elastic_alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply transformation
        transformed = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_REFLECT_101)
        
        return transformed
    
    def augment_image(self, image):
        """Apply random augmentations to image"""
        augmented = image.copy()
        
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            augmented = self.rotate_image(augmented, angle)
        
        # Random zoom
        if self.zoom_range > 0:
            zoom_factor = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            augmented = self.zoom_image(augmented, zoom_factor)
        
        # Elastic transformation
        if random.random() < 0.3:  # Apply 30% of the time
            augmented = self.elastic_transform(augmented)
        
        # Add noise
        augmented = self.add_noise(augmented)
        
        return augmented


class TensorFlowAugmentation:
    """TensorFlow-based augmentation for GPU acceleration"""
    
    def __init__(self, rotation_range=0.1, zoom_range=0.1, noise_std=0.01):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.noise_std = noise_std
    
    def augment_batch(self, images):
        """Apply augmentations to a batch of images"""
        # Convert to TensorFlow operations
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        
        # Random rotation
        if self.rotation_range > 0:
            angles = tf.random.uniform([tf.shape(images)[0]], 
                                     -self.rotation_range, self.rotation_range)
            images = tf.image.rot90(images, k=tf.cast(angles / (np.pi/2), tf.int32))
        
        # Random zoom
        if self.zoom_range > 0:
            scales = tf.random.uniform([tf.shape(images)[0]], 
                                     1 - self.zoom_range, 1 + self.zoom_range)
            
            # Apply zoom to each image in batch
            augmented_images = []
            for i in range(tf.shape(images)[0]):
                img = images[i]
                scale = scales[i]
                
                h, w = tf.shape(img)[0], tf.shape(img)[1]
                new_h, new_w = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32), \
                              tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
                
                img_resized = tf.image.resize(img[tf.newaxis, ...], [new_h, new_w])[0]
                
                if scale > 1.0:
                    # Crop center
                    start_y = (new_h - h) // 2
                    start_x = (new_w - w) // 2
                    img_cropped = img_resized[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Pad
                    pad_y = (h - new_h) // 2
                    pad_x = (w - new_w) // 2
                    img_padded = tf.image.pad_to_bounding_box(img_resized, pad_y, pad_x, h, w)
                    img_cropped = img_padded
                
                augmented_images.append(img_cropped)
            
            images = tf.stack(augmented_images)
        
        # Add noise
        if self.noise_std > 0:
            noise = tf.random.normal(tf.shape(images), 0, self.noise_std)
            images = images + noise
            images = tf.clip_by_value(images, 0, 1)
        
        return images.numpy()


def create_augmented_data_loader(data_loader, augmentation_config=None):
    """Create data loader with augmentation"""
    
    if augmentation_config is None:
        augmentation_config = {
            'rotation_range': 15.0,
            'zoom_range': 0.1,
            'noise_std': 0.01,
            'elastic_alpha': 1.0,
            'elastic_sigma': 8.0
        }
    
    augmenter = CharacterAugmentation(**augmentation_config)
    
    class AugmentedDataLoader:
        def __init__(self, original_loader, augmenter):
            self.original_loader = original_loader
            self.augmenter = augmenter
            self.alphabet_data = original_loader.alphabet_data
            self.alphabet_names = original_loader.alphabet_names
        
        def sample_triplet(self, augment=True):
            """Sample triplet with optional augmentation"""
            anchor_img, positive_img, negative_img = self.original_loader.sample_triplet()
            
            if augment:
                anchor_img = self.augmenter.augment_image(anchor_img)
                positive_img = self.augmenter.augment_image(positive_img)
                negative_img = self.augmenter.augment_image(negative_img)
            
            return anchor_img, positive_img, negative_img
        
        def generate_batch(self, batch_size, augment=True):
            """Generate batch with optional augmentation"""
            anchors, positives, negatives = [], [], []
            
            for _ in range(batch_size):
                anchor, positive, negative = self.sample_triplet(augment)
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
            
            return (np.array(anchors), np.array(positives), np.array(negatives))
    
    return AugmentedDataLoader(data_loader, augmenter)


# Keras layer for on-the-fly augmentation
class AugmentationLayer(layers.Layer):
    """Keras layer for real-time augmentation"""
    
    def __init__(self, rotation_range=0.1, zoom_range=0.1, noise_std=0.01, **kwargs):
        super().__init__(**kwargs)
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.noise_std = noise_std
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # Apply random augmentations
        augmented = inputs
        
        # Random rotation (using rotation matrix)
        if self.rotation_range > 0:
            angle = tf.random.uniform([], -self.rotation_range, self.rotation_range)
            angle_rad = angle * np.pi / 180.0
            
            # Create rotation matrix
            cos_a = tf.cos(angle_rad)
            sin_a = tf.sin(angle_rad)
            rotation_matrix = tf.reshape([cos_a, -sin_a, sin_a, cos_a], [2, 2])
            
            # Apply rotation (simplified for square images)
            # Note: This is a simplified rotation - for production use tf.contrib.image.rotate
            augmented = tf.image.rot90(augmented, k=tf.cast(angle / 90, tf.int32))
        
        # Random zoom and crop
        if self.zoom_range > 0:
            scale = tf.random.uniform([], 1 - self.zoom_range, 1 + self.zoom_range)
            original_shape = tf.shape(augmented)
            new_shape = tf.cast(tf.cast(original_shape[:2], tf.float32) * scale, tf.int32)
            
            augmented = tf.image.resize(augmented, new_shape)
            augmented = tf.image.resize_with_crop_or_pad(augmented, original_shape[0], original_shape[1])
        
        # Add noise
        if self.noise_std > 0:
            noise = tf.random.normal(tf.shape(augmented), 0, self.noise_std)
            augmented = augmented + noise
            augmented = tf.clip_by_value(augmented, 0, 1)
        
        return augmented
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'rotation_range': self.rotation_range,
            'zoom_range': self.zoom_range,
            'noise_std': self.noise_std,
        })
        return config
