# 3D CNN for Medical Image Segmentation

This project implements a 3D Convolutional Neural Network (CNN) for binary segmentation of volumetric medical images. The network ensures that the output segmentation map matches the input dimensions through the use of upsampling and transpose convolution layers.

## Overview

The model processes volumetric data with dimensions `(Depth, Height, Width, Channels)` and outputs a binary segmentation map with the same spatial dimensions as the input. The architecture uses downsampling for feature extraction and upsampling for reconstructing the spatial dimensions.

## Dataset
This project uses the **KITS19 Dataset**, a publicly available dataset for kidney tumor segmentation challenges. It includes:
- 3D volumetric CT scans of the abdomen.
- Ground truth segmentation masks for kidney and tumor regions.

The dataset is preprocessed to resize scans and masks to a consistent shape (e.g., 128x128x128).

To learn more about the KITS19 dataset, visit the [kits19](https://github.com/neheller/kits19).

## Requirements

Install the necessary dependencies using the following:

```bash
pip install tensorflow numpy scipy os SimpleITK
```

## Model Architecture

### Downsampling
- **Convolutional Layers**: Extract features.
- **MaxPooling3D**: Reduce spatial dimensions by half.

### Upsampling
- **Conv3DTranspose**: Restore feature maps.
- **UpSampling3D**: Double the spatial dimensions.

### Final Layer
- **Conv3D**: Produces a single-channel output for binary segmentation.

## Code Example

```python
from tensorflow.keras import models, layers

def create_3d_cnn(input_shape):
    model = models.Sequential()

    # Downsampling
    model.add(layers.Conv3D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.Conv3D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=2))
    model.add(layers.Conv3D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=2))

    # Upsampling
    model.add(layers.Conv3DTranspose(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.UpSampling3D(size=2))
    model.add(layers.Conv3DTranspose(64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.UpSampling3D(size=2))
    model.add(layers.Conv3DTranspose(32, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.UpSampling3D(size=2))

    # Output layer
    model.add(layers.Conv3D(1, kernel_size=1, activation='sigmoid', padding='same'))

    return model
```

## Training the Model

1. **Prepare the Data**:
   Ensure that input data has dimensions `(Batch, Depth, Height, Width, Channels)`.
   ```python
   y_train = y_train[..., np.newaxis]  # Add channel dimension if missing
   y_val = y_val[..., np.newaxis]
   y_test = y_test[..., np.newaxis]
   ```

2. **Compile and Train**:
   ```python
   model = create_3d_cnn(input_shape=(128, 128, 128, 1))
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   history = model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=20,
       batch_size=4,
       callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
   )
   ```

## Output Verification

To verify the output shape:
```python
print("Model output shape:", model.output_shape)
print("y_train shape:", y_train.shape)
```

## Notes
- Ensure target (`y_train`) and output shapes match.
- The upsampling ensures the output dimensions are restored to the input dimensions.
