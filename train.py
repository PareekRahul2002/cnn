import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Define the model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(None, None, 3)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer=Adam(), loss='mse')

# Define the directories
lr_folder = 'lr_image'
hr_folder = 'hr_image'

# Load and preprocess the dataset
X = []
y = []

for image_file in os.listdir(lr_folder):
    # Load and preprocess the low-resolution image
    lr_img_array = load_and_preprocess_image(os.path.join(lr_folder, image_file))
    
    # Load and preprocess the corresponding high-resolution image
    hr_img_array = load_and_preprocess_image(os.path.join(hr_folder, image_file))
    
    X.append(lr_img_array)
    y.append(hr_img_array)

X = np.array(X)
y = np.array(y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define callbacks, including model checkpoint to save the best model during training
checkpoint = ModelCheckpoint('super_resolution_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, callbacks=[checkpoint])

# Optionally, you can save the final trained model
model.save('final_super_resolution_model.h5')
