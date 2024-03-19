import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tqdm import tqdm

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
model.compile(optimizer='adam', loss='mse')

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = load_img(image_path)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    return img_array

# Define the directories
lr_folder = 'lr_image'
hr_folder = 'hr_image'

# Create the hr_image folder if it doesn't exist
if not os.path.exists(hr_folder):
    os.makedirs(hr_folder)

# Get the list of image files in lr_image folder
image_files = os.listdir(lr_folder)

# Iterate through the images and perform super-resolution
for image_file in tqdm(image_files, desc='Processing Images'):
    lr_image_path = os.path.join(lr_folder, image_file)
    
    # Load and preprocess the low-resolution image
    lr_img_array = load_and_preprocess_image(lr_image_path)
    
    # Expand dimensions to match the model input shape
    lr_img_array = lr_img_array.reshape(1, lr_img_array.shape[0], lr_img_array.shape[1], lr_img_array.shape[2])

    # Generate high-resolution image
    hr_img_array = model.predict(lr_img_array)
    
    # Reshape and save the high-resolution image
    hr_img_array = hr_img_array.reshape(hr_img_array.shape[1], hr_img_array.shape[2], hr_img_array.shape[3])
    hr_img = array_to_img(hr_img_array * 255.0)  # De-normalize before saving
    hr_img.save(os.path.join(hr_folder, image_file))
