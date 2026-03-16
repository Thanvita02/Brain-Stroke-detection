import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare dataset
train_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    "dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_generator, epochs=5)

# Save model
model.save("stroke_model.h5")

print("Model training completed")