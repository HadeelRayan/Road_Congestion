import tensorflow as tf
from tensorflow.keras import layers, models
import preprocessImage as pp

input_shape = (128, 128, 3)  # Height = 128, Width = 128, Channels = 3
cnn_input = pp.main()
# Create the model
# Initialize the Sequential model
model = models.Sequential()

# 1. Convolutional Layer
# This layer will extract features from the input image using 32 filters/kernels of size 3x3.
# The 'relu' activation function adds non-linearity, allowing the network to learn more complex patterns.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# 2. Max Pooling Layer
# This layer reduces the spatial dimensions (width, height) of the output from the previous layer.
# It helps to reduce the computation required and also helps to extract the dominant features.
model.add(layers.MaxPooling2D((2, 2)))

# 3. Convolutional Layer
# Adding a second convolutional layer with 64 filters/kernels helps the network to learn more complex features.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 4. Max Pooling Layer
# Another max pooling layer to further reduce the dimensionality of the feature maps.
model.add(layers.MaxPooling2D((2, 2)))

# 5. Convolutional Layer
# A third convolutional layer with more filters to allow the network to learn even more complex features.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 6. Flatten Layer
# This layer flattens the output from the convolutional layers to form a single long feature vector.
model.add(layers.Flatten())

# 7. Dense Layer (Fully Connected Layer)
# A dense layer with 64 units is used to perform classification on the features extracted by the convolutional layers.
# The 'relu' activation function is used here as well.
model.add(layers.Dense(64, activation='relu'))

# 8. Output Layer
# The final layer is a dense layer with as many units as there are green dots. This example assumes we have 10 green dots.
# The 'sigmoid' activation function is used to output a probability for each green dot being inside the convex hull.
model.add(layers.Dense(10, activation='sigmoid'))

# Compile the model
# The model is compiled with the Adam optimizer and binary crossentropy loss, which is suitable for binary classification tasks.
# We are also tracking the accuracy during training.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Summary of the model
# The model summary function gives you an overview of the model architecture, the output shapes and the number of parameters.
model.summary()

#for prediction 
predictions = model.predict(cnn_input)
