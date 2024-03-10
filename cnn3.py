from tensorflow.python.keras import models,layers
#from tensorflow.python.layers import layers

import pre_processing as pp

cnn_input = pp.main()


height = 128  # height of the image
width = 128   # width of the image
channels = 3  # number of color channels

# Define the number of points in your convex hull, and thus, the output size
num_points_in_convex_hull = 5  # for example, if you expect up to 5 points in the convex hull
output_size = num_points_in_convex_hull * 2  # times 2 for x and y coordinates

# CNN Model
def create_cnn_model(height, width, channels, output_size):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
    # ... other layers ...
    model.add(layers.Flatten())  # Flattening the output of the convolutional layers
    model.add(layers.Dense(output_size, activation='linear'))  # Output layer for regression

    model.compile(optimizer='adam', loss='mse')
    return model


# MDP Framework
class ConvexHullMDP:

    def _init_(self, cnn_model, num_green_dots):
        self.cnn_model = cnn_model
        self.num_green_dots = num_green_dots
        # ... other initialization ...

    def step(self, action):
        # Modify the convex hull based on the action (add or remove a green dot)
        # ...
        # Predict with the CNN which green dots are inside the updated convex hull
        new_state = self.apply_action(action)  # You need to implement this method

        predictions = self.cnn_model.predict(cnn_input[1])

        # Convert predictions to binary vector
        inside_convex_hull = predictions > 0.5  # Threshold can be adjusted

        # Calculate reward, possibly based on the coverage of green dots inside the convex hull
        reward = self.calculate_reward(inside_convex_hull)
        done = False #self.check_if_done(new_state)  # Replace with actual condition
        return new_state, reward, done

    def calculate_reward(self, inside_convex_hull):
        # Define how to calculate reward. This could be based on the number of dots inside
        # the convex hull and possibly their corresponding heat values.
        # ...
        reward = None  # Replace with actual reward calculation
        return reward

initial_state = None  # Replace with actual initial state
def policy(state):
    action = None  # Replace with actual policy logic to determine the next action
    return action

# Example usage within an MDP loop
num_green_dots = output_size // 2 # ... set the number of green dots ...
mdp = ConvexHullMDP(cnn_input, num_green_dots)
state = mdp.initial_state()
done = False

while not done:
    action = policy(state)  # Define your policy function based on your MDP strategy
    state, reward, done = mdp.step(action)
    # ... learning/updating policy ...