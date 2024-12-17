import sys
import cv2
import gymnasium
import numpy as np
from numpy.random import default_rng
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


class PIEnv(gymnasium.Env):

    def __init__(self, map, clean=None, regularizer=1e-6):
        # list of current vertices in the convex hull, key=id, value=bool
        self.convexhull = {}
        # list of all intersections key=idm value=(x,y)
        self.intersection_dict = {}
        # list of all intersection and whether they are part of the convex hull key=id, value=bool
        self.intersection_state_dict = {}
        self.state = None
        # regularizer for the reward - proportional to the size of the area added
        self.regularizer = regularizer
        # To track previously toggled actions
        self.action_history = set()

        image = cv2.imread(map)
        if clean is not None:
            self.clean = cv2.imread(clean)
        else:
            self.clean = image.copy()

            # Process green dots
            self._process_green_dots(image)

            # Initialize action space
            self.action_space = gymnasium.spaces.Discrete(len(self.intersection_dict))

            # Define the observation space
            self.observation_space = gymnasium.spaces.Box(
                low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
            )

        # Generate heatmap
        self.heat_map = self._get_heatmap()

        # update all intersections to be false - corresponding to empty convex hull
        for key, val in self.intersection_dict.items():
            self.intersection_state_dict[key] = False
        
        # action space is a scalar integer representing the index of an intersection
        self.action_space = gymnasium.spaces.Discrete(len(self.intersection_dict))

    def _process_green_dots(self, image):
        """
        Detects green dots in the image and stores their centroids.
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define HSV thresholds for green dots
        lower_green = np.array([35, 150, 20])
        upper_green = np.array([85, 255, 120])

        # Create a mask for green regions
        mask_dark_green = cv2.inRange(image_hsv, lower_green, upper_green)

        # Find contours of green regions
        contours_dark_green, _ = cv2.findContours(mask_dark_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract centroids of green dots
        index = 0
        for contour in contours_dark_green:
            if cv2.contourArea(contour) > 5:  # Filter small noise
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    self.intersection_dict[index] = (cX, cY)
                    self.intersection_state_dict[index] = False
                    index += 1

    def reset(self, seed=None):
        """
        Resets the environment, generating a random initial convex hull.
        :return:
        Heatmap,
        Current convex hull mask,
        Dot indicators.
        """
        self.action_history.clear()  # Clear action history
        # reset all dictionaries
        self.convexhull = {}
        self.intersection_state_dict = dict.fromkeys(self.intersection_state_dict, False)

        #generate a random convexhull to start from
        if seed is None:
            rng = default_rng()
        else:
            rng = default_rng(seed=seed)
        numbers = rng.choice(len(self.intersection_dict), size=10, replace=False)

        self.convexhull = {}
        for num in numbers:
            self.convexhull[num] = True
            self.intersection_state_dict[num] = True

        # generate the appropriate img state, which is a stack of Heatmap, Current convex hull mask, Dot indicators
        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        """
        Updates the convex hull based on the selected action.
        :return:
        Next state,
        Reward,
        Indicators for episode termination or truncation (both False here).
        """

        # Check if the action has already been toggled
        if action in self.action_history:
            reward = -0.1  # Light penalty for redundant actions
            info = {"message": "Repeated action - ignored"}
            return self.state, reward, False, False, info

        # Save the previous convex hull
        prev_convex_hull = self.convexhull.copy()

        # Toggle the intersection state and add the action to history
        if self.intersection_state_dict[action]:
            self.intersection_state_dict[action] = False
            del self.convexhull[action]
        else:
            self.intersection_state_dict[action] = True
            self.convexhull[action] = True

        self.action_history.add(action)  # Track toggled actions

        # Generate the new state and calculate reward
        next_state = self._get_state()
        reward = self._get_reward(prev_convex_hull)

        # Update the environment state
        self.state = next_state

        # Set termination and truncation flags to False
        terminated = False
        truncated = False

        # Always return `info` as a dictionary, even if it's empty
        info = {"message": "Action accepted"}
        #print(f"Action History: {self.action_history}")

        return next_state, reward, terminated, truncated, info

    def render(self):
        """
        Renders a viewable image of the city section with the current convex hull
        :return:
        The original clean image in RGB, and the contour of the convex hull drawn on it
        """
        # Include all green dots in the convex hull calculation
        vertices = list(self.intersection_dict.values())
        new_map = self.clean.copy()

        # draw convex hull
        convex_hull = cv2.convexHull(np.array(vertices))
        cv2.drawContours(new_map, [convex_hull], -1, (128, 0, 128), 2)

        return new_map

    def render_with_vertices(self):
        """
        Renders a viewable image of the city section with the current convex hull and the intersections
        :return:
        The original clean image in RGB, and the contour of the convex hull drawn on it.
        Additionally, all the intersections are drawn as circles, white circles represent active intersections,
        and black circles represents inactive intersections.
        """
        vertices = []
        for vertex_id, state in self.convexhull.items():
            if state:
                vertices.append(self.intersection_dict[vertex_id])

        if len(vertices) < 3:
            print("Not enough points to create a convex hull")
            return self.clean.copy()

        new_map = self.clean.copy()

        # draw convex hull
        convex_hull = cv2.convexHull(np.array(vertices))
        cv2.drawContours(new_map, [convex_hull], -1, (128, 0, 128), 2)

        # draw intersection centroids with colors
        for int_id, cords in self.intersection_dict.items():
            if self.intersection_state_dict[int_id]:
                cv2.circle(new_map, cords, 5, (255, 255, 255), -1)
            else:
                cv2.circle(new_map, cords, 5, (0, 0, 0), -1)

        return new_map

    def show(self, img):
        """
        Displays images for visualization using OpenCV
        """
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _get_state(self):
        """
        Generates the state of the system
        :return:
        A 2 channel image, first channel the grayscale heat map, second channel the FILLED convex hull
        """
        ch_channel = np.zeros(self.heat_map.shape, np.float32)
        heatmap_channel = self.heat_map.copy()
        dots_channel = np.zeros(self.heat_map.shape, np.float32)

        # Generate the convex hull mask
        vertices = list(self.intersection_dict.values())

        ### ADDED #################
        #images but may have unintended values. normalize them between 0 and 1
        if len(vertices) > 2:
            convex_hull = cv2.convexHull(np.array(vertices))
            cv2.drawContours(ch_channel, [convex_hull], -1, color=1.0, thickness=cv2.FILLED)

        """
        for vertex_id, state in self.convexhull.items():
            if state:
                vertices.append(self.intersection_dict[vertex_id])
        convex_hull = cv2.convexHull(np.array(vertices))
        cv2.drawContours(ch_channel, [convex_hull], -1, color=1.0, thickness=cv2.FILLED)
        """

        # Generate the dots mask
        for int_id, cords in self.intersection_dict.items():
            if self.intersection_state_dict[int_id]:
                cv2.circle(dots_channel, cords, 5, color=1.0, thickness=-1)
            else:
                cv2.circle(dots_channel, cords, 5, color=-1.0, thickness=-1)

        ### ADDED #################
        # Resize channels to (84, 84)
        ch_channel_resized = cv2.resize(ch_channel, (84, 84), interpolation=cv2.INTER_AREA)
        heatmap_channel_resized = cv2.resize(heatmap_channel, (84, 84), interpolation=cv2.INTER_AREA)
        dots_channel_resized = cv2.resize(dots_channel, (84, 84), interpolation=cv2.INTER_AREA)

        # Stack resized channels
        state = np.stack((dots_channel_resized, heatmap_channel_resized, ch_channel_resized), axis=0)
        # Scale to [0, 255] and convert to uint8
        state = (state * 255).clip(0, 255).astype(np.uint8)
        return state

    def _get_reward(self, prev_convexhull):
        """
        Compares previous and current convex hulls.
        Computes reward based on: Heatmap values under new hull regions, Penalization for hull size.
        """
        added_area = len(self.convexhull) - len(prev_convexhull)
        reward = 0.0

        if added_area > 0:
            reward = added_area  # Reward proportional to area added
        else:
            reward -= 0.5  # Penalize insignificant changes

        return reward

        """
        # generate old convex hull
        vertices = []
        for vertex_id, state in prev_convexhull.items():
            if state:
                vertices.append(self.intersection_dict[vertex_id])
        convex_hull = cv2.convexHull(np.array(vertices))

        # generate the mask of the old convex hull
        prev_mask = np.zeros(self.clean.shape, np.uint8)
        cv2.drawContours(prev_mask, [convex_hull], -1, color=(255, 255, 255), thickness=cv2.FILLED)
        prev_mask = cv2.cvtColor(prev_mask, cv2.COLOR_BGR2GRAY)

        # generate new convex hull
        vertices = []
        for vertex_id, state in self.convexhull.items():
            if state:
                vertices.append(self.intersection_dict[vertex_id])
        convex_hull = cv2.convexHull(np.array(vertices))

        # generate the mask of the new convex hull
        curr_mask = np.zeros(self.clean.shape, np.uint8)
        cv2.drawContours(curr_mask, [convex_hull], -1, color=(255, 255, 255), thickness=cv2.FILLED)
        curr_mask = cv2.cvtColor(curr_mask, cv2.COLOR_BGR2GRAY)

        # generate a difference map between the two convex hulls
        diff = (curr_mask.astype(float) - prev_mask.astype(float)) / 255

        #calc reward: sum values / count values * sign - added count values * reg
        count_v = np.count_nonzero(diff)
        reward = 0
        if count_v > 0:
            sum_v = np.sum(self.heat_map * diff).astype(float)
            reward = sum_v / count_v - (self.regularizer * count_v) * np.sign(sum_v)

        return reward"""

    def _get_heatmap(self):
        """
        Processes the map to highlight "red zones" based on HSV and grayscale thresholds.
        """
        image_hsv = cv2.cvtColor(self.clean, cv2.COLOR_BGR2HSV)
        image_gray = cv2.cvtColor(self.clean, cv2.COLOR_BGR2GRAY)

        # Define thresholds for red color HSV
        lower_red = np.array([int(000 * 255 / 360), 100, 00])
        upper_red = np.array([int(360 * 255 / 360), 250, 255])

        # Create a binary mask of red areas HSV
        mask_red = cv2.inRange(image_hsv, lower_red, upper_red)

        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask_red.shape, np.uint8)
        mask.fill(0)
        for c in contours_red:
            M = cv2.moments(c)
            if M["m00"] < 500:
                continue
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

        heat_map = mask * image_gray
        heat_map = heat_map.astype(np.float32) / np.max(heat_map)
        return heat_map


if __name__ == "__main__":
    # Initialize the environment
    env = PIEnv(map="map_image.png")
    check_env(env)  # Check the environment for compatibility
    env = make_vec_env(lambda: env, n_envs=1)

    #model = DQN("CnnPolicy", env, verbose=1, buffer_size=50, normalize_images=False)
    #model = DQN("MlpPolicy", env, verbose=1, buffer_size=5000)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=5000,
        exploration_initial_eps=1.0,  # Start with full exploration
        exploration_final_eps=0.1,  # End with low exploration
        exploration_fraction=0.2  # Fraction of training spent decaying epsilon
    )
    model.learn(total_timesteps=50000)

    obs = env.reset()
    for step in range(10):
        #action, _states = model.predict(obs, deterministic=True)
        # Allow a small chance for random exploration during testing
        if np.random.rand() < 0.1:  # 10% chance of a random action
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs, deterministic=True)

        # Ensure action is scalar
        if isinstance(action, (list, np.ndarray)):
            action = int(action[0])  # Extract scalar action

        # Step through the environment
        obs, reward, done, info = env.step(action)

        # Safely extract the first environment's info
        if isinstance(info, list) and len(info) > 0:
            info_message = info[0].get('message', 'No message')
        else:
            info_message = 'No message'

        # Each step represents a single interaction with the environment.
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        #print(f"  Info: {info[0]['message']}")
        # Safely extract the first environment's info
        if isinstance(info, list) and len(info) > 0:
            info_message = info[0].get('message', 'No message')
        else:
            info_message = 'No message'

        print(f"  Info: {info_message}")

        if done:
            obs = env.reset()

    rendered_map = env.envs[0].env.render_with_vertices()
    cv2.imwrite("rendered_map.png", rendered_map)
    # Call the show method from the original environment
    env.envs[0].env.show(rendered_map)
    env.close()

    print("Simulation completed.")
