import os
import cv2
import gymnasium
import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from datetime import datetime
from torch.cuda.amp import autocast
import logging
import json
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# Final training hyperparameters
LEARNING_RATE = 1e-4  # Reduced for more stable learning
BATCH_SIZE = 128  # Increased batch size
GAMMA = 0.95  # Balanced between immediate and future rewards
BUFFER_SIZE = 10000  # Keep current buffer size
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.998  # Slower decay for better exploration
TARGET_UPDATE_FREQ = 4
NUM_EPISODES = 500  # More episodes for better learning
MAX_STEPS = 10  # More steps per episode

# Training configuration
# Training configuration
training_config = {
    'num_episodes': 1000,      # Increased from 500
    'batch_size': 128,         # Keep same
    'debug': True,
    'checkpoint_freq': 20,     # Increased from 10
    'early_stop_patience': 150, # Increased from 100
    'max_steps': 15            # Increased from 10
}

class QNetwork(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(QNetwork, self).__init__()

        # Modified CNN architecture with smaller kernels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=2),  # Increased channels, smaller stride
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Added layer
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Calculate the flattened size
        self.flatten_size = self._get_conv_output_size(input_channels)

        # Enhanced fully connected layers with dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_dim)
        )
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    def _get_conv_output_size(self, input_channels):
        x = torch.randn(1, input_channels, 84, 84)
        x = self.conv_layers(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, state, action, reward, next_state, done):
        with self.lock:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        with self.lock:
            batch = random.sample(self.buffer, batch_size)

        # Process batch items in parallel
        with ThreadPoolExecutor() as executor:
            processed_batch = list(executor.map(self._process_batch_item, batch))

        # Combine processed items
        states, actions, rewards, next_states, dones = zip(*processed_batch)
        return (torch.cat(states), torch.LongTensor(actions),
                torch.FloatTensor(rewards), torch.cat(next_states),
                torch.FloatTensor(dones))

    @staticmethod
    def _process_batch_item(item):
        state, action, reward, next_state, done = item
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_channels, action_dim, lr=3e-4, gamma=0.99, buffer_size=10000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            torch.set_float32_matmul_precision('high')

        # Initialize networks
        self.q_net = QNetwork(input_channels, action_dim).to(self.device)
        self.target_net = QNetwork(input_channels, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Initialize scaler and optimizer
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        # Exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_END

        # Target network update parameters
        self.target_update_counter = 0
        self.target_update_freq = TARGET_UPDATE_FREQ

        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def select_action(self, state):
        with torch.amp.autocast(device_type=self.device.type):
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)

            state = state.to(self.device)
            with autocast():
                q_values = self.q_net(state)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        if self.scaler:  # GPU training
            with torch.cuda.amp.autocast():
                current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = self.target_net(next_states).max(1)[0]
                    target_q = rewards + (self.gamma * next_q * (1 - dones))
                loss = nn.functional.smooth_l1_loss(current_q, target_q)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:  # CPU training
            current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + (self.gamma * next_q * (1 - dones))
            loss = nn.functional.smooth_l1_loss(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()

    def sync_target_network(self):
        if self.target_update_counter >= self.target_update_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_update_counter = 0


class PIEnv(gymnasium.Env):
    def __init__(self, map, clean=None, regularizer=1e-8):
        self.convexhull = {}
        self.intersection_dict = {}
        self.intersection_state_dict = {}
        self.state = None
        self.regularizer = regularizer
        self.action_history = set()

        # Load and process the map
        image = cv2.imread(map)
        if clean is not None:
            self.clean = cv2.imread(clean)
        else:
            self.clean = image.copy()

        self._process_green_dots(image)
        self.action_space = gymnasium.spaces.Discrete(len(self.intersection_dict))
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )

        self.heat_map = self._get_heatmap()
        for key in self.intersection_dict:
            self.intersection_state_dict[key] = False

    def _process_green_dots(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 150, 20])
        upper_green = np.array([85, 255, 120])
        mask_dark_green = cv2.inRange(image_hsv, lower_green, upper_green)
        contours_dark_green, _ = cv2.findContours(mask_dark_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_dark_green) == 0:
            raise ValueError("No green dots detected in map image!")
        index = 0
        for contour in contours_dark_green:
            if cv2.contourArea(contour) > 5:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    self.intersection_dict[index] = (cX, cY)
                    self.intersection_state_dict[index] = False
                    index += 1

    def preprocess_traffic_zones(self):
        """
        First phase: Identify distinct high-traffic zones using density-based clustering.
        This helps break down the large area into manageable sections.
        """
        # Convert heatmap to normalized form for processing
        normalized_heatmap = (self.heat_map * 255).astype(np.uint8)

        # Find potential high-traffic zones using adaptive thresholding
        high_traffic_mask = cv2.adaptiveThreshold(
            normalized_heatmap,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size for local neighborhood
            2  # Constant subtracted from mean
        )

        # Find connected components in high traffic areas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            high_traffic_mask, connectivity=8
        )

        # Filter zones based on size and traffic intensity
        valid_zones = []
        for i in range(1, num_labels):  # Skip background (label 0)
            # Get zone properties
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Calculate average traffic intensity in this zone
            zone_mask = (labels == i).astype(np.uint8)
            zone_traffic = cv2.mean(self.heat_map, mask=zone_mask)[0]

            # Filter based on size and traffic intensity
            if 0.01 <= (area / (self.heat_map.shape[0] * self.heat_map.shape[1])) <= 0.1 and zone_traffic > 0.6:
                valid_zones.append({
                    'bbox': (x, y, w, h),
                    'mask': zone_mask,
                    'traffic': zone_traffic,
                    'points': []
                })

        # Assign intersection points to zones
        for idx, (x, y) in self.intersection_dict.items():
            for zone in valid_zones:
                bbox = zone['bbox']
                if (bbox[0] <= x <= bbox[0] + bbox[2] and
                        bbox[1] <= y <= bbox[1] + bbox[3] and
                        zone['mask'][y, x] > 0):
                    zone['points'].append(idx)
                    break

        return valid_zones

    def preprocess_high_traffic_zones(self):
        normalized_map = (self.heat_map * 255).astype(np.uint8)
        binary_map = cv2.threshold(normalized_map, 200, 255, cv2.THRESH_BINARY)[1]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_map, connectivity=8
        )

        high_traffic_zones = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 100 and area < 10000:  # Focus on smaller areas
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                if w < 100 and h < 100:  # Additional size constraint
                    high_traffic_zones.append({
                        'bbox': (x, y, w, h),
                        'center': (int(x + w / 2), int(y + h / 2))
                    })

        return high_traffic_zones

    def get_nearest_intersections(self, zone_center, k=3):
        """
        Find k nearest intersections to a zone center
        """
        distances = []
        for idx, (x, y) in self.intersection_dict.items():
            dist = np.sqrt(
                (x - zone_center[0]) ** 2 +
                (y - zone_center[1]) ** 2
            )
            distances.append((idx, dist))

        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[1])
        return [idx for idx, _ in distances[:k]]

    def reset(self, seed=None):
        """
        Enhanced reset focusing on traffic hotspots
        """
        self.action_history.clear()
        self.convexhull = {}
        self.intersection_state_dict = dict.fromkeys(self.intersection_state_dict, False)

        rng = default_rng(seed=seed)

        # Get all intersections with their traffic values
        traffic_points = []
        for idx, (x, y) in self.intersection_dict.items():
            traffic_value = self.heat_map[y, x]
            if traffic_value > 0.7:  # Only consider high traffic points
                traffic_points.append((idx, traffic_value))

        # Sort by traffic value
        traffic_points.sort(key=lambda x: x[1], reverse=True)

        # Select 3 initial points from top traffic points
        if len(traffic_points) > 0:
            num_init_points = min(3, len(traffic_points))
            initial_points = [point[0] for point in traffic_points[:num_init_points]]
        else:
            # Fallback to random selection
            initial_points = rng.choice(list(self.intersection_dict.keys()),
                                        size=min(3, len(self.intersection_dict)),
                                        replace=False)

        # Initialize the hull with these points
        for point in initial_points:
            self.convexhull[point] = self.intersection_dict[point]
            self.intersection_state_dict[point] = True

        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        """
        Modified step function focusing on building high-traffic zones
        """
        prev_convexhull = self.convexhull.copy()

        # Check if action is valid
        if self.intersection_state_dict[action]:
            return self.state, -5, True, False, {"message": "Repeated action"}

        # Update convex hull
        self.intersection_state_dict[action] = True
        self.convexhull[action] = self.intersection_dict[action]

        # Get vertices
        vertices = [self.intersection_dict[vertex_id]
                    for vertex_id, state in self.convexhull.items() if state]

        # Calculate distances between vertices
        distances = []
        if len(vertices) > 1:
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    dist = np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j]))
                    distances.append(dist)
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0

        # Calculate reward
        reward = self._get_reward(prev_convexhull)

        # Update state
        self.state = self._get_state()

        # Check termination conditions
        done = False
        if len(self.convexhull) >= 5:  # Too many points
            done = True
        elif len(vertices) >= 3:  # Check quality only if we have a valid hull
            # Calculate hull properties
            convex_hull = cv2.convexHull(np.array(vertices, dtype=np.float32))
            hull_area = cv2.contourArea(convex_hull)
            mask = np.zeros(self.clean.shape[:2], np.uint8)
            cv2.drawContours(mask, [np.array(convex_hull, dtype=np.int32)], -1, 255, -1)
            hull_mask = mask / 255.0
            traffic_sum = np.sum(self.heat_map * hull_mask)
            traffic_density = traffic_sum / (hull_area + 1e-6)

            # Terminate if we found a good solution
            if (traffic_density > 0.9 and  # High traffic
                    hull_area < 10000 and  # Compact
                    avg_distance < 60):  # Close points
                done = True
                reward += 250  # Bonus for good solution

        return self.state, reward, done, False, {"message": "Action accepted"}

    def _get_reward(self, prev_convexhull):
        """
        Enhanced reward function focusing on traffic density and compactness
        """
        vertices = [self.intersection_dict[vertex_id]
                    for vertex_id, state in self.convexhull.items() if state]

        if len(vertices) < 3:
            # Small positive reward to encourage exploration
            return 1.0

        # Calculate basic properties
        convex_hull = cv2.convexHull(np.array(vertices, dtype=np.float32))
        hull_area = cv2.contourArea(convex_hull)
        mask = np.zeros(self.clean.shape[:2], np.uint8)
        cv2.drawContours(mask, [np.array(convex_hull, dtype=np.int32)], -1, 255, -1)
        hull_mask = mask / 255.0

        # Calculate traffic metrics
        traffic_sum = np.sum(self.heat_map * hull_mask)
        traffic_density = traffic_sum / (hull_area + 1e-6)
        total_area = self.clean.shape[0] * self.clean.shape[1]
        coverage_ratio = hull_area / total_area

        # Traffic density reward (primary objective with higher thresholds)
        if traffic_density > 0.95:
            traffic_reward = 400
        elif traffic_density > 0.85:
            traffic_reward = 250
        elif traffic_density > 0.75:
            traffic_reward = 150
        else:
            traffic_reward = -150

        # Size reward (prefer compact zones)
        if coverage_ratio < 0.01:  # Very compact
            size_reward = 100
        elif coverage_ratio < 0.02:  # Acceptable
            size_reward = 50
        else:  # Too large
            size_reward = -200

        # Configuration reward (reduced importance)
        num_points = len(vertices)
        if 3 <= num_points <= 4:
            config_reward = 50
        else:
            config_reward = -50

        # Compactness reward
        distances = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist = np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j]))
                distances.append(dist)
        avg_distance = np.mean(distances)

        if avg_distance < 50:
            compact_reward = 100
        elif avg_distance < 100:
            compact_reward = 50
        else:
            compact_reward = -100

        # Final reward calculation with adjusted weights
        total_reward = (
                traffic_reward * 0.6 +  # Increased from 0.4
                size_reward * 0.3 +  # Kept the same
                compact_reward * 0.05 +  # Reduced from 0.2
                config_reward * 0.05  # Reduced from 0.1
        )

        return total_reward

    def render(self, save_path=None):
        # Extract vertices from the convex hull
        vertices = [
            self.intersection_dict[vertex_id]
            for vertex_id, state in self.convexhull.items() if state
        ]
        new_map = self.clean.copy()

        # Check if there are enough points to create a convex hull
        if len(vertices) > 2:  # Convex hull requires at least 3 points
            try:
                convex_hull = cv2.convexHull(np.array(vertices, dtype=np.float32))
                if len(convex_hull) > 0:  # Ensure the convex hull is not empty
                    cv2.drawContours(new_map, [convex_hull.astype(int)], -1, (128, 0, 128), 2)  # Draw the convex hull
            except cv2.error as e:
                print(f"OpenCV error while drawing convex hull: {e}")
        else:
            print("Not enough points to draw a convex hull.")

        # Save or display the rendered image
        if save_path:
            cv2.imwrite(save_path, new_map)  # Save the rendered image to the file
            print(f"Image saved to {save_path}")
        else:
            cv2.imshow("Convex Hull", new_map)
            cv2.waitKey(1)  # Display the image briefly
        return new_map

    def _get_state(self):
        ch_channel = np.zeros(self.heat_map.shape, np.float32)
        heatmap_channel = self.heat_map.copy()
        dots_channel = np.zeros(self.heat_map.shape, np.float32)
        vertices = list(self.intersection_dict.values())
        if len(vertices) > 2:
            convex_hull = cv2.convexHull(np.array(vertices))
            cv2.drawContours(ch_channel, [convex_hull], -1, color=1.0, thickness=cv2.FILLED)
        for int_id, cords in self.intersection_dict.items():
            if self.intersection_state_dict[int_id]:
                cv2.circle(dots_channel, cords, 5, color=1.0, thickness=-1)
        ch_channel_resized = cv2.resize(ch_channel, (84, 84), interpolation=cv2.INTER_AREA)
        heatmap_channel_resized = cv2.resize(heatmap_channel, (84, 84), interpolation=cv2.INTER_AREA)
        dots_channel_resized = cv2.resize(dots_channel, (84, 84), interpolation=cv2.INTER_AREA)
        state = np.stack((dots_channel_resized, heatmap_channel_resized, ch_channel_resized), axis=0)
        state = state.astype(np.float32)
        return state

    def _get_heatmap(self):
        image_hsv = cv2.cvtColor(self.clean, cv2.COLOR_BGR2HSV)
        image_gray = cv2.cvtColor(self.clean, cv2.COLOR_BGR2GRAY)
        lower_red = np.array([0, 100, 0])
        upper_red = np.array([360, 250, 255])
        mask_red = cv2.inRange(image_hsv, lower_red, upper_red)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask_red.shape, np.uint8)
        for c in contours_red:
            M = cv2.moments(c)
            if M["m00"] < 500:
                continue
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        # Focus on darkest red areas
        heat_map = mask * image_gray
        max_val = np.max(heat_map)
        if max_val == 0:
            max_val = 1e-6  # Prevent division by zero
            heat_map = np.zeros_like(heat_map, dtype=np.float32)  # Create valid array
        else:
            heat_map = heat_map.astype(np.float32) / max_val

        return heat_map


def setup_logging(output_dir):
    """
    Sets up logging configuration for training monitoring.
    Creates a timestamped log file in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return timestamp


def train_agent(env, agent, num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE, debug=False,
                checkpoint_freq=10, early_stop_patience=20, max_steps=MAX_STEPS):
    """
    Enhanced training loop with improved monitoring and early stopping
    """
    # Setup training monitoring
    output_dir = "training_outputs"
    timestamp = setup_logging(output_dir)
    checkpoint_dir = f"{output_dir}/checkpoints_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize tracking variables
    rewards_list = []
    average_rewards = []
    episode_durations = []
    best_average_reward = float('-inf')
    episodes_without_improvement = 0
    training_start_time = time.time()
    last_save_reward = float('-inf')

    # Training metrics dictionary
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_losses': [],
        'epsilon_values': [],
        'average_rewards': [],
        'high_traffic_coverage': []
    }

    logging.info(f"Starting training with batch size {batch_size}")

    try:
        for episode in range(num_episodes):
            episode_start_time = time.time()
            state, _ = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            total_reward = 0
            episode_losses = []

            # Episode loop
            for step in range(max_steps):
                # Select and perform action
                with autocast():
                    action = agent.select_action(state)

                # Environment interaction
                next_state, reward, done, _, info = env.step(action)
                next_state = torch.FloatTensor(next_state).unsqueeze(0)

                # Store transition and update
                agent.buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Training step with gradient clipping
                if len(agent.buffer) >= batch_size:
                    loss = agent.update(batch_size)
                    if loss is not None:
                        episode_losses.append(loss)

                if done:
                    break

            # Post-episode updates
            agent.sync_target_network()
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

            # Calculate episode statistics
            episode_duration = time.time() - episode_start_time
            episode_durations.append(episode_duration)
            rewards_list.append(total_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0

            # Calculate moving average with larger window for stability
            window_size = 15  # Increased from 10
            if len(rewards_list) >= window_size:
                avg_reward = np.mean(rewards_list[-window_size:])
                average_rewards.append(avg_reward)

                # Check for improvement with higher threshold
                if avg_reward > best_average_reward + 1.0:  # Added minimum improvement threshold
                    best_average_reward = avg_reward
                    episodes_without_improvement = 0

                    # Save best model with minimum reward threshold
                    if avg_reward > 30 and avg_reward > last_save_reward + 5:  # Only save significantly better models
                        torch.save({
                            'episode': episode,
                            'model_state_dict': agent.q_net.state_dict(),
                            'optimizer_state_dict': agent.optimizer.state_dict(),
                            'reward': best_average_reward,
                        }, f'{checkpoint_dir}/best_model_reward_{avg_reward:.1f}.pth')
                        last_save_reward = avg_reward
                else:
                    episodes_without_improvement += 1

            # Update metrics
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_lengths'].append(step + 1)
            metrics['episode_losses'].append(avg_loss)
            metrics['epsilon_values'].append(agent.epsilon)
            if len(average_rewards) > 0:
                metrics['average_rewards'].append(average_rewards[-1])

            # Enhanced logging
            if debug or (episode + 1) % 5 == 0:
                logging.info(
                    f"Episode {episode + 1}/{num_episodes} - "
                    f"Reward: {total_reward:.2f} - "
                    f"Avg Loss: {avg_loss:.4f} - "
                    f"Epsilon: {agent.epsilon:.3f} - "
                    f"Duration: {episode_duration:.2f}s - "
                    f"Steps: {step + 1}"
                )

            # Periodic checkpoints with minimum performance threshold
            if (episode + 1) % checkpoint_freq == 0 and total_reward > 20:
                checkpoint_path = f'{checkpoint_dir}/checkpoint_episode_{episode + 1}.pth'
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.q_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': total_reward,
                }, checkpoint_path)

            # Enhanced early stopping with multiple conditions
            if (episodes_without_improvement >= early_stop_patience and
                    episode > 100 and  # Don't stop before 100 episodes
                    agent.epsilon < 0.5):  # Only stop when exploration is reduced
                logging.info(f"Early stopping triggered after {episode + 1} episodes")
                break

            # Save environment state visualization with improved frequency
            if (episode + 1) % 5 == 0 and total_reward > -100:  # Changed from > 0 to > -100
                env.render(save_path=f"{output_dir}/episode_{episode + 1}_reward_{total_reward:.1f}.png")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training error occurred: {str(e)}", exc_info=True)

    finally:
        # Save final metrics
        training_duration = time.time() - training_start_time
        metrics['total_training_time'] = training_duration
        metrics['average_episode_duration'] = np.mean(episode_durations)

        with open(f'{output_dir}/training_metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        # Generate and save training visualizations
        plot_training_results(metrics, output_dir, timestamp)

        logging.info(f"Training completed in {training_duration:.2f} seconds")
        return rewards_list, average_rewards, metrics


def plot_training_results(metrics, output_dir, timestamp):
    """
    Creates and saves comprehensive training visualization plots.
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics')

    # Plot episode rewards and moving average
    axes[0, 0].plot(metrics['episode_rewards'], label='Episode Reward', alpha=0.6)
    if len(metrics['average_rewards']) > 0:
        axes[0, 0].plot(metrics['average_rewards'], label='Moving Average', linewidth=2)
    axes[0, 0].set_title('Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot episode lengths
    axes[0, 1].plot(metrics['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot average losses
    axes[1, 0].plot(metrics['episode_losses'])
    axes[1, 0].set_title('Average Loss per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot epsilon decay
    axes[1, 1].plot(metrics['epsilon_values'])
    axes[1, 1].set_title('Epsilon Decay')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_plots_{timestamp}.png')
    plt.close()


def check_early_stop(rewards_list, window_size=10, threshold=-1000):
    """
    Check if we should stop training early based on recent performance
    """
    if len(rewards_list) < window_size:
        return False

    recent_rewards = rewards_list[-window_size:]
    avg_reward = sum(recent_rewards) / window_size

    return avg_reward > threshold


if __name__ == "__main__":
    # Setup environment and agent
    print(f"CUDA available: {torch.cuda.is_available()}")

    env = PIEnv(map="map_image.png")
    agent = DQNAgent(
        input_channels=3,
        action_dim=env.action_space.n,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE
    )

    # Run training with all optimizations
    rewards_list, average_rewards, metrics = train_agent(
        env=env,
        agent=agent,
        **training_config
    )
