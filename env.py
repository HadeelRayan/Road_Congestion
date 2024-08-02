
import sys

import cv2
import gymnasium
import numpy as np
from numpy.random import default_rng


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

        image = cv2.imread(map)
        if clean is not None:
            self.clean = cv2.imread(clean)
        else:
            self.clean = image.copy()

        ### find the red zone
        self.heat_map = self._get_heatmap()

        ## find the green zones and their coordinates
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lower_green = np.array([0, 150, 0])
        upper_green = np.array([100, 255, 100])
        mask_dark_green = cv2.inRange(image_rgb, lower_green, upper_green)
        # Find contours of dark green dots within the red shape
        contours_dark_green, _ = cv2.findContours(mask_dark_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find the center of each green contour and filter out noisy contours
        index = 0
        for c in contours_dark_green:
            try:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                continue

            self.intersection_dict[index] = (cX, cY)
            index = index + 1

        # update all intersections to be false - corresponding to empty convex hull
        for key, val in self.intersection_dict.items():
            self.intersection_state_dict[key] = False

        # action space is an scalar integer representing the index of an intersection
        self.action_space = gymnasium.spaces.Discrete(len(self.intersection_dict))

    def reset(self, seed=None):

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

        # generate the appropriate img state
        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        prev_convex_hull = self.convexhull.copy()
        if self.intersection_state_dict[action]:
            self.intersection_state_dict[action] = False
            del self.convexhull[action]
        else:
            self.intersection_state_dict[action] = True
            self.convexhull[action] = True

        # get new convex hull state and R(s,s')
        next_state = self._get_state()
        reward = self._get_reward(prev_convex_hull)

        self.state = next_state
        return next_state, reward, False, False, None

    def render(self):
        """
        Renders a viewable image of the city section with the current convex hull
        :return:
        The original clean image in RGB, and the contour of the convex hull drawn on it
        """
        vertices = []
        for vertex_id, state in self.convexhull.items():
            if state:
                vertices.append(self.intersection_dict[vertex_id])

        new_map = self.clean.copy()

        # draw convex hull
        convex_hull = cv2.convexHull(np.array(vertices))
        cv2.drawContours(new_map, [convex_hull], -1, (128, 0, 128), 2)

        return new_map

    def render_with_vertices(self):
        """
        enders a viewable image of the city section with the current convex hull and the intersections
        :return:
        The original clean image in RGB, and the contour of the convex hull drawn on it.
        Additionally all the intersections are drawn as circles, white circles represent active intersections,
        and black circles represents inactive intersections.
        """
        vertices = []
        for vertex_id, state in self.convexhull.items():
            if state:
                vertices.append(self.intersection_dict[vertex_id])

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
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def render2(self):
    #     vertices = []
    #     for vertex_id, state in self.convexhull.items():
    #         if state:
    #             vertices.append(self.intersection_dict[vertex_id])
    #
    #     new_map = self.clean.copy()
    #
    #     # draw convex hull
    #     convex_hull = cv2.convexHull(np.array(vertices))
    #     cv2.drawContours(new_map, [convex_hull], -1, (128, 0, 128), 2)
    #
    #     # draw intersection centroids with colors
    #     for int_id, cords in self.intersection_dict.items():
    #         if self.intersection_state_dict[int_id]:
    #             cv2.circle(new_map, cords, 5, (255, 255, 255), -1)
    #         else:
    #             cv2.circle(new_map, cords, 5, (0, 0, 0), -1)
    #
    #     cv2.imshow('Image with Convex Hull Around Perimeter Area', new_map)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     return new_map
    #     # return mask

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
        vertices = []
        for vertex_id, state in self.convexhull.items():
            if state:
                vertices.append(self.intersection_dict[vertex_id])
        convex_hull = cv2.convexHull(np.array(vertices))
        cv2.drawContours(ch_channel, [convex_hull], -1, color=1.0, thickness=cv2.FILLED)

        # generate the dots mask
        for int_id, cords in self.intersection_dict.items():
            if self.intersection_state_dict[int_id]:
                cv2.circle(dots_channel, cords, 5, color=1.0, thickness=-1)
            else:
                cv2.circle(dots_channel, cords, 5, color=-1.0, thickness=-1)

        state = np.stack((ch_channel, heatmap_channel, dots_channel),axis=0)
        return state

        # vertices = []
        # for vertex_id, state in self.convexhull.items():
        #     if state:
        #         vertices.append(self.intersection_dict[vertex_id])
        #
        # new_map = self.clean.copy()
        #
        # # draw convex hull
        # convex_hull = cv2.convexHull(np.array(vertices))
        # cv2.drawContours(new_map, [convex_hull], -1, (128, 0, 128), 2)
        #
        # return new_map

    def _get_reward(self, prev_convexhull):
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
            reward = sum_v/count_v - (self.regularizer * count_v)*np.sign(sum_v)

        return reward

    def _get_heatmap(self):
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





