import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

if __name__ == "__main__":
    image = cv2.imread('HeatMapWithTrafficLight_full.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define thresholds for red color
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])

    # Create a binary mask of red areas
    mask_red = cv2.inRange(image_rgb, lower_red, upper_red)

    # Find contours of red regions
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a binary mask of dark green areas
    lower_green = np.array([0, 150, 0])
    upper_green = np.array([100, 255, 100])
    mask_dark_green = cv2.inRange(image_rgb, lower_green, upper_green)

    # Find contours of dark green dots within the red shape
    contours_dark_green, _ = cv2.findContours(mask_dark_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine all points from different dark green regions into a single array
    all_points_dark_green = np.concatenate([contour for contour in contours_dark_green])

    # Compute the convex hull of all dark green points within the red shape
    convex_hull = cv2.convexHull(all_points_dark_green)

    # Create a copy of the original image to preserve the colors
    result_image = image.copy()

    # Draw the boundaries of the convex hull around the dark green dots
    cv2.drawContours(result_image, [convex_hull], -1, (128, 0, 128), 2)

    result_filename = 'result_with_convex_hull.jpg'
    cv2.imwrite(result_filename, result_image)

    # Display the result
    cv2.imshow('Image with Convex Hull Around Dark Green Dots', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
