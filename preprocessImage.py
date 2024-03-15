import numpy as np
import cv2


def main():
    # Define a target size
    target_size = (128, 128)

    # Paths to the images
    green_dots_path = 'green_dots_on_white.jpg'
    convex_hull_path = 'convex_hull_image.jpg'
    heatmap_path = 'red_areas_on_white.jpg'

    # Load and resize the images
    green_dots_image = cv2.imread(green_dots_path, cv2.IMREAD_GRAYSCALE)
    green_dots_image = cv2.resize(green_dots_image, target_size, interpolation=cv2.INTER_AREA)

    convex_hull_image = cv2.imread(convex_hull_path, cv2.IMREAD_GRAYSCALE)
    convex_hull_image = cv2.resize(convex_hull_image, target_size, interpolation=cv2.INTER_AREA)

    heatmap_image = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    heatmap_image = cv2.resize(heatmap_image, target_size, interpolation=cv2.INTER_AREA)

    # Normalize the images
    green_dots_image = green_dots_image / 255.0
    convex_hull_image = convex_hull_image / 255.0
    heatmap_image = heatmap_image / 255.0

    # Stack the images to create a multi-channel input for the CNN
    cnn_input = np.stack((green_dots_image, convex_hull_image, heatmap_image), axis=-1)
    return cnn_input


if __name__ == "__main__":
    main()