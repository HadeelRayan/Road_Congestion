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

    # Apply the mask to the original image to keep only the red areas in their original color
    red_areas_original_color = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_red)

    # Create a white background image
    white_background = np.ones_like(image_rgb) * 255

    # Keep everything else white
    everything_else_white = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(mask_red))

    # Combine the red areas with the white background
    result_image = cv2.bitwise_or(red_areas_original_color, everything_else_white)

    # Save the result image
    cv2.imwrite('red_areas_on_white.jpg', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    # Display the result image
    cv2.imshow('Red Areas Image', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

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

    # Create a white background image
    white_background = np.ones_like(image_rgb) * 255

    # Draw the boundaries of the convex hull around the dark green dots on a separate image
    convex_hull_image = white_background.copy()
    cv2.drawContours(convex_hull_image, [convex_hull], -1, (128, 0, 128), 2)

    # Save the image with convex hull
    convex_hull_filename = 'convex_hull_image.jpg'
    cv2.imwrite(convex_hull_filename, cv2.cvtColor(convex_hull_image, cv2.COLOR_RGB2BGR))

    # Create a mask for everything except the green dots
    mask_not_green_dots = cv2.bitwise_not(mask_dark_green)

    # Keep only the green dots in their original color
    green_dots_original_color = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_dark_green)

    # Keep everything else white
    everything_else_white_gd = cv2.bitwise_and(white_background, white_background, mask=mask_not_green_dots)

    # Combine the green dots with the white background
    result_image_green_dots = cv2.bitwise_or(green_dots_original_color, everything_else_white_gd)

    # Save the green dots image
    cv2.imwrite('green_dots_on_white.jpg', cv2.cvtColor(result_image_green_dots, cv2.COLOR_RGB2BGR))

    # Display the result images
    cv2.imshow('Image with Convex Hull Around Dark Green Dots', cv2.cvtColor(convex_hull_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Green Dots Image', cv2.cvtColor(result_image_green_dots, cv2.COLOR_RGB2BGR))
    cv2.imshow('Red Areas Image', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()