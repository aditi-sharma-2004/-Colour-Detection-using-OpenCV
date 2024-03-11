import cv2
import numpy as np
from collections import Counter

def find_prominent_colors(image, num_colors=5):
    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Use k-means clustering to find dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to integer values
    centers = np.uint8(centers)
    
    # Count occurrences of each label
    label_counts = Counter(labels.flatten())
    
    # Sort colors by frequency
    sorted_centers = [centers[i] for i, _ in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)]
    
    return sorted_centers

def display_colors(colors):
    # Create a blank image to display colors
    color_display = np.zeros((100, len(colors) * 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        # Convert BGR to RGB for correct display
        rgb_color = color[::-1]
        color_display[:, i * 100:(i + 1) * 100] = rgb_color
    
    # Display colors
    cv2.imshow('Dominant Colors', color_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    # Load an image
    image_path = 'img2.jpeg'
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to load the image.")
    else:
        # Find prominent colors in the image
        prominent_colors = find_prominent_colors(image)
        print("Prominent Colors:", prominent_colors)
        
        # Display the prominent colors
        display_colors(prominent_colors)
