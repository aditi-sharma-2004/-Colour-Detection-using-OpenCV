import cv2
import numpy as np
from collections import Counter

def get_prominent_colors(image, num_colors=5):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
   #find the dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
   
    centers = np.uint8(centers)
    # Count the occurrence
    label_counts = Counter(labels.flatten())

    # Sort the colors
    sorted_centers = [centers[i] for i, _ in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)]
    return sorted_centers

def display_colors(colors):
    color_display = np.zeros((100, len(colors) * 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        color_display[:, i * 100:(i + 1) * 100] = color
    cv2.imshow('Dominant Colors', color_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    image_path = 'your_image.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
    else:
        prominent_colors = get_prominent_colors(image)
        print("Prominent Colors:", prominent_colors)
        display_colors(prominent_colors)
