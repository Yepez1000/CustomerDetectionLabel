import cv2
import numpy as np
import pytesseract
import os
import matplotlib.pyplot as plt
def deskew(image_path):
    # Read the image
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    plt.imshow(image)
    plt.show()
    if image is None:
        raise ValueError("Image not found or unable to open")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Check if any lines are detected
    if lines is None:
        print("No lines detected")
        return image

    # Calculate the angle of rotation based on the detected lines
    angles = []
    for line in lines:
        for rho, theta in line:
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:  # Ensure the angle is within a reasonable range
                angles.append(angle)
    
    # If no angles are within range, return the original image
    if len(angles) == 0:
        print("No valid angles detected")
        return image

    # Calculate the median angle to avoid extreme outliers
    median_angle = np.median(angles)

    # Rotate the image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def main():
     for filename in os.listdir('/Users/edgaryepez/AmericanShip/BoxImg/2024_07_23'):
        if filename.endswith('.jpg'):
            image_path = os.path.join('/Users/edgaryepez/AmericanShip/BoxImg/2024_07_23', filename)
            save_path = '/Users/edgaryepez/AmericanShip/TesseractTrain/Img_Prep/TrainingImages'
            deskewed_image = deskew(image_path)
            plt.imshow(deskewed_image)
            plt.show()
            
            
    # image_path = '/Users/edgaryepez/AmericanShip/BoxImg/cannotRead/1Z245W950313895228_2024-07-23 120425773.jpg'
    # deskewed_image = deskew(image_path)

    # Save the output
    # output_path = 'deskewed_image.jpg'
    # cv2.imwrite(output_path, deskewed_image)

    # Display the result
    

if __name__ == '__main__':
    main()
