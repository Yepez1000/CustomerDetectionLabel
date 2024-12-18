import cv2
from pyzbar.pyzbar import decode

import pytesseract
import time
import pandas as pd
import os
from PIL import Image, ImageOps,ImageEnhance, ImageFilter
import numpy as np
import shutil #Only to copy files

def load_image(image_path):
    """Load an image from the specified file."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    return image

def resize_image(image, max_width=1900, max_height=1900):
    """Resize the image to fit within the specified dimensions, maintaining aspect ratio."""
    height, width = image.shape[:2]
    if width > height:
        new_width = min(max_width, width)
        new_height = int(new_width * height / width)
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * width / height)
    else:
        new_height = min(max_height, height)
        new_width = int(new_height * height / width)
        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width * height / width)
            
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def detect_barcode(image):
    """Detect barcode in the image and extract the text."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray)
    barcode_data = [barcode.data.decode('utf-8') for barcode in barcodes]
    return barcode_data

def extract_text(image):
    """Extract text from the image using OCR."""
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    custom_config = r'--tessdata-dir "/opt/homebrew/share/tessdata" -l eng'
    text = pytesseract.image_to_string(image, config=custom_config)



    return text

def show_image(image):
    """Display the image in a window."""
    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def preprocess_for_ocr(image):
    """Preprocess the image for better OCR results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresh = deskew(thresh)
    return thresh
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply adaptive thresholding to get a binary image
    binary_image = cv2.adaptiveThreshold(blurred_image, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    
    # Apply dilation and erosion to remove small noise and emphasize text
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(binary_image, kernel, iterations=1)
    processed_image = cv2.erode(processed_image, kernel, iterations=1)
    
    # Invert the image (if needed, depending on text color)
    #processed_image = cv2.bitwise_not(processed_image)
    
    # Optionally, deskew the image if it's not aligned properly
    #processed_image = deskew(processed_image)
    
    return processed_image
    
    
def increase_contrast(image):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L-channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge the channels back
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    
    # Convert LAB image back to BGR color space
    contrast_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return contrast_image
    
def enhance_image(image):
    # Check if the image is a NumPy array (as it would be when loaded with cv2)
    if isinstance(image, np.ndarray):
        # Convert the image to a PIL Image
        image = Image.fromarray(image)

    # Convert the image to grayscale
    image = image.convert('L')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.4)  # Adjust contrast (use <1.0 for less contrast)

    # Apply a sharpening filter
    image = image.filter(ImageFilter.SHARPEN)

    # Convert back to a NumPy array if you need to continue processing with OpenCV
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
    
    
# Function to deskew the image
def deskew(image):
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:  # Color image (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated    
def rotate_image_90_right(image):
    """Rotates the given image 90 degrees to the right (clockwise)."""
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image
# Function to get all .jpg images in a folder
def get_jpg_images(folder_path):
    # List to hold image names
    image_names = []

    # Loop through the folder and get the names of all .jpg files
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.jpg'):
            image_names.append(file_name)
    
    return image_names    
    
def main(folder_path):
    """Main function to load, process, and display the image with barcode and text detection."""
    start_time = time.time()  # Record the start time
    customer_data = {}
    counterTotalImages = 0
    counterCorrectDetections = 0
    df = pd.read_csv('/Users/edgaryepez/AmericanShip/tflite1/objdetection/CustomersView3.csv')
    for index, row in df.iterrows():
        client_id = row['Unique STE # AS-']
        customer_data[client_id] = row['First Name'] + " " + row['Last Name']
    
    
    image_names = get_jpg_images(folder_path)
    
    for image_name in image_names:
        counterTotalImages += 1
        image_path = folder_path+"/"+image_name
        print(f"Processing image: {image_path}")
        # You can add more processing logic here (e.g., opening the image, rotating it, etc.)
        
        try:
            image = load_image(image_path)
            #image = rotate_image_90_right(image)
            #image = deskew(image)
            
            #image = enhance_image(image)#ok
            #image = increase_contrast(image)#ok
            #image = preprocess_image(image)
            #image = preprocess_for_ocr(image)
            #image = deskew(image)
            imageResized = resize_image(image)
            #show_image(image)
            
            
            print("1*********************************")
            extracted_text = extract_text(imageResized)
            extracted_text = extracted_text.lower()
            print(extracted_text)
            customerFound = False
            for client_id in customer_data:
                customerIdStr = "as"+str(client_id)
                customerIdStr= customerIdStr.lower()
                #print(customerIdStr)
                if(customerIdStr in extracted_text):
                    customer = str(customer_data[client_id])
                    print(customer)
                    customer = customer.lower()
                    if(customer in extracted_text):
                        print("CUSTOMER FOUND: " + customerIdStr + " " + customer_data[client_id])
                        counterCorrectDetections +=1
                        customerFound = True
            if (not customerFound):
                print("Customer not found but will try again by increasing contrast on image")
                imageContrasted = increase_contrast(image)
                imageResizedContrasted = resize_image(imageContrasted)
                extracted_text = extracted_text + extract_text(imageResizedContrasted)
                extracted_text = extracted_text.lower()
                print(extracted_text)
                customerFound = False
                for client_id in customer_data:
                    customerIdStr = "as"+str(client_id)
                    customerIdStr= customerIdStr.lower()
                    #print(customerIdStr)
                    if(customerIdStr in extracted_text):
                        customer = str(customer_data[client_id])
                        print(customer)
                        customer = customer.lower()
                        if(customer in extracted_text):
                            print("CUSTOMER FOUND: " + customerIdStr + " " + customer_data[client_id])
                            counterCorrectDetections +=1
                            customerFound = True
                        else:
                            print("Customer not found")
            if (not customerFound):
                print("Customer not found but will try again by doing extra preprocessing")
                image2 = enhance_image(image)
                image2 = preprocess_for_ocr(image2)
                image2Resized = resize_image(image2)
                extracted_text = extracted_text + extract_text(image2Resized)
                extracted_text = extracted_text.lower()
                print(extracted_text)
                customerFound = False
                for client_id in customer_data:
                    customerIdStr = "as"+str(client_id)
                    customerIdStr= customerIdStr.lower()
                    #print(customerIdStr)
                    if(customerIdStr in extracted_text):
                        customer = str(customer_data[client_id])
                        print(customer)
                        customer = customer.lower()
                        if(customer in extracted_text):
                            print("CUSTOMER FOUND: " + customerIdStr + " " + customer_data[client_id])
                            counterCorrectDetections +=1
                            customerFound = True
                        else:
                            print("Customer not found")            
            if (not customerFound):
                #destination_folder = ''
                #shutil.copy(image_path, destination_folder)
                #show_image(image2Resized)
                print("Not found")
                
        except Exception as e:
            print(f"An error occurred: {e}")
    print("Total images ", counterTotalImages)
    print("Correct detections ",counterCorrectDetections) 
    
    #############################
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
if __name__ == "__main__":
    folderPath = "/Users/edgaryepez/AmericanShip/Images/allImagesSept17"
    main(folderPath)
