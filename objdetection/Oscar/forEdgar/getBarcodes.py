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

def resize_image(image, max_width=1900, max_height=1900): #1500 ok
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

    return image
    
    
  

def get_jpg_images(folder_path):
    # List to hold image names
    image_names = []

    # Loop through the folder and get the names of all .jpg files
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.jpg'):
            image_names.append(file_name)
    
    return image_names    
def process_tracking_number(tracking_number):
    def replace_all(s, old, new):
        return s.replace(old, new)

    def get_alpha(s):
        return ''.join(filter(str.isalpha, s))

    # Initialize result dictionary
    result = {
        "Tracking": tracking_number,
        "Carrier": "",
        "Tracking_Message": ""
    }
    print ("Tracking number", tracking_number)

    track_length = len(tracking_number)
    print("Tracking number length",track_length)

    if track_length == 34:
        usps = tracking_number
        if usps == "42078":
            result["Tracking"] = tracking_number
            result["Carrier"] = "USPS"
        else:
            # FEDEX
            result["Tracking"] = tracking_number
            result["Carrier"] = "FedEx"
    elif track_length == 12:
        print("Tracking number length is 12")
        # FEDEX
        if tracking_number[:2] == "42":
            result["Tracking"] = ''
        else:
            result["Carrier"] = "FedEx"
        

    elif track_length == 10:
        # DHL
        result["Carrier"] = "DHL"
    elif track_length == 29:
        # USPS
        result["Tracking"] = "Error"
        # You can add a more specific message here if needed
        # result["Tracking"] = "Error: Code not recognized"
    elif track_length == 30:
        # USPS
        result["Tracking"] = tracking_number
        result["Carrier"] = "USPS"
    elif track_length == 31:
        # USPS
        tracking_number = replace_all(tracking_number, "↔", "")
        result["Tracking"] = tracking_number
        result["Carrier"] = "USPS"
    elif track_length == 22:
        # USPS, Estafeta
        if "A" in tracking_number:
            result["Tracking"] = tracking_number
            result["Carrier"] = "Estafeta"
        elif len(get_alpha(tracking_number)) > 0:
            result["Tracking"] = "Error: Code not recognized"
        else:
            result["Carrier"] = "USPS"
    elif track_length == 18:
        if tracking_number[:2] == "1Z":
            result["Carrier"] = "UPS"
        else:
            result["Tracking"] = "Error: Code not recognized"
            result["Carrier"] = ""
    else:
        result["Tracking_Message"] = f"Bar code not recognized {tracking_number}"
        result["Tracking"] = ""


    
    if "\x1d" in result["Tracking"]:
        result["Tracking"] = replace_all(result["Tracking"], "\x1d", "")

        print(result["Tracking"])
    print("Result",result)
    return result
def main(folder_path):
    """Main function to load, process, and display the image with barcode and text detection."""
    start_time = time.time()  # Record the start time
    counterTotalImages = 0
    counterCorrectDetections = 0

    image_names = get_jpg_images(folder_path)
    for image_name in image_names:
        print("000000000000000000000000000")
        counterTotalImages += 1
        image_path = folder_path+"/"+image_name
        print(f"Processing image: {image_path}")


        image = load_image(image_path)
        
        #image = enhance_image(image)
        #image = increase_contrast(image)
        
        
        imageResized = resize_image(image)
        barcodeDetected= False
        try:
            for code in decode(imageResized):
        
                print(code.type) 
                
                tracking_number = code.data.decode('utf-8')
                print(code.data.decode('utf-8').strip())
                result = process_tracking_number(tracking_number)
                print(result)
                if result["Carrier"] is not None:
                    barcodeDetected= True
                    counterCorrectDetections += 1
                    print("Barcode detected: Tracking ",result["Tracking"], ", Carrier ", result["Carrier"])
                    break
                else:
                    result = {
                        "Tracking": "",
                        "Carrier": "",
                        "Tracking_Message": ""
                    }
                    print("Barcode not detected")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            
            
            
            
        if    not barcodeDetected:
            print("preprocessing")
            imageOCR = preprocess_for_ocr(image)
            imageOCR = resize_image(imageOCR)
            #barcodeDetected= False
            try:
                for code in decode(imageOCR):
            
                    print(code.type) 
                    
                    tracking_number = code.data.decode('utf-8')
                    print(code.data.decode('utf-8').strip())
                    result = process_tracking_number(tracking_number)
                    print(result)
                    if result["Carrier"] is not None:
                        barcodeDetected= True
                        counterCorrectDetections += 1
                        print("Barcode detected: Tracking ",result["Tracking"], ", Carrier ", result["Carrier"])
                        break
                    else:
                        result = {
                            "Tracking": "",
                            "Carrier": "",
                            "Tracking_Message": ""
                        }
                        print("Barcode not detected")
            
            except Exception as e:
                print(f"An error occurred: {e}")
                
                
                
                
        if    not barcodeDetected:
            print("enhancing image")
            imageCONT = enhance_image(image)
            imageCONT = resize_image(imageCONT)
            
            #barcodeDetected= False
            try:
                for code in decode(imageCONT):
            
                    print(code.type) 
                    
                    tracking_number = code.data.decode('utf-8')
                    print(code.data.decode('utf-8').strip())
                    result = process_tracking_number(tracking_number)
                    print(result)
                    if result["Carrier"] is not None:
                        barcodeDetected= True
                        counterCorrectDetections += 1
                        print("Barcode detected: Tracking ",result["Tracking"], ", Carrier ", result["Carrier"])
                        break
                    else:
                        result = {
                            "Tracking": "",
                            "Carrier": "",
                            "Tracking_Message": ""
                        }
                        print("Barcode not detected")
            
            except Exception as e:
                print(f"An error occurred: {e}")
            
        if    not barcodeDetected:
            print("no detection")
            #destination_folder = 'noBarcode'
            #shutil.copy(image_path, destination_folder)
            #show_image(imageCONT)
            
    print("Total images ", counterTotalImages)
    print("Correct detections ",counterCorrectDetections) 
    
    #############################
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
if __name__ == "__main__":
    folderPath = "allImages"  # Replace with your image file path
    main(folderPath)
