import os
import random
import glob
import numpy as np
import time
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter
import cv2
import pytesseract
import re
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from playsound import playsound
import pandas as pd
import time
import math
import subprocess


AS_found_count = [0, 0]
found_match_count = [0]



df = pd.read_csv('/Users/edgaryepez/AmericanShip/tflite1/objdetection/CustomersView3.csv')
df['Unique STE # AS-'] = df['Unique STE # AS-'].astype(str)
# namedict = {}
# asdict = {}
# for index, row in df.iterrows():
#     first_name = row['First Name']
#     second_name = row['Last Name']
#     client_id = row['Unique STE # AS-']

#     namedict[first_name] = [second_name, client_id]
    
#     asdict[client_id] = [first_name, second_name]
    

# print(namedict['Jorge'])

def convert_to_pil(image):
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image
    else:
        raise TypeError("The image must be a NumPy array or PIL Image.")
    
# def found_AS_and_Match2(image):

    
#     # Set the Tesseract executable path
#     os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'
#     pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


#     # Perform OCR using the custom model
#     custom_config = r'--tessdata-dir "/opt/homebrew/share/tessdata" -l box5'
#     text = pytesseract.image_to_string(image, config=custom_config)


#     # for word in text.split():
       

#     #     matches = df[df['First Name'].str.lower() == first_name.lower()]
     
#     #     for _, row in matches.iterrows():
#     #         if row is not np.nan:
#     #             print(row)



#     # """Extract customer information from the provided text."""
#     # # Initialize variables

#     # # Clean up text by removing non-alphanumeric characters (excluding spaces)
#     text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
#     print("text2\n",text)
#     for index, row in df.iterrows():
#         first_name = row['First Name']
#         second_name = row['Last Name']
#         client_id = row['Unique STE # AS-']
        

#         # Create a regex pattern to search for name and ID
#         pattern = rf"{client_id}"
#         pattern2 = rf"{first_name} "
#         pattern3 = rf"{second_name}"

#         # Check if the pattern is found in the text

        
        
#         if (re.search(pattern, text, re.IGNORECASE) and (re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE))) or (re.search(pattern2, text, re.IGNORECASE) and re.search(pattern3, text, re.IGNORECASE)):
#             print(f"Found: {first_name} {second_name} with ID {client_id}")
            
            
#             found_match_count[1] += 1
#             return True, False
        

#     print("No match")


#     return False, False


def found_AS_and_Match(image):

    foundtheAS = False
    foundthematch = False
    # Set the Tesseract executable path
    os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


    # Perform OCR using the custom model
    custom_config = r'--tessdata-dir "/opt/homebrew/share/tessdata" -l box3'
    text = pytesseract.image_to_string(image, config=custom_config)
    



    """Extract customer information from the provided text."""
    # Initialize variables

    # Clean up text by removing non-alphanumeric characters (excluding spaces)
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    print("text1\n",text)
    for index, row in df.iterrows():
        first_name = row['First Name']
        second_name = row['Last Name']
        client_id = row['Unique STE # AS-']
        

        # Create a regex pattern to search for name and ID
        pattern = rf"{client_id}"
        pattern2 = rf"{first_name} "
        pattern3 = rf"{second_name}"

        # Check if the pattern is found in the text

        
        
        if (re.search(pattern, text, re.IGNORECASE) and (re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE))) or (re.search(pattern2, text, re.IGNORECASE) and re.search(pattern3, text, re.IGNORECASE)):

            print(f"text1:Found: {first_name} {second_name} with ID {client_id}")

            matches = df[(df['Unique STE # AS-'].str.contains(client_id, na=False))]


            print(matches)

            
            
            found_match_count[0] += 1
            return True, matches
        

    print("No match")


    return False, None





    # print("this is cleaned text",cleaned_text)

    # # Split the cleaned text into words
    # words = cleaned_text.split()

    # print("this is words",words)

    # # Extract first name (assuming it's the first word)
    # if words:
    #     first_name = words[0]

    # # Convert words list to string for further processing
    # text_str = ' '.join(words)

    # print("this is text_str",text_str)

    # # Extract AS number using regex
    # as_number_match = re.search(r'AS\d{5}', text_str)
    # if as_number_match:
    #     as_number = int(as_number_match.group()[2:])



    # # Ensure 'Unique STE # AS-' column is of string type
    # df['Unique STE # AS-'] = df['Unique STE # AS-'].astype(str)

    # # Query DataFrame for matching first name
    # potential_matches = df[df['First Name'].str.lower() == first_name.lower()]

    # # Further refine matches by last name
    # for _, row in potential_matches.iterrows():
    #     last_name = row['Last Name']
    #     if re.search(last_name, text_str, re.IGNORECASE):
    #         return True, row

    # # Query DataFrame for matching AS number
    # if as_number is not None:
    #     as_matches = df[df['Unique STE # AS-'].str.contains(str(as_number), na=False)]
    #     if not as_matches.empty:
    #         first_name = as_matches['First Name']
    #         last_name = as_matches['Last Name']
            
    #         # Create a regex pattern to search for the names
    #         pattern2 = rf"{first_name} "
    #         pattern3 = rf"{last_name}"

    #         # Check if the names are found in the text
    #         if re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE):
    #             print("Found a match!")
    #             return True, as_matches

    # print("No match found!")
    # return False, ""


def deskew(image_path):
    # Read the image
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
   
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
        # print("No lines detected")
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
        # print("No valid angles detected")
        return image

    # Calculate the median angle to avoid extreme outliers
    median_angle = np.median(angles)

    # Rotate the image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def enhance_image(image):
    new_size = (image.shape[1] * 1, image.shape[0] * 1)

    # Resize the image using OpenCV
    enlarged_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    if isinstance(enlarged_image, np.ndarray):
        image = Image.fromarray(enlarged_image)

    image = image.convert('L')

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    image = image.filter(ImageFilter.SHARPEN)

    return image


def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, num_test_images=10, savepath='/content/results', txt_only=False):

  # Grab filenames of all images in test folder
  images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')

  # Load the label map into memory
  with open(lblpath, 'r') as f:
      labels = [line.strip() for line in f.readlines()]

  # Load the Tensorflow Lite model into memory
  interpreter = Interpreter(model_path=modelpath)
  interpreter.allocate_tensors()

  # Get model details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  float_input = (input_details[0]['dtype'] == np.float32)

  input_mean = 127.5
  input_std = 127.5

  # Randomly select test images
  images_to_test = random.sample(images, num_test_images)

  # Loop over every image and perform detection
  for image_path in images_to_test:

      # Load image and resize to expected shape [1xHxWx3]
      print(image_path)
      image = deskew(image_path)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      imH, imW, _ = image.shape
      image_resized = cv2.resize(image_rgb, (width, height))
      input_data = np.expand_dims(image_resized, axis=0)

      # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
      if float_input:
          input_data = (np.float32(input_data) - input_mean) / input_std

      # Perform the actual detection by running the model with the image as input
      interpreter.set_tensor(input_details[0]['index'],input_data)
      interpreter.invoke()

      # Retrieve detection results
      boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

      detections = []

      # Loop over all detections and draw detection box if confidence is above minimum threshold



      crop_img = image[0+ 5:image.shape[0], 0 + 10:image.shape[1]]

      


      for i in range(len(scores)):
          if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))

            #   cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            #   plt.imshow(image)
            #   plt.show()

              # Crop Image
             
              crop_img = image[ymin:ymax, xmin:xmax]

              

              enhanced_image = enhance_image(crop_img)

            #   plt.imshow(enhanced_image)
            #   plt.show()

            #   os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'
            #   pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


            # # Perform OCR using the custom model
            #   custom_config = r'--tessdata-dir "/opt/homebrew/share/tessdata" -l eng'
            #   text = pytesseract.image_to_string(enhanced_image, config=custom_config)

            #   print(text)

              matchfound, matches = found_AS_and_Match(enhanced_image)

            #   if (matchfound):
            #       for row in matches.iterrows():
            #           print(row[1]["Last Name"])
                      
           
             
                  
             

            #   if matchfound == False:
            #       os.makedirs('/Users/edgaryepez/AmericanShip/tflite1/objdetection/2024_07_23_CannotRead', exist_ok=True)
            #       savepath = os.path.join('//Users/edgaryepez/AmericanShip/tflite1/objdetection/2024_07_23_CannotRead',os.path.basename(image_path))
            #       print(savepath)
            #       enhanced_image.save(savepath)
            #     enhanced_image = np.array(enhanced_image)
            #     text =easyOCR(enhanced_image)
            #     __, matchfound = found_AS_and_Match(enhanced_image)


            #   if (_):
            #     AS_found_count[0] += 1


                  


              # Draw label
              object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
              label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
              label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
              cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
              cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

              detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
        
    #   if detections == []:
    #     print('No detections found.')
    #     os.makedirs('/Users/edgaryepez/AmericanShip/tflite1/objdetection/2024_07_23_No_detections', exist_ok=True)
    #     savepath = os.path.join('/Users/edgaryepez/AmericanShip/tflite1/objdetection/2024_07_23_No_detections',os.path.basename(image_path))
    #     print(savepath)
    #     newimage = convert_to_pil(image)
    #     newimage.save(savepath)


      # All the results have been drawn on the image, now display the image
      if txt_only == False: # "text_only" controls whether we want to display the image results or just save them in .txt files
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(7,10))
        plt.imshow(image)
        # plt.show()

      # Save detection results in .txt files (for calculating mAP)
      elif txt_only == True:

        # Get filenames and paths
        image_fn = os.path.basename(image_path)
        base_fn, ext = os.path.splitext(image_fn)
        txt_result_fn = base_fn +'.txt'
        txt_savepath = os.path.join(savepath, txt_result_fn)

        # Write results to text file
        # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
        with open(txt_savepath,'w') as f:
            for detection in detections:
                f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

  return

# Set up variables for running user's model
# PATH_TO_IMAGES='/Users/edgaryepez/AmericanShip/All2024_07_23/2024_07_23'   # Path to test images folder
PATH_TO_IMAGES='/Users/edgaryepez/AmericanShip/Images/allImagesSept17'
PATH_TO_MODEL='/Users/edgaryepez/AmericanShip/tflite1/custom_model_liteBOX/detect.tflite'  # Path to .tflite model file
PATH_TO_LABELS='/Users/edgaryepez/AmericanShip/tflite1/custom_model_liteBOX/labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold=0.1   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
images_to_test = 5 # Number of images to run detection on

# Run inferencing function!

start_time = time.time()  # Record the start time
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Calculate the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")

print((images_to_test), "images were tested")
print((found_match_count), "% of the images were found a match in AmericanShips")


