# Script to run custom TFLite model on test images to detect objects
# Source: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_image.py

# Import packages
import subprocess
import cv2
import easyocr
import os
import cv2
import numpy as np
import sys
import glob
import random
from PIL import Image, ImageEnhance, ImageFilter
import re
import importlib.util
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import easyocr
import math
import os
import copy
import pytesseract
import re
import subprocess
import pandas as pd


import easyocr
from tensorflow.lite.python.interpreter import Interpreter

import pytesseract
import math

import matplotlib
import matplotlib.pyplot as plt

confidenceofOCR = [0,0,0,0,0]
ASfound = [0,0,0,0,0]
foundmatch = [0]

# %matplotlib inline

def found_AS_and_Match(image):

    foundtheAS = False
    foundthematch = False
    # Set the Tesseract executable path
    os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


    # Perform OCR using the custom model
    custom_config = r'--tessdata-dir "/opt/homebrew/share/tessdata" -l eng'
    text = pytesseract.image_to_string(image, config=custom_config)

    pattern = re.compile(r'AS\d{5}')
    match = pattern.search(text)
    if match:
        foundtheAS = True
        found_as = match.group()
        print("Found AS ",found_as)
        
        
        ASfound[0] += 1

        df = pd.read_csv('/Users/edgaryepez/AmericanShip/tflite1/objdetection/CustomersView.csv')

        # Define the text to search

        # Loop through each row in the DataFrame
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
                print(f"Found: {first_name} {second_name} with ID {client_id}")
                
                foundmatch[0] += 1
                foundthematch = True
                break
        print(text) 
        return True

    
    return False


def rotate_image_and_crop(image, comp_point, top_left_word_end_down):
    """
    Rotate the image by a specified angle.
    """
    # Visualize points
    image_with_points = image.copy()
    image_with_points = cv2.circle(image_with_points, (comp_point[0], comp_point[1]), 5, (255, 0, 0), -1)
    image_with_points = cv2.circle(image_with_points, (top_left_word_end_down[0], top_left_word_end_down[1]), 5, (255, 0, 0), -1)
    plt.imshow(image_with_points)
    # plt.show()

    # Calculate rotation angle
    theta = np.arctan(-(comp_point[1] - top_left_word_end_down[1]) / (comp_point[0] - top_left_word_end_down[0]))
    rotation_factor = -np.degrees(theta)

    # print("rotation",rotation_factor)

    # Rotate image if necessary
    if abs(rotation_factor) < 1:
        return image, False
        
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    pattern = re.compile(r'AS\d{5}')

    rotation_factor += 1
    for i in range(10):

        angle = rotation_factor/(i+1)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        # text = pytesseract.image_to_string(rotated, lang='eng', config=r'-l box')
        # print(text)

        if i >= 10:
            break

        if found_AS_and_Match(rotated):
            print(f"After rotation{i} AS found")
            plt.imshow(rotated)
            # plt.show()
            return rotated, True
            
    # else:
    #     print("No match found within 10 iterations")


    return rotated, False

def find_reference_word(referencewordpt,results, is_comp = False):
    """
    Find the reference word in the image.
    """
    next_level = None
    reference_word_top_right, reference_word_top_left, reference_word_bottom_right = None, None, None
    distance_from_reference = 100000
    ref_loc = 0
    for i, detection in enumerate(results):
        top_left = tuple([int(val) for val in detection[0][0]])
        top_right = tuple([int(val) for val in detection[0][1]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        bottom_left = tuple([int(val) for val in detection[0][3]])

      

        if (bottom_left[1] - top_left[1] > 26):
            continue

        distance0 = math.sqrt((referencewordpt[0]-top_left[0])**2 + (referencewordpt[1]-top_left[1])**2)
        if is_comp == True and (referencewordpt[1]-top_left[1] < -10 or referencewordpt[0]-top_left[0] >20):
            continue
        elif referencewordpt[1]-top_left[1] > 20 or referencewordpt[1]-top_left[1] < -20:
            continue
        elif distance0 < distance_from_reference:
            distance_from_reference = distance0
            next_level = bottom_left
            reference_word_top_right = top_right
            reference_word_top_left = top_left
            reference_word_bottom_right = bottom_right
            ref_loc = i

    

    return next_level, reference_word_top_right, reference_word_top_left, reference_word_bottom_right, ref_loc

def find_closest_points(results, image):
    """
    Find the closest points in the text detection results.
    """
    top_left_word_end_down = [0, 0]
    Closest_Point = []
    referencewordpt = [0,0]
    next_level = None


    while(referencewordpt):
        if next_level is not None:
            referencewordpt = next_level
            next_level = None

        next_level, reference_word_top_right, reference_word_top_left, reference_word_bottom_right, ref_loc = find_reference_word(referencewordpt, results)
        # print(next_level)
        if next_level is None:
            # print("here")
            next_level = referencewordpt[0], referencewordpt[1]+10
        else:
            del results[ref_loc]
            # cv2.rectangle(image, (reference_word_top_left), (reference_word_bottom_right), (0, 255, 0), 2)
            plt.imshow(image)
            # plt.show()
            _,_,comp_top_left, comp_bottom_right, _ = find_reference_word(reference_word_top_right, results, True)
            

            # cv2.rectangle(image, (comp_top_left), (comp_bottom_right), (0, 0, 255), 2)
            plt.imshow(image)
            # plt.show()

            if comp_top_left is not None:
                return reference_word_bottom_right,comp_bottom_right

            
    
        if next_level[1] > image.shape[0]:
            break
        
    return top_left_word_end_down, Closest_Point

def detect_and_save_text_regions(results, image, save_dir, filename):
    """Detects text regions using EasyOCR, crops them, and saves as new images."""


    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Remove .jpg extension from the filename
    base_filename = os.path.splitext(filename)[0]

    # Iterate through each detection and save cropped region
    for i, detection in enumerate(results):
        # Extract bounding box coordinates
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        bottom_left = tuple([int(val) for val in detection[0][3]])

        if bottom_left[1] - top_left[1] > 26:
            continue

        # Ensure coordinates are within image bounds
        if top_left[0] < 0 or top_left[1] < 0 or bottom_right[0] > image.shape[1] or bottom_right[1] > image.shape[0]:
            print(f"Skipping out-of-bounds detection at index {i}")
            continue

        # Crop region from original image
        cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # print text from result
        print(detection[1])

        # Check if cropped_image is empty
        if cropped_image.size == 0:
            print(f"Skipping empty crop at index {i}")
            continue
        
        # plt.imshow(cropped_image)
        # plt.show()

        # Save cropped image as .tif
        save_path = os.path.join(save_dir, f"{base_filename}_detection_{i+1}.tif")
        cv2.imwrite(save_path, cropped_image)


        subprocess.run(f'tesseract {save_path} {os.path.join(save_dir, f"{base_filename}_detection_{i+1}")} batch.nochop makebox', shell=True)

        file_path = os.path.join(save_dir, f"{base_filename}_detection_{i+1}.gt.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Create the file if it doesn't exist and write to it
        with open(file_path, 'w') as file:
            file.write(detection[1])
        print(f"Successfully created and wrote to {file_path}")

    print(f"Saved {len(results)} text regions from {filename} to {save_dir}")


def extract_text_from_image(image):
    """
    Extract text from the image and align it.
    """
    # Check if the image is a valid NumPy array

    

    
    # Initialize the easyocr reader
    reader = easyocr.Reader(['en'])

    # Read text from the image object using easyocr
    results = reader.readtext(image)

    return results
    
    
def run_all_processes(image, save_path):
    
    image = np.array(image)
    
    results = extract_text_from_image(image)
 
    resultscpy = copy.deepcopy(results)
    top_left_word_end_down, comp_point = find_closest_points(results, image)
    results = resultscpy
    if not comp_point:
        print("No match")

        return image, False
    else:
        rotated, rotation_factor = rotate_image_and_crop(image, top_left_word_end_down, comp_point)
        if rotation_factor is True:
            return rotated, True

            results = extract_text_from_image(rotated)
            plt.imshow(image, cmap='gray')
            # plt.show()
        else: 
            print("No match")
            return image, False 
        
    
        
    # detect_and_save_text_regions(results, image, save_path, os.path.basename(image_path))


def OCRconfidence(image):
  data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=r'-l box')

  overall_confidence = 0
  alltext = ''
  asf = 0
  # Iterate over each word and its bounding box
  for i in range(len(data['text'])):
      text = data['text'][i]
      confidence = int(data['conf'][i]) if 'conf' in data else None
      x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
      # print(f"Text: {text}, Confidence: {confidence}, BBox: (left: {x}, top: {y}, width: {w}, height: {h})")
      alltext = alltext + " " + text
      if "AS" in text:
         asf += 1
         
      
      overall_confidence = overall_confidence + confidence
  print(alltext)
  # print("Overall confidence: ", overall_confidence) # Print the overall_confidence
  return overall_confidence, asf
### Define function for inferencing with TFLite model and displaying results
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
  for r, image_path in enumerate(images_to_test):

      # Load image and resize to expected shape [1xHxWx3]
      image = cv2.imread(image_path)
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
      for i in range(len(scores)):
          if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))

              # cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

              # Crop Image
             
              crop_img = image[ymin:ymax, xmin:xmax]

              enhanced_image = enhance_image(crop_img)

              

             
              asf = found_AS_and_Match(enhanced_image)
           
              if not asf:
                  enhanced_image, asf= run_all_processes(enhanced_image, savepath)
        
             
              if asf == False:
                  savepath = os.path.join('/Users/edgaryepez/AmericanShip/BoxImg/cannotRead',os.path.basename(image_path))
                  print(savepath)
                  enhanced_image = Image.fromarray(enhanced_image)
                  enhanced_image.save(savepath)
        



              # Apply EasyOCR
              # reader = easyocr.Reader(["en"],gpu = False)
              # result = reader.readtext(crop_img)
              # text = ' '.join(item[1] for item in result)
              # print (text)
              

              # Draw label
              object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
              label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
              label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
              cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
              cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

              detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])


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
PATH_TO_IMAGES='/Users/edgaryepez/AmericanShip/BoxImg/2024_07_23'   # Path to test images folder
PATH_TO_MODEL='/Users/edgaryepez/AmericanShip/tflite1/custom_model_liteBOX/detect.tflite'  # Path to .tflite model file
PATH_TO_LABELS='/Users/edgaryepez/AmericanShip/tflite1/custom_model_liteBOX/labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold=0.1   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
images_to_test = 704  # Number of images to run detection on

# Run inferencing function!
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)

print((ASfound[0]/704)*100, "% of the images were found as AmericanShips")
print("foundmatch ", foundmatch[0]/704)




