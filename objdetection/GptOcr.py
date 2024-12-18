import os
import pandas as pd
import matplotlib.pyplot as plt
import openai
import base64
import re
import random
import glob
import time

# Initialize the OpenAI client with your API key

def found_AS_and_Match(text):
    # Clean up text by removing non-alphanumeric characters (excluding spaces)
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    print("text:\n",text)
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

            matches = df[(df['Unique STE # AS-'].str.contains(client_id, na=False))]


            print(matches)

            
            
            found_match_count[0] += 1
            return True, matches
        

    print("No match found")


    return False, None

def image_to_base64(image_path):
    # Convert an image file to base64 string
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return base64_string

def extract_text_with_gpt(image_path):


    # Process the image (already on disk, or could be processed)
    captcha_image_base64 = image_to_base64(image_path)
    
    # Call the GPT-4 vision model
    try:
        result = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",  # GPT-4 vision model
            messages=[
                {
                    "role": "user",
                    "content": "Where does this packagae ship to, respond with only the text and don't add quotes to it",
                    "additional_message": {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{captcha_image_base64}"
                    }
                }
            ],
            max_tokens=1000  # Adjust the max tokens if necessary
        )
        
        # Extract the text from the result
        extracted_text = result['choices'][0]['message']['content']
        print("Extracted text:", extracted_text)
        found_AS_and_Match(extracted_text)
        return extracted_text
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def get_images(imgpath, num_test_images):
    # List to hold image names
    images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')
    images_to_test = random.sample(images, num_test_images)

    for image_path in os.listdir(images_to_test):
        captcha_text = extract_text_with_gpt(image_path)
        if not captcha_text:
            print("No captcha text found")





openai.api_key = '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'

image_dir = 'Oscar/forEdgar/allImages'

df = pd.read_csv('/Users/edgaryepez/AmericanShip/tflite1/objdetection/CustomersView3.csv')
df['Unique STE # AS-'] = df['Unique STE # AS-'].astype(str)

found_match_count = [0]

start_time = time.time()  # Record the start time
get_images(image_dir)
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Calculate the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")

print(found_match_count[0],"matches found")




