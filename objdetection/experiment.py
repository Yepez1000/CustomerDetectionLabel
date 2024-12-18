import os
from flask import Flask, request, jsonify, send_file, send_from_directory, current_app
from PIL import Image
from io import BytesIO
import random
import glob
import numpy as np
import time
from tensorflow.lite.python.interpreter import Interpreter
from werkzeug.utils import secure_filename
import cv2
import pytesseract
import re
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image, ImageEnhance, ImageFilter
from playsound import playsound
from datetime import datetime
from flask_cors import CORS  # Import COR
#from pyzbar.pyzbar import decode
from pyzbar import pyzbar
import requests
import json
import logging
import pygame



# df = pd.read_csv('static/CustomersView.csv')

# customer_data = {}



df = pd.read_csv('/Users/edgaryepez/AmericanShip/tflite1/objdetection/CustomersView.csv')
df['Unique STE # AS-'] = df['Unique STE # AS-'].astype(str)
FirstNamedict = {}
LastNamedict = {}
ASdict = {}


for index, row in df.iterrows():
    
    FirstNamedict[row['First Name']] = [row['Last Name'], row['Unique STE # AS-']]
    LastNamedict[row['Last Name']] = [row['First Name'], row['Unique STE # AS-']]
    ASdict[row['Unique STE # AS-']] = [row['First Name'], row['Last Name']]




barcodeFound = False
matchfound = False
customerFound = False
theName = ""
theAs = ""

# pygame.init()
# pygame.mixer.init()
# confirm2 = pygame.mixer.Sound("static/confirm2.wav")
# packageFound = pygame.mixer.Sound("static/packageFound.wav")


found_match_count = [0]




def detectCustomerAndTracking(extracted_text):
        global customerFound, matchfound, theName, theAs
        
        
        
        start_time = time.time() 

            
        for client_id in customer_data:

         

            customerIdStr = "AS"+str(client_id)
            customerIdStr= customerIdStr.lower()
          
            if(customerIdStr in extracted_text):
             
                customer = str(customer_data[client_id])
                customer = customer.lower()
                print("customerIdStr ", customerIdStr)
                print("Customer ",customer)
                if(customer in extracted_text):
                    print("CUSTOMER FOUND: " + customerIdStr + " " + customer_data[client_id])
                    
                    
                    customerFound = True
                    confirm2.play()
                    theName = customer
                    theAs = customerIdStr
                        
       
        if customerFound:
            print("ALL INFORMATION FOUND")
            packageFound.play()
            matchfound= True
            
        else:
            print("NOT FOUND")
            
        end_time = time.time()  # Record the end time 
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Elapsed time to decode image: {elapsed_time} seconds") 
            
        return matchfound, theName , theAs

def new_found_AS_and_Match(text):

    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)

    words = text.split()

    print("this is words",words)

    # Extract first name (assuming it's the first word)
    if words:
        first_name = words[0]

    # Convert words list to string for further processing
    text_str = ' '.join(words)

    print("this is text_str",text_str)

    # Extract AS number using regex
    as_number_match = re.search(r'AS\d{5}', text_str)
    if as_number_match:
        as_number = int(as_number_match.group()[2:])



    # Ensure 'Unique STE # AS-' column is of string type
    df['Unique STE # AS-'] = df['Unique STE # AS-'].astype(str)

    # Query DataFrame for matching first name
    potential_matches = df[df['First Name'].str.lower() == first_name.lower()]

    # Further refine matches by last name
    for _, row in potential_matches.iterrows():
        last_name = row['Last Name']
        if re.search(last_name, text_str, re.IGNORECASE):
            return True, row

    # Query DataFrame for matching AS number
    if as_number is not None:
        as_matches = df[df['Unique STE # AS-'].str.contains(str(as_number), na=False)]
        if not as_matches.empty:
            first_name = as_matches['First Name']
            last_name = as_matches['Last Name']
            
            # Create a regex pattern to search for the names
            pattern2 = rf"{first_name} "
            pattern3 = rf"{last_name}"

            # Check if the names are found in the text
            if re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE):
                print("Found a match!")
                return True, as_matches

    print("No match found!")
    return False, ""


#time complexity too long

def found_AS_and_Match(text):

    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)

    print(text)

   
    for word in text.split():

        if word.isnumeric() and len(word) < 5:
            continue

        if word[:2] == "AS":
            word = word[2:]

        matches = df[
            # (df['First Name'].str.lower() == word.lower()) | 
            # (df['Last Name'].str.lower() == word.lower()) |
            (df['Unique STE # AS-'].str.contains(word, na=False))
        ]
        if matches.empty:
            continue
        print(matches)
        

        # for row in matches.iterrows():
        #     # last_name = row['Last Name']
        #     print('\n')
        #     print(word,"\n",row)

       
       

        # matches = df[df['First Name'].str.lower() == first_name.lower()]
     
        # for _, row in matches.iterrows():
        #     if row is not np.nan:
        #         print(row)



    return False


def matching(text):
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)



    for word in text.split():
        if word[:2] == "AS":
            word = word[2:]

        if word in FirstNamedict:       
            lastname,uniqueid = FirstNamedict[word]
            pattern2 = rf"{lastname} "
            pattern3 = rf"{uniqueid}"

            # Check if the names are found in the text
            if re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE):
                print("Found a match!",  word, lastname, uniqueid)
                return True

        if word in LastNamedict:

            firstname, uniqueid = LastNamedict[word]

            pattern2 = rf"{firstname} "
            pattern3 = rf"{uniqueid}"

            # Check if the names are found in the text
            if re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE):
                print("Found a match!",  word, firstname, uniqueid)
                return True
           

            
        if word in ASdict:
            firstname,lastname = ASdict[word]

            pattern2 = rf"{firstname} "
            pattern3 = rf"{lastname}"

            # Check if the names are found in the text
            if re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE):
                print("Found a match!",  word, firstname, lastname)
                return True
         

     
text = """ BRINGGO SHIP LLC    BRGO543
2450 COURAGE ST STE 108

   AS14016 BGMX75720
BROWNSVILLF TX78521 5134"""

found_AS_and_Match(text)