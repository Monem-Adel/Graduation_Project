from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pytesseract import pytesseract 
import re
#import sys
#sys.path.append(r"D:\\4th\\Second Sem\\Graduation Project")
#from DetectArrowDir import image_to_binary_matrix

class_names = ['Final_State', 'Loop','Start_State', 'State', 'Transition']

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


def classify_image(image, model, class_names):
    image_pil = Image.fromarray(image)
    
    image_transformed = data_transform(image_pil).unsqueeze(0) 
    
    model.eval() 
    with torch.no_grad():
        output = model(image_transformed)
    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    return class_names[class_index]


def crop_transition(image, crop_index):
    #mage = cv2.imread(image_path)

    #if image is None:
        #print("Failed to load image.")
        #return

    # Crop the image from the top to the specified index
    cropped_image = image[crop_index:, :]
    remaining_image = image[:crop_index, :]

    #if not os.path.exists(output_folder):
        #os.makedirs(output_folder)

    #cv2.imwrite(os.path.join(output_folder, "cropped_image.jpg"), cropped_image)
    #cv2.imwrite(os.path.join(output_folder, "remaining_image.jpg"), remaining_image)

    return cropped_image , remaining_image


def classify_and_extract_states(image, model_yolo, model_classification, class_names):
    states = []

    
    results = model_yolo.predict(source=image, conf=0.25)

    
    for result in results:
        for x0, y0, x1, y1 in result.boxes.xyxy:
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)

            
            cropped_image = image[y0:y1, x0:x1]

            
            predicted_class = classify_image(cropped_image, model_classification, class_names)

            
            if predicted_class in ['Start_State', 'State', 'Final_State']:
                states.append({'type': predicted_class, 'bbox': (x0, y0, x1, y1)})

    return states

# OCR
def extract_text_from_image(image):
     
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    def image_to_binary(image, threshold=127):
        
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        _, binary_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)

        return binary_image

    def sharpen_objects(binary_image, iterations=3):
        
        laplacian_kernel = np.array([[1, 1, 1],
                                     [1, -9, 1],
                                     [1, 1, 1]], dtype=np.float32)
        for _ in range(iterations):

            
            sharpened_image = cv2.filter2D(binary_image, -1, laplacian_kernel)

            
            sharpened_image = np.clip(sharpened_image, 0, 255)

            
            sharpened_image = np.uint8(sharpened_image)

            return sharpened_image

    def remove_special_characters(text):
        
        pattern = r"[^\w\s]" 

        
        cleaned_text = re.sub(pattern, "", text)

        return cleaned_text

    def place_binary_image(binary_image, output_size=(500, 500)):
        
        background = np.ones((output_size[1], output_size[1]), dtype=np.uint8)

        
        bg_height, bg_width = background.shape
        bin_height, bin_width = binary_image.shape
        x_offset = (bg_width - bin_width) // 2
        y_offset = (bg_height - bin_height) // 2

        
        background[y_offset:y_offset+bin_height, x_offset:x_offset+bin_width] = binary_image

        return background

    
    threshold_value = 127  
    binary_image = image_to_binary(image, threshold=threshold_value)

    
    sharpened_image = sharpen_objects(binary_image, iterations=5)

    
    placed_image = place_binary_image(sharpened_image, output_size=(500, 500))

    
    pil_binary_image = Image.fromarray(placed_image)

     
    pytesseract.tesseract_cmd = path_to_tesseract 

    
    text = pytesseract.image_to_string(pil_binary_image, lang='eng', config='--psm 10')

    
    cleaned_text = remove_special_characters(text.strip())

    return cleaned_text
#===============================================================================================
def classify_and_extract_trans(image,model_yolo,model_classification,class_names):
    trans =[]
    resultss = model_yolo.predict(source=img,conf=0.25)
        
    for result in resultss:
        for x0, y0, x1, y1 in result.boxes.xyxy:
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)

            
            cropped_image = image[y0:y1, x0:x1]

            
            predicted_class = classify_image(cropped_image, model_classification, class_names)

            
            if predicted_class in ['Transition', 'Loop']:
                trans.append({'type': predicted_class, 'bbox': (x0, y0, x1, y1)})

    return trans
#=================================================================================================
# Crop Start_State image to extract name of state from it
def crop_image_with_percentagesSS(image, top_percentSS, bottom_percentSS, left_percentSS, right_percentSS):
    height, width = image.shape[:2]
    
    
    top_crop = int(height * top_percentSS)
    bottom_crop = int(height * (1 - bottom_percentSS))
    left_crop = int(width * left_percentSS)
    right_crop = int(width * (1 - right_percentSS))
    
    
    cropped_image = image[top_crop:bottom_crop, left_crop:right_crop]
    
    return cropped_image

# Crop Normal state image to extract name of state from it
def crop_image_with_percentagesS(image, top_percentS, bottom_percentS, left_percentS, right_percentS):
    height, width = image.shape[:2]
    
    
    top_crop = int(height * top_percentS)
    bottom_crop = int(height * (1 - bottom_percentS))
    left_crop = int(width * left_percentS)
    right_crop = int(width * (1 - right_percentS))
    
    
    cropped_image = image[top_crop:bottom_crop, left_crop:right_crop]
    
    return cropped_image

# Crop Final state image to extract name of state from it
def crop_image_with_percentagesFS(image, top_percentFS, bottom_percentFS, left_percentFS, right_percentFS):
    height, width = image.shape[:2]
    
    
    top_crop = int(height * top_percentFS)
    bottom_crop = int(height * (1 - bottom_percentFS))
    left_crop = int(width * left_percentFS)
    right_crop = int(width * (1 - right_percentFS))
    
    
    cropped_image = image[top_crop:bottom_crop, left_crop:right_crop]
    
    return cropped_image

def image_to_binary_matrix(image):
    
    if len(image.shape) > 2:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image
        
    
    _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
    
    
    binary_image = cv2.bitwise_not(binary_image)
    
    
    binary_image[binary_image == 255] = 1
    
    
    binary_matrix = np.array(binary_image, dtype=int)
    
    return binary_matrix

def compute_column_sum(binary_matrix):
        column_sums = np.sum(binary_matrix, axis=0)
        max_sum_index = np.argmax(column_sums)
        return column_sums, max_sum_index

def find_top_left_right_most(binary_matrix):
        
        column_sums, max_sum_index = compute_column_sum(binary_matrix)
        topmost_row = np.argmax(binary_matrix[:, max_sum_index])
        topmost_col = max_sum_index

        
        leftmost_row, leftmost_col = np.unravel_index(np.argmax(binary_matrix), binary_matrix.shape)[::-1]
        for row in range(binary_matrix.shape[0]):
            for col in range(binary_matrix.shape[1]):
                if binary_matrix[row, col] == 1:
                    if col < leftmost_col:
                        leftmost_row, leftmost_col = row, col

        
        rightmost_row, rightmost_col = np.unravel_index(np.argmax(binary_matrix), binary_matrix.shape)[::-1]
        for row in range(binary_matrix.shape[0]):
            for col in range(binary_matrix.shape[1]):
                if binary_matrix[row, col] == 1:
                    if col > rightmost_col:
                        rightmost_row, rightmost_col = row, col

        return (topmost_row, topmost_col), (leftmost_row, leftmost_col), (rightmost_row, rightmost_col)

def euclidean_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

def determine_direction(topmost, leftmost, rightmost):
        
        distance_to_left = euclidean_distance(topmost, leftmost)
        distance_to_right = euclidean_distance(topmost, rightmost)

        if distance_to_left < distance_to_right:
            return "left"
        elif distance_to_right < distance_to_left:
            return "right"
        else:
            return "neither"

def binary_image_size(binary_image):
        height, width = binary_image.shape
        return height, width


model_yolo = YOLO('SegModel.pt')
model_classification = Net()
model_classification.load_state_dict(torch.load('classification_model_weights.pth'))
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
])
image_dir = '.\\jflab'

# Values for Start_State
top_percentSS = 0.3
bottom_percentSS = 0.3
left_percentSS = 0.4
right_percentSS = 0.1

# Values for State
top_percentS = 0.2
bottom_percentS = 0.1
left_percentS = 0.2
right_percentS = 0.1

# Values for Final_State
top_percentFS= 0.2
bottom_percentFS = 0.2
left_percentFS = 0.2
right_percentFS = 0.2

for dirname, _, filenames in os.walk(image_dir):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        _, file_ext = os.path.splitext(filename)
        if file_ext.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        
        
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Unable to read image from {file_path}")
            continue
        
        
        img = cv2.resize(img, (500, 500))
        states = classify_and_extract_states(img, model_yolo, model_classification, class_names)
        trans = classify_and_extract_trans(img,model_yolo,model_classification,class_names)
        
        filtered_states = []

        for state in states:
            if state['type'] in ['Start_State', 'State', 'Final_State']:
                filtered_states.append(state)
        for state in filtered_states:
            x0, y0, x1, y1 = state['bbox']
            cropped_state_img = img[y0:y1, x0:x1]
            if state['type'] == 'Start_State':
                cropped_text_img = crop_image_with_percentagesSS(cropped_state_img, top_percentSS, bottom_percentSS, left_percentSS, right_percentSS)
            elif state['type'] == 'State':
                cropped_text_img = crop_image_with_percentagesS(cropped_state_img, top_percentS, bottom_percentS, left_percentS, right_percentS)
            elif state['type'] == 'Final_State':
                cropped_text_img = crop_image_with_percentagesFS(cropped_state_img, top_percentFS, bottom_percentFS, left_percentFS, right_percentFS)
            extracted_text = extract_text_from_image(cropped_text_img)
            state['Label'] = extracted_text
        print(f"Array of States: {filtered_states}")
        
        
        filtered_trans = []
        x = 10

        for tran in trans:
            if tran['type'] in ['Transition', 'Loop']:
                filtered_trans.append(tran)
        for tran in filtered_trans:
            x0, y0, x1, y1 = tran['bbox']
            cropped_state_image = img[y0:y1, x0:x1]
            if tran['type'] == 'Transition':
                _,cropped_text_image = crop_transition(cropped_state_image,x)
                croppedimage,_ = crop_transition(cropped_state_image,x)
                binary_matrix = image_to_binary_matrix(croppedimage)
                topmost, leftmost, rightmost = find_top_left_right_most(binary_matrix)
                direction = determine_direction(topmost, leftmost, rightmost)
                if direction == "left":
                    a = "left arrow"
                elif direction == "right":
                    a = "right arrow"
                else:
                    a = "Can't detect arrow dir"
            elif tran['type'] == 'Loop':    
                _,cropped_text_image = crop_transition(cropped_state_image,x)
            extracted_textt = extract_text_from_image(cropped_text_image)
            arrowdir = a
            tran['Label'] = extracted_textt
            if tran['type'] == 'Transition':
                tran['Direction'] = arrowdir
        print(f"Array of Transitions: {filtered_trans}")
#============================================================================================-