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
import easyocr
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

     
    path_to_tesseract = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    
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
#EasyOCR
def OCR(image): 
    def zoom(img, zoom_factor=2.5):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    #blur = cv2.GaussianBlur(image,(5,5),0)
    #image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(f"Image loaded successfully. Shape: {gray.shape}")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 10, 200)
    zoomed_edges = zoom(edged, 2.5)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(zoomed_edges)
    for (bbox, text, prob) in result:
        return text
#####################################################################################
def enhance_and_extract_text(image,zoom_factor):
    def zoom(img, zoom_factor):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

    # Load the image
    #image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thresh, 30, 200)
    zoomed_edgess = cv2.resize(image, (1080, 1920))
    zoomed_edges = zoom(zoomed_edgess, zoom_factor)
    #zoomed_edges = cv2.resize(image, (1080, 1920))
    reader = easyocr.Reader(['en'])
    result = reader.readtext(zoomed_edges)
    extracted_text = ""
    for (bbox, text, prob) in result:
        extracted_text += text + " "
    return extracted_text.strip()

#####################################################################################
def OCRT(image): 
    def zoom(img, zoom_factor=4):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    #blur = cv2.GaussianBlur(image,(5,5),0)
    #image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(f"Image loaded successfully. Shape: {gray.shape}")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 10, 200)
    zoomed_edges = zoom(edged, 4)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(zoomed_edges)
    for (bbox, text, prob) in result:
        return text

def OCRL(image): 
    def zoom(img, zoom_factor=1):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    #blur = cv2.GaussianBlur(image,(5,5),0)
    #image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(f"Image loaded successfully. Shape: {gray.shape}")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 10, 200)
    zoomed_edges = zoom(edged, 1)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(zoomed_edges)
    for (bbox, text, prob) in result:
        return text

def OCRSS(image): 
    def zoom(img, zoom_factor=4):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    #blur = cv2.GaussianBlur(image,(5,5),0)
    #image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(f"Image loaded successfully. Shape: {gray.shape}")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 10, 200)
    zoomed_edges = zoom(edged, 4)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(zoomed_edges)
    for (bbox, text, prob) in result:
        return text
#===============================================================================================
def classify_and_extract_trans(img,model_yolo,model_classification,class_names):
    trans =[]
    resultss = model_yolo.predict(source=img,conf=0.25)
        
    for result in resultss:
        for x0, y0, x1, y1 in result.boxes.xyxy:
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)

            
            cropped_image = img[y0:y1, x0:x1]

            
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
    # Convert the image to grayscale
    if len(image.shape) > 2:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image
    
    # Apply a binary threshold to the grayscale image
    _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

    # Invert the binary image
    binary_image = cv2.bitwise_not(binary_image)
    
    # Convert binary image to a binary matrix
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

def compute_row_sum(binary_matrix):
        row_sums = np.sum(binary_matrix, axis=1)
        max_sum_index = np.argmax(row_sums)
        return row_sums, max_sum_index

def get_topmost_and_bottommost_index(binary_matrix):
        topmost_coordinate = None
        bottommost_coordinate = None
    
        # Iterate over rows from top to bottom for finding the top-most '1'
        for i in range(len(binary_matrix)):
            indices = np.where(binary_matrix[i] == 1)[0]
            if len(indices) > 0:
                topmost_coordinate = (i, indices[0])
                break
    
        # Iterate over rows from bottom to top for finding the bottom-most '1'
        for i in range(len(binary_matrix) - 1, -1, -1):
            indices = np.where(binary_matrix[i] == 1)[0]
            if len(indices) > 0:
                bottommost_coordinate = (i, indices[-1])
                break
    
        return topmost_coordinate, bottommost_coordinate

def get_rightmost_in_highest_row(binary_matrix):
        row_sums, max_sum_index = compute_row_sum(binary_matrix)
        highest_row = binary_matrix[max_sum_index]
        indices = np.where(highest_row == 1)[0]
        if len(indices) > 0:
            any_index = indices[0]  # You can choose any index. Here, we are choosing the first one.
            return max_sum_index, any_index
        else:
            return max_sum_index, None

def euclidean_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

def compute_closest_distance(binary_matrix):
        topmost_coordinate, bottommost_coordinate = get_topmost_and_bottommost_index(binary_matrix)
        highest_row_index, any_index_in_highest_row = get_rightmost_in_highest_row(binary_matrix)
        any_index_coordinate = (highest_row_index, any_index_in_highest_row)

        # Compute distances
        distance_to_topmost = euclidean_distance(any_index_coordinate, topmost_coordinate)
        distance_to_bottommost = euclidean_distance(any_index_coordinate, bottommost_coordinate)

        # Determine which one is nearer
        if distance_to_topmost < distance_to_bottommost:
            nearer = "UP"
            top = topmost_coordinate
            bottom = bottommost_coordinate
            nearer_distance = distance_to_topmost
            head = top 
            tail = bottom
            return nearer,head,tail 
        else:
            nearer = "Down"
            top = topmost_coordinate
            bottom = bottommost_coordinate
            nearer_distance = distance_to_bottommost
            head = bottom
            tail = top
            return nearer,head,tail       

def crop_image_from_left(image, crop_width):
    # Get image dimensions
    height, width, _ = image.shape
    
    # Define the region of interest (ROI) for cropping from the left
    x_start_left = 0
    x_end_left = crop_width
    y_start_left = 0
    y_end_left = height
    
    # Crop the image from the left
    cropped_image_left = image[y_start_left:y_end_left, x_start_left:x_end_left]
    
    # Define the region of interest (ROI) for the remaining part of the image
    x_start_remaining = crop_width
    x_end_remaining = width
    y_start_remaining = 0
    y_end_remaining = height
    
    # Crop the remaining part of the image
    cropped_image_remaining = image[y_start_remaining:y_end_remaining, x_start_remaining:x_end_remaining]
    
    return cropped_image_left, cropped_image_remaining

def binary_image_size(binary_image):
        height, width ,d= binary_image.shape
        return height, width, d

# model_yolo = YOLO("D:\\3loom\\4thYear\\2ndSemester\\GraduationProject\\Graduation_Project\\Image_Processing\\SegModel.pt")
model_yolo = YOLO(r'H:\Graduation Project\Graduation_Project\Image_Processing\65v2.pt')
model_classification = Net()
# model_classification.load_state_dict(torch.load("D:\\3loom\\4thYear\\2ndSemester\\GraduationProject\\Graduation_Project\\Image_Processing\\classification_model_weights.pth"))
model_classification.load_state_dict(torch.load(r'H:\Graduation Project\Graduation_Project\Image_Processing\ClassificationModel_weights.pth'))
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
])
# image_dir = '.\\jflab'

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

# for dirname, _, filenames in os.walk(image_dir):
#     for filename in filenames:
#         file_path = os.path.join(dirname, filename)
#         _, file_ext = os.path.splitext(filename)
#         if file_ext.lower() not in ['.png', '.jpg', '.jpeg']:
#             continue
        
        
#         img = cv2.imread(file_path)
#         if img is None:
#             print(f"Error: Unable to read image from {file_path}")
#             continue
##################### WARNING: be carful while editing here cause it related with scanner file ##############
def preparing_img(img_path):
    image = cv2.imread(img_path,1)
    if image is None:
        raise ValueError(f"Error: Unable to read image from {img_path}")
    return image

def get_classified_States(img_path):
        img = preparing_img(img_path)
        print(img.shape)
        zoom_factorr = 0.5
        #img = img_path
        states = classify_and_extract_states(img, model_yolo, model_classification, class_names)
        filtered_states = []

        for state in states:
            if state['type'] in ['Start_State', 'State', 'Final_State']:
                filtered_states.append(state)
        for state in filtered_states:
            x0, y0, x1, y1 = state['bbox']
            cropped_state_img = img[y0:y1, x0:x1]
            if state['type'] == 'Start_State':
                cropped_text_img = enhance_and_extract_text(cropped_state_img,zoom_factorr)
            elif state['type'] == 'State':
                cropped_text_img = enhance_and_extract_text(cropped_state_img,zoom_factorr)
            elif state['type'] == 'Final_State':
                cropped_text_img = enhance_and_extract_text(cropped_state_img,zoom_factorr)
            extracted_text = cropped_text_img
            state['name'] = extracted_text
        return filtered_states
#=======================================================================================
#=========================================================================================
def get_classified_transitions(img_path):
      try:
        img = preparing_img(img_path)
        print("Image shape:", img.shape) 
        a = None
        head_coord = None
        tail_coord = None
        img = preparing_img(img_path)
        trans = classify_and_extract_trans(img,model_yolo,model_classification,class_names)
        filtered_trans = []
        x = 25
        crop_width_value = 29
        zoom_factorr = 0.5
        for tran in trans:
            if tran['type'] in ['Transition', 'Loop']:
                filtered_trans.append(tran)
        for tran in filtered_trans:
            x0, y0, x1, y1 = tran['bbox']
            cropped_trans_image = img[y0:y1, x0:x1]
            #print(cropped_trans_image.shape)
            if tran['type'] == 'Transition':
                h , w ,d= binary_image_size(cropped_trans_image)
                if w > h:
                    _,cropped_text_image = crop_transition(cropped_trans_image,x)
                    croppedimage,_ = crop_transition(cropped_trans_image,x)
                    binary_matrix = image_to_binary_matrix(croppedimage)
                    topmost, leftmost, rightmost = find_top_left_right_most(binary_matrix)
                    direction = determine_direction(topmost, leftmost, rightmost)
                    extracted_textt = enhance_and_extract_text(cropped_trans_image,zoom_factorr)
                    tran['Label'] = extracted_textt
                    if direction == "left":
                        a = "Left"
                        head_coord = leftmost
                        tail_coord = rightmost
                    elif direction == "right":
                        a = "Right"
                        head_coord = rightmost
                        tail_coord = leftmost
                    else:
                        a = "Can't detect arrow dir"
                elif h > w:
                    _,cropped_arrow_image = crop_image_from_left(cropped_trans_image,crop_width_value)
                    binary_matrix = image_to_binary_matrix(cropped_arrow_image)
                    directionoftrans,head_coord , tail_coord = compute_closest_distance(binary_matrix)
                    extracted_textt = enhance_and_extract_text(cropped_trans_image,zoom_factorr)
                    a = directionoftrans
                    tran['Label'] = extracted_textt

            elif tran['type'] == 'Loop':    
                #_,cropped_text_image = crop_transition(cropped_trans_image,x)
                extracted_textt = enhance_and_extract_text(cropped_trans_image,zoom_factorr)
                tran['Label'] = extracted_textt
            arrowdir = a
            head = head_coord
            tail = tail_coord
            if tran['type'] == 'Transition':
                tran['Direction'] = arrowdir
                tran['Head'] = head
                tran['Tail'] = tail
         #print(f"Array of Transitions: {filtered_trans}")
        return filtered_trans
      except Exception as e:
        print("Error during image processing:", e)

# print("array of transition")
# print(get_classified_transitions(r'H:\Graduation Project\Graduation_Project\Drawing_Automatons\auto_2.jpg'))
# print(f"Array of States")
# print(get_classified_States(r'H:\Graduation Project\Graduation_Project\Drawing_Automatons\auto_2.jpg'))
#============================================================================================-
##############################################################################################