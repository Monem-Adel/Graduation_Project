import cv2 as cv2

import FinalFile as ff



class Model:

        def __init__(self, image_path) :
            self.image_path = image_path 
            image = cv2.imread(image_path)
            self.image = cv2.resize(image, (500, 500))
            if self.image is None:
                raise Exception("Unable to read image from file path {self.image_path}")

        def __init__(self, image : cv2.Image) :
            self.image = cv2.resize(image, (500, 500))
            if self.image is None:
                 raise Exception("Unable to read image from file path {self.image_path}")


        def get_states_array(self):
            
            
            states = ff.classify_and_extract_states(self.image, ff.model_yolo, ff.model_classification, ff.class_names)
            
            filtered_states = []

            for state in states:
                if state['type'] in ['Start_State', 'State', 'Final_State']:
                    filtered_states.append(state)
            for state in filtered_states:
                x0, y0, x1, y1 = state['bbox']
                cropped_state_img = self.image[y0:y1, x0:x1]
                if state['type'] == 'Start_State':
                    cropped_text_img = ff.crop_image_with_percentagesSS(cropped_state_img, ff.top_percentSS, ff.bottom_percentSS, ff.left_percentSS, ff.right_percentSS)
                elif state['type'] == 'State':
                    cropped_text_img = ff.crop_image_with_percentagesS(cropped_state_img, ff.top_percentS, ff.bottom_percentS, ff.left_percentS, ff.right_percentS)
                elif state['type'] == 'Final_State':
                    cropped_text_img = ff.crop_image_with_percentagesFS(cropped_state_img, ff.top_percentFS, ff.bottom_percentFS, ff.left_percentFS, ff.right_percentFS)
                extracted_text = ff.extract_text_from_image(cropped_text_img)
                state['Label'] = ff.extracted_text
            
            return filtered_states
        
        def get_transitions_array(self):

        
            trans = ff.classify_and_extract_trans(self.image,ff.model_yolo,ff.model_classification,ff.class_names)
            filtered_trans = []
            x = 10

            for tran in trans:
                if tran['type'] in ['Transition', 'Loop']:
                    filtered_trans.append(tran)
            for tran in filtered_trans:
                x0, y0, x1, y1 = tran['bbox']
                cropped_state_image = self.image[y0:y1, x0:x1]
                if tran['type'] == 'Transition':
                    _,cropped_text_image = ff.crop_transition(cropped_state_image,x)
                    croppedimage,_ = ff.crop_transition(cropped_state_image,x)
                    binary_matrix = ff.image_to_binary_matrix(croppedimage)
                    topmost, leftmost, rightmost = ff.find_top_left_right_most(binary_matrix)
                    direction = ff.determine_direction(topmost, leftmost, rightmost)
                    if direction == "left":
                        a = "left"
                    elif direction == "right":
                        a = "right"
                    else:
                        a = "Can't detect arrow dir"
                elif tran['type'] == 'Loop':    
                    _,cropped_text_image = ff.crop_transition(cropped_state_image,x)
                extracted_textt = ff.extract_text_from_image(cropped_text_image)
                arrowdir = a
                tran['Label'] = extracted_textt
                if tran['type'] == 'Transition':
                    tran['Direction'] = arrowdir
            return filtered_trans 
        
