import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import re

class OCR:
    
    def __init__(self, ocr_thres = 60):
        self.ocr_thres = ocr_thres

    def get_grayscale(self, image):
        """Convert pdf image to binary and invert background and font colors

        Args:
            image (numpy array): BGR numpy array of pdf page

        Returns:
            [numpy array]: binary image of pdf page
        """
        temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Try to filter highlighter marks
        return temp[:,:,2]

    def prepare_for_ocr(self, img):
        """Prepare img for ocr
        Args:
            img (numpy array): BGR numpy array of pdf page.
        Returns:
             [numpy array] : processed image
        """
        gray = self.get_grayscale(img)
        ret, thresh = cv2.threshold(gray, 190, 255, 2)
        minimum = int(np.min(thresh))
        thresh = cv2.subtract(thresh, minimum)
        thresh[np.where(thresh >= 190-minimum)] = 255
        #thresh = cv2.multiply(thresh, 0.8)
        #thresh[np.where(thresh >= 178)] = 255
        ret, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #ret, thresh = cv2.threshold(gray, 200, 255, 2)
        #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)
        area = int(gray.shape[0] * gray.shape[1])
        if area > 6000:
            ocr_img = cv2.resize(thresh, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        else:
            img_ocr = cv2.resize(thresh, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
            kernel = np.ones((5,5),np.uint8)
            ocr_img =  cv2.dilate(img_ocr, kernel, iterations = 1)
        return ocr_img
    
    
    def read_block_texts(self, blocks, image, thes = 25):
        """Given a page single image generate json obj.
        Args:
            img (numpy array): BGR numpy array of pdf page.
            blocks (list): list of bounding boxes
            thres (int): digital margin 
        Returns:
             [dict] : dict with ocr result, text postion and text size
        """
        json_obj = []
        max_x = image.shape[1]
        max_y = image.shape[0]
        
        #tessract option
        custom_config = r'--oem 3 --psm 6'
        
        # blocks
        block_id = 1
        for block in blocks:
            
            if block['type'] == 'text':
                group = block['bb']
                
                #expand bb for better ocr result 
                if group[0] - thes < 0:
                    xa = 0
                else:
                    xa =  group[0] - thes
                
                if group[2] + thes > max_x:
                    xb = max_x
                else:
                    xb =  group[2] + thes
                
                if group[1] - thes < 0:
                    ya = 0
                else:
                    ya =  group[1] - thes
                    
                if group[3] + thes > max_y:
                    yb = max_y
                else:
                    yb =  group[3] + thes
                
                ocr_img = self.prepare_for_ocr(image[ya:yb, xa:xb])
                words = pytesseract.image_to_data(ocr_img, output_type=Output.DICT,  lang = 'por')
                text = ''
                n_boxes = len(words['text'])
                
                # jump whitespaces
                x = 0
                for i in range(n_boxes):
                    only_cha = re.sub("[ \n]", "", words['text'][i])
                    if len(only_cha) > 0:
                        break
                    x += 1
                    
                if x == n_boxes:
                    continue
                        
                thes_y = np.quantile(words['height'], 0.51)
                height = words['top'][x] +  words['height'][x]
                for i in range(n_boxes):
                    if int(words['conf'][i]) >= self.ocr_thres:
                        height_w = words['top'][i] +  words['height'][i]
                        if np.abs(height - height_w) > thes_y:
                            text += '\n' + words['text'][i] + ' '
                        else:
                            text += words['text'][i] + ' '
                        height = height_w
                
                only_cha = re.sub("[^A-Za-z0-9]", "", text)
                if len(only_cha) >= 2:
                    width  = np.abs(group[0] - group[2])
                    height = np.abs(group[1] - group[3])
                    json_obj.append({"type":"text","x":int(group[0]), "y":int(group[1]), 
                                     "width":int(width), "height":int(height), "text":text.strip(), 'id':block_id})
                    block_id += 1
            elif block['type'] == 'table':
                group = block['bb']
                width  = np.abs(group[0] - group[2])
                height = np.abs(group[1] - group[3])
                json_dict = {'id':block_id, "type":"table","x":int(group[0]), "y":int(group[1]), "width":int(width), "height":int(height)}
                
                cells = []
                cell_id = 1
                for cell in block['cells']:
                    xa,ya,xb,yb =  cell
                    ocr_img = self.prepare_for_ocr(image[ya:yb, xa:xb])
                    words = pytesseract.image_to_data(ocr_img, output_type=Output.DICT,  lang = 'por')
                    text = ''
                    n_boxes = len(words['text'])
                    
                    # jump whitespaces
                    x = 0
                    for i in range(n_boxes):
                        only_cha = re.sub("[ \n]", "", words['text'][i])
                        if len(only_cha) > 0:
                            break
                        x += 1
                    
                    if x  == n_boxes:
                        continue
                    
                    thes_y = np.quantile(words['height'], 0.51)
                    height = words['top'][x] +  words['height'][x]
                    for i in range(x, n_boxes):
                        
                        # jump white space
                        if len(re.sub("[ \n]", "", words['text'][i])) < 1:
                            continue
                        
                        # check conf
                        if int(words['conf'][i]) > self.ocr_thres:
                            height_w = words['top'][i] +  words['height'][i]
                            if np.abs(height - height_w) > thes_y:
                                text += '\n' + words['text'][i] + ' '
                            else:
                                text += words['text'][i] + ' '
                            height = height_w
                    
                    only_cha = re.sub("[^A-Za-z0-9]", "", text)
                    if len(only_cha) >= 2:
                        width  = np.abs(group[0] - group[2])
                        height = np.abs(group[1] - group[3])
                        cell_dict = {'id':cell_id, "type":"cell", "x":int(group[0]), "y":int(group[1]),
                                     "width":int(width), "height":int(height), 'text':text.strip()}
                        cell_id += 1
                        cells.append(cell_dict)
                if len(cells) > 0:
                    cells.reverse()
                    json_dict['cells'] = cells
                    json_obj.append(json_dict)
                    block_id += 1
                    
        # Order by id
        json_obj = sorted(json_obj , key = lambda k: [k["id"]])
        return json_obj
