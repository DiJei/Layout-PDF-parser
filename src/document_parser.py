import cv2
import numpy as np
from PIL import Image
from pdf2image import pdfinfo_from_path, convert_from_path
import concurrent.futures
import os
import re
import pytesseract

# Core Modules
from .ocr.ocr import  OCR
from .layout_parser.layout_parser import LayoutParser
from .doc_classifier.doc_classifier import DocumentClassifier

class DocumentParser():
    
    def __init__(self, ocr_thres = 51, path_classifier_model = "doc_classifier/model/model.csv", roi_tables = True):
        self.layout_parser = LayoutParser(roi_tables = roi_tables)
        self.ocr = OCR(ocr_thres = ocr_thres)  
        try:
            self.doc_classifier = DocumentClassifier(path_classifier_model)
        except:
            self.doc_classifier = None
            print('log: could not load doc classifier model')
        
        
    def pre_processing_pdf(self, path_to_doc):
        """Prepare pdf file for layout parser
        Args:
            path_to_doc (str): path to pdf file
        Returns:
            [list]: list of BGR image numpy array
        """
        img_pages = []
        pages = []
        info = pdfinfo_from_path(path_to_doc, userpw=None, poppler_path=None)
        maxPages = info["Pages"]
        pace = 5
        for page in range(1, maxPages+1, pace):
            try:
                pages += convert_from_path(path_to_doc, size=(4136, None), dpi=500, first_page=page, last_page = min(page+pace-1,maxPages))
            except:
                pass
        #pages = convert_from_path(path_to_doc, 500)
        
        for page in pages:
            image = np.array(page) 
            # Convert RGB to BGR 
            image = image[:, :, ::-1].copy()
            
            img_pages.append(cv2.resize(image, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA))
            
        return img_pages

    def rotate_page(self, page):
        """Rotate image page if necessary
        Args:
            image (numpy array): BGR image numpy array
        Returns:
            image (numpy array): BGR image numpy array
        """
        try:
            rot_data = pytesseract.image_to_osd(page)
            rot = re.search('(?<=degrees: )\d+', rot_data).group(0)
        except:
            rot = '0'
        
        if rot == '90':
            rotated = cv2.rotate(page, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rot == '180':
            rotated = cv2.rotate(page, cv2.ROTATE_180)
        elif rot == '270':
            rotated = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated = page
        return rotated
    
    
    def build_json_object(self, image, page):
        """Given a page image run layout parser and ocr building json object of page
        Args:
            image (numpy array): BGR image numpy array
            page (int): page number of the pdf file
        Returns:
            [list]: list of dict with text position, size, and ocr result.
            [int]: page of pdf file.
        """
        image = self.rotate_page(image)
        block_text = self.layout_parser.get_contour(image)
        result = self.ocr.read_block_texts(block_text, image)
        return (result, page)
    
    def generate_json(self, path_to_doc):
        """Run layout parser and ocr over a single pdf file.
        Args:
            path_to_doc (str): path to pdf file.
        Returns:
            [dict]: dict key are page number (starting with 1) with texts elements.
        """
        os.environ['OMP_THREAD_LIMIT'] = '1'
        img_pages = self.pre_processing_pdf(path_to_doc)
        json_object = {}
        page = 1
        blocks = []
        new_image = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for img_page in img_pages:
                futures.append(executor.submit(self.build_json_object, image=img_page, page=page))
                page += 1
            for future in concurrent.futures.as_completed(futures):
                json_object[future.result()[1]] = future.result()[0]

        return json_object
    
    def run_classifier(self, json_object):
        """Run doc classifier model and return class label:
        Args:
            json_object (dict): pre formated data from layout parser and ocr
        Returns:
            (str): class document label
        """
        
        # No model load --> no result
        if not self.doc_classifier:
            return ''
        
        
        res, vector = self.doc_classifier.run(json_object)
        return max(res, key = res.get)