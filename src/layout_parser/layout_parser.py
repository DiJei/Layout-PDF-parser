import cv2
import numpy as np


class LayoutParser:
    
    def __init__(self, roi_tables = True):
        self.roi_tables = roi_tables

    def get_grayscale(self, image):
        """Convert pdf image tget_grayscaleo binary and invert background and font colors

        Args:
            image (numpy array): BGR numpy array of pdf page

        Returns:
            [numpy array]: binary image of pdf page
        """
        temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_not(temp)
    

    def rotate_bound(self, image, angle):
        """Rotate hold the border of image
        
        Args:
            image (numpy array): BGR numpy array of pdf page
            angle (float): angle in degrees
        
        Returns:
            [numpy array]: binary image of pdf page
        """
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))
        rotated = cv2.resize(rotated, (w,h), interpolation = cv2.INTER_AREA)
        (h, w) = rotated.shape[:2]
        return  rotated

    def get_skew_angle(self, img):
        """Given a binary image try to correct rotation.
        
        Args:
            img (numpy array): binary array
        Returns:
            [numpy array]:  binary array
            [float angle]:  angle in degrees
        """

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        #gray = cv2.dilate(gray,kernel,iterations = 2)
        
        # Text White amd background black
        #gray = cv2.bitwise_not(gray)
        #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        filtered = self.filter_doc_contour(img)
        
        coords = np.column_stack(np.where(filtered > 0))
        angle = cv2.minAreaRect(coords)[-1]
        print(f'Angle: {angle}')
        if -95 < angle < -85:
            angle = (90 + angle)
        elif 85 < angle < 95:
            angle = -(90 - angle)
        elif -5 < angle < 5:
            angle *= -1
        else:
            angle = 0
        print(f'new: {angle}')
        angle=0
        return self.rotate_bound(img, angle), angle

    def fix_skew_img(self, img):
        """Given a binary image try to correct rotation.
        
        Args:
            img (numpy array): binary array
        Returns:
            [numpy array]:  binary array
            [float angle]:  angle in degrees
        """
        
        angle = 0

        x = int(img.shape[1] * 0.01)
        y = int(img.shape[0] * 0.01)
                
        kernel = np.ones((1, x), np.uint8)
        img_temp = cv2.dilate(img, kernel, iterations = 1)
        img_temp = cv2.erode(img_temp, kernel, iterations = 1)
        
        kernel = np.ones((y, 1), np.uint8)
        img_temp = cv2.dilate(img_temp, kernel, iterations = 1)
        img_temp = cv2.erode(img_temp, kernel, iterations = 1)
        
        contours, hierarchy = cv2.findContours(img_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            c = max(contours, key = cv2.contourArea)
            angle = cv2.minAreaRect(c)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = angle * -1
        
        rotate = self.rotate_bound(img, angle)        
        return rotate, angle
    
    def remove_vertical_horizontal_lines(self, img):
        """Given a binary image remove vertical and horizontal lines which are not texts
        
        Args:
            img (numpy array): binary array
            
        Returns:
            [numpy array]:  binary array
        """
        k_thes = int(img.shape[1] * 0.10)
        
        # filter horizontal lines
        kernel = np.ones((1, k_thes), np.uint8)
        img_hor = cv2.erode(img, kernel, iterations = 1)
        img_hor = cv2.dilate(img_hor, kernel, iterations = 1)

        # filter vertical lines
        kernel = np.ones((k_thes, 1), np.uint8)
        img_ver = cv2.erode(img, kernel, iterations = 1)
        img_ver = cv2.dilate(img_ver, kernel, iterations = 1)
        
        img_lines = cv2.add( img_ver, img_hor)
        
        no_lines = cv2.absdiff(img, img_lines)
        
        return no_lines

    def remove_sections(self, img, roi):
        """
        """
        
        temp = np.array(img.copy()).astype('uint8')
        
        for rect in roi:
            cv2.rectangle(temp, (rect[0], rect[1]), (rect[2], rect[3]), (255,255,255), -1)
        
        return temp

    def get_tables(self, img, k_thes = 50, kn = 5):
        """Given a BGR array image return list of dicts with a representation
        of a table and list of cells
        Args:
            image (numpy array): BGR numpy array of pdf page
            k_thes (int): thes for select lines
            kn (int): kernel size for cleaning img
        Returns:
            [list]: list of dict items
        """
        
        tables = {}
        table_id = 0
        
        #convert to gray scale
        gray = self.get_grayscale(img)
        
        # filter horizontal lines
        kernel = np.ones((1, k_thes), np.uint8)
        img_hor = cv2.erode(gray, kernel, iterations = 1)
        img_hor = cv2.dilate(img_hor, kernel, iterations = 1)
        
        # filter vertical lines
        kernel = np.ones((k_thes, 1), np.uint8)
        img_ver = cv2.erode(gray, kernel, iterations = 1)
        img_ver = cv2.dilate(img_ver, kernel, iterations = 1)
        
        img_lines = cv2.add( img_ver, img_hor)
        img_lines = np.array(img_lines).astype('uint8')
        
        thresh = cv2.threshold(img_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # clean image
        kernel = np.ones((kn,kn),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        
        if len(boxes) < 1:
            return tables
            
        max_size = int(gray.shape[0] * 0.55)
        #Remove small cnt and change x,y,w,h --> x1,y1,x2,y2
        for box, order in zip(boxes, hier[0]):
            if box[-1] > k_thes and box[-2] > k_thes and box[-1] < max_size:
                # check if is a parent
                if order[-1] == -1:
                    table_id += 1
                    tables[table_id] = {"table":[int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])],"cells":[]}
                elif table_id in tables.keys():
                    tables[table_id]['cells'].append([int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])])
        
        # filter result for more than one cell table x square
        filtered = []
        for table in tables:
            if len(tables[table]['cells']) > 1:
                filtered.append(tables[table])
        
        return filtered
        
        
        
    def filter_doc_contour(self, img, kn = 2, kd = 3):
        """Filter contours from pdf image

        Args:
            img (numpy array): BGR numpy array of pdf page
            ko (int, optional): Size of kernel used in noisy remover process. Defaults to 3.
            kd (int, optional): Size of kernel for dilation. Defaults to 3.

        Returns:
            [numpy array]: processed binary image
        """
        gray = self.get_grayscale(img)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # correct skew
        #thresh, angle = self.fix_skew_img(thresh)

        # remove boxes
        thresh = self.remove_vertical_horizontal_lines(thresh)
        
        
        # clean image
        kernel = np.ones((kn,kn),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations = 1)

        return cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def get_bounding_boxes(self, img):
        """Given a filtered img find boudning boxes from contours

        Args:
            img (numpy array): binary image

        Returns:
            boundRect(list): list of bounding boxes
        """ 
        contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # get bounding box
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
        return boundRect
    
    def get_thes_x_y(self, boundRect, qa_x_min = 0.60, qa_x_max = 0.90, qa_y_min = 0.60, qa_y_max = 0.90):
        """With list of bounding boxes calculate threshold for x axis and y axis

        Args:
            boundRect (list): list of bounding boxes
            qa_x_min (float, optional): Quantile for min x axis values. Defaults to 0.60.
            qa_x_max (float, optional): Quantile for max x axis values. Defaults to 0.90.
            
            qa_y_min (float, optional): Quantile for y min axis values. Defaults to 0.60.
            qa_y_max (float, optional): Quantile for x max axis values. Defaults to 0.90.
        Returns:
            [float, float], [float, float]: threshold min max values for x and y in px
        """
        w = []
        h = []
        for bb in boundRect:
            w.append(bb[-2])
            h.append(bb[-1])
        max_x = np.mean(w) + 3 * np.std(w)
        max_y = np.mean(h) + 3 * np.std(h)
        return [int(np.quantile(w, qa_x_min)), int(np.quantile(w, qa_x_max))], [int(np.quantile(h, qa_y_min)), int(np.quantile(h, qa_y_max))], max_x, max_y

    def is_bb_intersect(self, boxA, boxB):
        """Check if two bounding boxes intersect

        Args:
            boxA (list): List with x,y points of left bottom and right top points -> Ax, Ay, Bx, By
            boxB (list): List with x,y points of left bottom and right top points -> Ax, Ay, Bx, By

        Returns:
            bool: True intersect, False no intersection.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return False
        return True
    
    def get_block_texts(self, bounding_boxes, th_x, th_y):
        """Given bounding boxes list agregate boxes usings th_x, thy_y as params
        when no more intersection occurs stop loop

        Args:
            bounding_boxes (list): list of bounding boxes
            th_x (int): threshold for x axis px.
            th_y (int): threshold for y axispx.

        Returns:
            [list]: list of text boxes
        """
        stop = False
        filter_block = False
        tested_blocks = bounding_boxes.copy()
        while not filter_block:
            if stop:
                # Remove outlier vertical blocks
                filter_block = True
                stop = False
                filtered_blocks = []
                for block in block_group:
                    if (block[0]>1860) and (block[3]-block[1] > 400):
                        pass
                    elif (block[2]<210) and (block[3]-block[1] > 400):
                        pass
                    else:
                        filtered_blocks.append(block)
                tested_blocks = filtered_blocks.copy()
            while(not stop):
                block_group = []
                tested_blocks = sorted(tested_blocks , key=lambda k: [k[0], k[1]])
                stop = True
                while len(tested_blocks) > 0:
                    # Set bb group
                    bb = tested_blocks.pop(0)
                    block_temp = bb.copy()
                    removed = []
                    thx = np.abs(block_temp[0] - block_temp[2])
                    if thx > th_x[1]:
                        thx = th_x[1]
                    elif thx < th_x[0]:
                        thx = th_x[0]
                    if filter_block:
                        thx = thx * 1.75

                    thy = np.abs(block_temp[1] - block_temp[3])
                    if thy > th_y[1]:
                        thy = th_y[1]
                    elif thy < th_y[0]:
                         thy = th_y[0]
                    
                    # Search intersection
                    for block in tested_blocks:
                        if self.is_bb_intersect([block_temp[0] - thx, block_temp[1] - thy, block_temp[2] + thx , block_temp[3] + thy], block):
                            stop = False
                            removed.append(block)
                            block_temp[0] = min(block_temp[0], block[0])
                            block_temp[1] = min(block_temp[1], block[1])
                            block_temp[2] = max(block_temp[2], block[2])
                            block_temp[3] = max(block_temp[3], block[3])
                    block_group.append(block_temp)
                    # Remove found boxes
                    for block in removed:
                        tested_blocks.remove(block)
                tested_blocks = block_group.copy()      
        return block_group

    def aggreate_text_blocks(self, block_group, thes_dist):
        """Join text blocks which are close enough by thes_dist
        Args:
            block_group (list): list of bounding box.
            thes_dist (int): threshold to check two blocks are close enough
        Returns:
             [list] : list of block texts.
        """
        temp = block_group.copy()
        running = True
        max_tries = len(block_group)
        
        if len(temp) <= 1:
            return temp
        
        while(running):
            x = 0
            temp = sorted(temp , key = lambda k: [k[1]])
            dist = []
            for key_group in temp:
                dist.append([])
                for group in temp:
                    dist_list = []
                    dist_list.append(np.linalg.norm(np.array([key_group[2], key_group[1]]) - np.array([group[0], group[1]])))
                    dist_list.append(np.linalg.norm(np.array([key_group[2], key_group[1]]) - np.array([group[0], group[3]])))
                    dist_list.append(np.linalg.norm(np.array([key_group[2], key_group[3]]) - np.array([group[0], group[1]])))
                    dist_list.append(np.linalg.norm(np.array([key_group[2], key_group[3]]) - np.array([group[0], group[3]])))
                    dist[x].append(min(dist_list))
                dist[x][x] = 0
                indexes = [ n for n,i in enumerate(dist[x]) if i < thes_dist and i > 0 ]
                if len(indexes) == 0:
                    running = False
                    x += 1
                else:
                    running = True
                    indexes.sort(reverse=True)
                    for ind in indexes:
                        if ind != x:
                            temp, new_ind = self.join_boxes(temp,x,ind)
                            x = new_ind
                    break
                max_tries = max_tries - 1
                if max_tries < 1:
                    break
        return temp
    
    def join_boxes(self, bb, a, b):
        """Given a list of bounding boxes join two
        Args:
            bb (list): list of bounding box.
            a (int): index of box
            b (int): index of box
        Returns:
             [list] : list of bounding boxes with len equal bb - 1
        """
        temp = bb.copy()
        xa = min(bb[a][0], bb[b][0])
        ya = min(bb[a][1], bb[b][1])
        xb = max(bb[a][2], bb[b][2])
        yb = max(bb[a][3], bb[b][3])

        if a > b:
            temp.remove(bb[a])
            temp.remove(bb[b])
        else:
            temp.remove(bb[b])
            temp.remove(bb[a])
        temp.append([xa, ya, xb, yb])
        
        return temp, len(temp)-1
        
    def get_contour(self, img, min_size = 5):
        """Given a BGR array image return list of blocks texts, this method
        is a wrapper for others functions in this class.
        Args:
            image (numpy array): BGR numpy array of pdf page
            min_size (int): min size in px to consider a box
        Returns:
            [list]: list of text boxes
        """
        # Rotate bounds
        #image, angle = self.get_skew_angle(img)
        image = img

        # Geta tables
        roi = []
        tables = self.get_tables(image)

        if self.roi_tables:
            # Remove img sections
            for table in tables:
                roi.append(table['table'])
        
        if len(roi) > 0:
            sect_img = self.remove_sections(image, roi)
        else:
            sect_img = image
        
        # Get contours
        filtered_img = self.filter_doc_contour(sect_img)
        bb_list =  self.get_bounding_boxes(filtered_img)
        
        if len(bb_list) == 0:
            return []

        th_x, th_y, max_x, max_y = self.get_thes_x_y(bb_list)

        # Format bb[xa,ya,w,h] to (xa,ya) (xb,yb) and filter outliers boxes
        bot_top_list = []
        for item in bb_list:
            if item[2] > min_size and item[3] > min_size:
                bot_top_list.append([item[0], item[1], item[0] + item[2], item[1] + item[3]])
            #if item[2] <= max_x and item[3] <= max_y:
             #   bot_top_list.append([item[0], item[1], item[0] + item[2], item[1] + item[3]])
        # Get block tests
        block_group = self.get_block_texts(bot_top_list, th_x, th_y)
        
        thes_dist = int(th_x[1])
        # agregate blocks close in x axis.

        blocks =  self.aggreate_text_blocks(block_group, thes_dist)
            
       
        ## Agregate result
        result = []
        for table in tables:
            result.append({'type':'table', 'bb':table['table'], 'cells':table['cells']})
        for block in blocks:
            result.append({'type':'text', 'bb':block})
            
        return result