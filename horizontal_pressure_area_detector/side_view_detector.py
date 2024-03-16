"""Class to detect horizontal pressure areas of 1 - 3 story homes."""

import dataclasses
import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore")


@dataclasses.dataclass(kw_only=True, frozen=True)
class areas:
    """Class to hold area's of each horizontal pressure point by floor.
    
      Note that in each dictionary, an index "0" equates to the roof, and is in
      order from top down. For example, if there was a 3 story building, index 0
      would be the roof, index 1 would be the upper floor, index 2 would be the
      main floor, and index 3 would be the bottom floor.

      It is possible to not have an index for each floor in every dictionary
      (in fact, this is usually the case).

    Attributes:
      a_area: A dictionary mapping a floors index to the sum of "a" horizontal
        pressure areas.
      b_area: A dictionary mapping a floors index to the sum of "b" horizontal
        pressure areas.
      c_area: A dictionary mapping a floors index to the sum of "c" horizontal
        pressure areas.
      d_area: A dictionary mapping a floors index to the sum of "d" horizontal
        pressure areas.
    """
    a_area: dict[int, float]
    b_area: dict[int, float]
    c_area: dict[int, float]
    d_area: dict[int, float]
    
    def __str__(self):
      """Converts object to debug string."""
      return f"""
        A areas: {self.a_area}
        B areas: {self.b_area}
        C areas: {self.c_area}
        D areas: {self.d_area}
        """


def detect_interest_region(image: cv2.typing.MatLike, image_len: float, image_ht: float):
    """_summary_

    Args:
        image(Array of uint8): Elevation image
        image_len (float): Length of image(in feet) 
        image_ht (float): Height of image(in feet)

    Returns:
       image_(Array of uint8): Interest region of elevation image
       image_len_i(float): Length of the interest region image(feet) 
       image_ht_i(float): Height of the interest region image(feet)
       [x1, y1](list) : The coordinate of left corner of the interest region
    """
  
    width = image.shape[1]
    height = image.shape[0]

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to segment the image
    threshold = cv2.adaptiveThreshold(
      gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
      threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on aspect ratio and area
    min_aspect_ratio = 0.5
    max_aspect_ratio = 30
    min_area = 100
    max_area = width*height * 1
    area_list = []
    text_regions = []
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      aspect_ratio = w / float(h)
      area = (w * h)
      if (aspect_ratio > min_aspect_ratio and area > min_area and
          aspect_ratio < max_aspect_ratio and area < max_area):
          text_regions.append((x, y, x + w, y + h))
          area_list.append(area)

    # Detect the main region
    main_region = text_regions[np.argmax(area_list)]
    main_region = list(main_region)

    # Detect right text region
    right_text = []
    right_x = []
    for text_region in text_regions:
        x1, y1, x2, y2 = text_region
        if x1 > 0.75 * width and y1 < 0.5 * height:
            right_text.append(text_region)
            right_x.append(x1)

    if len(right_x) > 2:
        main_region[2] = np.min(right_x)

    # Detect left text region
    left_text = []
    left_x = []
    for text_region in text_regions:
        x1, y1, x2, y2 = text_region
        if x1 < 0.25 * width and y1 < 0.5 * height:
            left_text.append(text_region)
            left_x.append(x1)

        if len(left_x) > 2:
           main_region[0] = np.max(left_x)

    # Get the interst region
    image_1 = np.ones(image.shape, dtype=np.uint8)*255
    x1, y1, x2, y2 = main_region
    image_1[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :]

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to find the edges of the building
    edges = cv2.Canny(gray, 50, 150)

    # Apply line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                          threshold = 20, minLineLength = 20, maxLineGap = 10)

    # Detect first vertical line and last vertical line
    vertical_x = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if np.abs(x1 - x2) < 3:
            vertical_x.append(x1)
    
    # Get the interest region
    if len(lines) < 200:
        main_region[0] = np.min(vertical_x) - 27
        main_region[1] = main_region[1] - 5  
        main_region[2] = np.max(vertical_x) + 27
    else:
      try:
          main_region[0] = np.min(vertical_x) - 7 
      except:
          pass
      main_region[1] = main_region[1] - 5  
      
      try:
          main_region[2] = np.max(vertical_x) + 7 
      except:
          pass
    main_region[3] = main_region[3] + 5  
  
    # Get the left top coordinate and right bottom coordinate of main region
    x1, y1, x2, y2 = main_region
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(image.shape[0], y2)
    x2 = min(image.shape[1], x2)
  
    #Calculate the length and height of interested region
    image_len_i = image_len*(x2-x1)/image.shape[1]
    image_ht_i = image_ht*(y2-y1)/image.shape[0]
  
    #Get the interest image
    image_ = image[y1:y2, x1:x2, :].copy()
    if image_.size == 0:
        return image_1, image_len, image_ht, [0, 0]
    return image_, image_len_i, image_ht_i, [x1, y1]

def get_line_map(lines, image, line_format = 'slant', direction = 'left'):
    
    if line_format == 'slant':
        #Line map of slant lines
        slant_line_map = np.zeros((image.shape[0], image.shape[1]))
      
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            #If line is slant, it is added in line map
            if direction == 'left':
                if abs(x1 - x2) > 30 and abs(y1 - y2) > 30 and y2 < y1: 
                  
                    for xx in range(x1, x2+1):
                       yy = y1 + (y2 -y1) * (xx-x1) / (x2-x1)
                       slant_line_map[int(yy), int(xx)] = 1  
            else:
                if abs(x1 - x2) > 30 and abs(y1 - y2) > 30 and y2 > y1: 
                  
                    for xx in range(x1, x2+1):
                       yy = y1 + (y2 -y1) * (xx-x1) / (x2-x1)
                       slant_line_map[int(yy), int(xx)] = 1 
        return slant_line_map
    
    if line_format == 'horizontal':
        horizontal_line_map = np.zeros((image.shape[0], image.shape[1]))
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 4:  #If line is horizontal, it is added in line map
                
                horizontal_line_map[y1, x1:x2+1] = 1 
        return horizontal_line_map
    
    if line_format == 'vertical':
        vertical_line_map = np.zeros((image.shape[0], image.shape[1]))
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 4:  #If line is horizontal, it is added in line map
                
                vertical_line_map[y1:y2+1, x1] = 1 
        return vertical_line_map   


def detect_left_wall(lines, plate_top1, plate_top2, plate_top, plate_bottom, 
                     image, wall = np.array([]), th=0.7,
                     depth = 0.2, gap = 10, roof = False, out_roof = False):
    """_summary_

    Args:
        lines (Array of int32): The coordinates of the detected lines in
                                      elevation image, Size(n, 1, 4)
        plate_top (Array of int32): The coordinates of the plate top,
                                      Size(1, 4)
        plate_bottom(Array of int32): The coordinates of the plate bottom, 
                                      Size(1, 4)
        plate_top1 (Array of int32): The coordinates of the upper plate top 
                                      of floor which wall is detected at, Size(1, 4)
        plate_top2 (Array of int32): The coordinates of the lower plate top 
                                     of floor which wall is detected at, Size(1, 4)
        image(Array of int32): Elevation image
        th(float): The threshold value to be used in detection of left wall line,
                    percent of detection length
        depth (float): The size of the search region of left wall
        gap(int): Tolence gap value to be used in detection left wall
        roof(bool): Whether or not to detect the side wall line of the roof
        

    Returns:
        left_wall(Array of int32): Array of the coordinates of lest wall lines
    """
  
    #Line map of horizontal lines
    horizontal_line_map = get_line_map(lines, image, line_format = 'horizontal')
    
    #Vertical lines
    n = np.where(abs(lines[:, 0, 0] - lines[:, 0, 2]) < 4)[0]
    vertical_lines = lines[n]
    lower_floor = False
    left_wall_detection = False
    bottom = plate_top2
    top = plate_top1
    if (plate_top2 == plate_bottom).all():
        lower_floor = True
  
    #Detect wall if first detection is failed  
    for k in range(2):
        if left_wall_detection:
            break
        left_vertical_lines = []
        left_vertical_len = []
        left_vertical_x = []
        left_wall = np.array([])      
      
        #Detect candidate lines of left wall
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            #Add candidate lines to out_roof list
            if out_roof:
               
                if (np.abs(x1 - x2) < 5 and x1 < image.shape[1]*depth and 
                    ((y1 > top[0][1] + 50 and y1 < bottom[0][1] ) and 
                    (y2 > top[0][1] + 50 and y2 < bottom[0][1] + 35))):
                      
                    
                    #Check the vertical line is dimension notation
                    len_horizon = len(np.where(horizontal_line_map[y1:y2, 
                                      x1 - 20:x1 - 10] == 1)[0])
                    if len_horizon > 10:
                       continue    
                    
                    left_vertical_lines.append(line)
                    for line in lines:
                      x11, y11, x22, y22 = line[0]
                      if (y22 > y2 and (y11 - y2) < gap and 
                         np.abs(x11 - x22) < 5 and np.abs(x1 - x22) < 5 
                         and y11 > top[0][1]):
                          
                          y2 = y22
                          left_vertical_lines[-1][0][3] = y2
            
                    left_vertical_x.append(x1)
                    left_vertical_len.append(np.abs(y2-y1))
                    
            #Add candidate lines to wall line list            
            else:
                #If line is vertical line and line is betwen top and between,
                #then line is added to candidate wall line list
                if (np.abs(x1 - x2) < 5 and x1 < image.shape[1]*depth and 
                    ((y1 > top[0][1] - 45 and y1 < bottom[0][1] + 45 and
                     y2> top[0][1] - 45 and y2 < bottom[0][1] + 45) or 
                    (y1 < top[0][1] + 45  and y2 > bottom[0][1] - 45 ))):
                  
                  left_vertical_lines.append(line)
                  #Search lines with gap
                  for line in lines:
                    x11, y11, x22, y22 = line[0]
                    if y11 > y22:
                        yy = y11
                        y11 = y22
                        y22 = yy
                        
                    # If the horizontal distance between two lines is less 
                    # than 5 and the coordinate of the second line's bottom y
                    # is between top and bottom, 
                    # then the second line is linked to the first 
                    if (y22 > y2 and (y11 - y2) < gap and np.abs(x11 - x22) < 3 and
                        np.abs(x1 - x22) < 3 and y22 < bottom[0][1] + 10):
                      y2 = y22
                      left_vertical_lines[-1][0][3] = y2
                  left_vertical_x.append(x1)
                  left_vertical_len.append(np.abs(y2-y1))                  
    
        #Detect left wall line
        for i in range(len(left_vertical_x)):
              n = np.argmin(left_vertical_x)
              x1, y1, x2, y2 = left_vertical_lines[n][0]
              if left_vertical_len[n] > th * np.abs(bottom[0][1] - top[0][1]):
                left_wall = left_vertical_lines[n]
                
                # Detect cylinder
                if lower_floor :
                    len_horizon = len(np.where(horizontal_line_map[
                        int(top[0][1] + 0.25*(bottom[0][1] - top[0][1])):
                        int(top[0][1] + 0.5*(bottom[0][1] - top[0][1])), 
                        x1:x1 + 40] == 1)[0])
                    if len_horizon > 10:
                        left_wall_detection = True
                        break
                    # The number of the vertical lines between -30 and 30 pixcel
                    n_vertical = np.where((abs(vertical_lines[:, 0, 0] - x1)< 25) &
                        (vertical_lines[:, 0, 1] < 
                        int(top[0][1] + 0.45*(bottom[0][1] - top[0][1]))) &
                        (vertical_lines[:, 0, 3] > 
                        int(top[0][1] + 0.25*(bottom[0][1] - top[0][1]))))
                    if len(n_vertical[0]) < 3 or len(n_vertical[0]) > 5:
                        left_wall_detection = True
                        break    
                    
                    # If the wall's position is same with the above wall
                    if wall.size > 0:
                        if abs(wall[-1][0] - x1) <6:
                            left_wall_detection = True
                            break                        
     
                else:  
                    left_wall_detection = True
                    break
              left_vertical_x[n] = image.shape[1]
        th = 0.75 * th
    return left_wall


def detect_roof_left(lines, lines2,  roof_top, plate_top, image, th = 0.7,
                     depth = 0.2, gap = 10, roof = False, plate_detect = False):
    """_summary_

    Args:
        lines(Array of int32): The coordinates of the detected lines 
                               in elevation image, Size(n, 1, 4)
        roof_top(Array of int32): The coordinates of the roof top, Size(1, 4)
        plate_top(Array of int32): The coordinates of the plate top, Size(1, 4)
        image(Array of int32): elevation image
        th(float): The threshold value to be used in detection of left wall line, 
                   percent of detection length
        depth(float): The size of the search region of left wall
        gap(int): Tolence gap value to be used in 
                       of detection left wall
        roof(bool): Whether or not to detect the sidewall line of the roof
        

    Returns:
        roof_left(Array of int32): Array of the coordinates of lest roof lines
    """
  
    # Line map of slant 
    slant_line_map = get_line_map(lines2, image, line_format = 'slant')
    threshold_slant_point = 0.05*len(np.where(slant_line_map == 1)[0])
    
    # Line map of horizontal lines
    horizontal_line_map = get_line_map(lines2, image, line_format = 'horizontal')
 
    # Revise y coordinate of roof top so that it is horizontal line
    if roof_top[0][1]<roof_top[0][3]:
        roof_top[0][3] = roof_top[0][1]
    else:
        roof_top[0][1] = roof_top[0][3]
    
    if lines is None:
        return []
    left_vertical_lines = []
    left_vertical_len = []
    left_vertical_x = []
    roof_left = np.array([])
    roof_left1 = np.array([])
  
    # Detect candidate lines of left wall
    for line in lines:
    
        x1, y1, x2, y2 = line[0]
        if y1 >y2 :
            yy = y2
            y2 = y1
            y1 = yy
    
        # Add vertical line the candidate line list
        if (np.abs(x1 - x2) < 5 and x1 < image.shape[1] * depth and 
            y1 > roof_top[0][1] - 20 and
            y1 < plate_top[0][1] - 20 and y2 < plate_top[0][1] + 50):
         
          # Check slant lines in front of line
          len_slant = len(np.where(slant_line_map[y1:int(y1 + 0.8 * (y2 - y1)),
                                                  :x1] == 1)[0])
          if len_slant > threshold_slant_point:
                continue
          if not plate_detect:        
              # Check the vertical line is dimension notation
              y1_ = max(0, y1 - 10)
              len_horizon = len(np.where(horizontal_line_map[y1_:
                                 int(y1 + 0.5 * (y2 - y1)), x1-20:x1 -2] == 1)[0])
              if len_horizon > 10 and abs(y1-y2) < 40:
                  continue
          left_vertical_lines.append(line)     
    
          # Process gap and add vertical line the candidate line list
          for line in lines:
            x11, y11, x22, y22 = line[0]
            if y11 >y22:
                yy = y22
                y22 = y11
                y11 = yy
                line[0][1] = y1
                line[0][3] = y2
                
            # Linked vertical line with gap    
            if (y22 > y2 and (y11 - y2) < gap and np.abs(x11 - x22) < 5 and
                np.abs(x1 - x22) < 4 and y11 > roof_top[0][1]):
              y2 = y22
              left_vertical_lines[-1][0][3] = y2
          ####    
          left_vertical_x.append(x1)
          left_vertical_len.append(np.abs(y2-y1))

    # Detect left roof wall line
    for i in range(len(left_vertical_x)):
        n = np.argmin(left_vertical_x)
        if left_vertical_len[n] > th * np.abs(roof_top[0][1] - plate_top[0][1]):
            roof_left = left_vertical_lines[n]
            roof_left[0][3] = roof_left[0][3] - 5
            break
        left_vertical_x[n] = image.shape[1]
        
    if plate_detect:
        if len(roof_left) == 0:
            try:
              roof_left = left_vertical_lines[np.argmax(left_vertical_len)]
            except:
              roof_left = []
    
    # If line is not in range, it is set as []
    if len(roof_left) > 0:
        if min(roof_left[0][1],  roof_left[0][3])- plate_top[0][1] > 50:
            roof_left = []   
            
    # If left roof is at above 75% of image 
    if len(roof_left) > 0:
        if roof_left[0][3] > 0.75*image.shape[0]:
            roof_left1 = roof_left.copy()
            roof_left = []         
 
    if len(roof_left) ==0 and not plate_detect:
        try:
            # Detect horizontal lines between roof top and roof top and plate top
            nn = np.where((lines[:, 0, 1] > roof_top[0][1] - 5) & 
                          (lines[:, 0, 1] < plate_top[0][1] + 5) &
                          (lines[:, 0, 3] > roof_top[0][3] - 5) &
                          (lines[:, 0, 3] < plate_top[0][3] + 5) &
                          (lines[:, 0, 2] >50) & (lines[:, 0, 0] <50) &
                          (abs(lines[:, 0, 1] - lines[:, 0, 3]) < 5))
            roof_left_c = lines[nn]
            
            # Find line that y coordinate is biggest
            if len(roof_left_c)>0:
                nn1 = np.where(roof_left_c[:, 0, 1] == max(roof_left_c[:, 0, 1]))[0][0]
                x2 = roof_left_c[nn1, 0, 0] 
                y2 = roof_left_c[nn1, 0, 1]
                x1 = x2
                y1 = roof_top[0][1]
        
                # Vertical roof left 
                roof_left = np.array([[x1, y1, x2, y2]])
              
            # Detect left wall by slant lines    
            roof_left2 = []
            for k in range(2):
                for x in range(1, int(depth*image.shape[1])):
                    y_r = roof_top[0][1]
                    y_p = plate_top[0][1]
                    
                    # The number of slant roof
                    n = np.where(slant_line_map[y_r:y_p + 1, x] == 1)[0]
                    if n.size > 0:
                        if n[-1] - n[0] > 0:
                            if n.size == 1:
                                roof_left2 = [[x, y_r + n[0],  x, y_r + n[0] + 10]]
                            else:
                                roof_left2 = [[x, y_r + n[0],  x, y_r + n[-1]]]
                            break
                if len(roof_left2) > 0:
                  break
            if roof_left2:
                  roof_left = roof_left2
        except:
            roof_left = roof_left1
       
    elif len(roof_left) ==0 and plate_detect:
        roof_left = roof_left1
    
    # Revise roof_left
    if not plate_detect:
        if roof_left[0][0] > plate_top[0][0] + 30:
            roof_left[0][0] = plate_top[0][0]
            roof_left[0][2] = plate_top[0][0]
   
    if plate_detect:
        # Detect slant roof lines in range 5% depth
        nn = 0
        for x in range(1, int(depth*image.shape[1])):
            y_r = roof_top[0][1]
            y_p = plate_top[0][1]
            
            # The number of slant roof
            n = np.where(slant_line_map[y_r:y_p + 1, x] == 1)[0]
            
            # If there is slant roof, then plate line is revised
            if len(n) > 1:
                y_s = n[-1]  # y coordinate of slant roof
                
                if roof_left[0][1] + 30 > y_s and 120 > x:
                    if n[1] - n[0] >10:
                        roof_left[0][3] = y_s + y_r 
                    else:
                        # Check the detected roof is up or blow
                        nn = [max(n)]
                        for k in range(10):
                            n_ = np.where(slant_line_map[y_r:y_p + 1, x + k] == 1)[0]
                            nn.append(max(n_))
                        if max(n) +2  >= max(nn):
                      
                              roof_left[0][3] = y_s + y_r + 15 
                        else:
                              roof_left[0][3] = y_s + y_r  
                    if roof_left[0][1] > roof_left[0][3]:
                        roof_left[0][1] = roof_left[0][3]
                break
    return roof_left

    
def detect_right_wall(lines, plate_top1, plate_top2, plate_top, plate_bottom,
                      image, wall = np.array([]), th=0.7,
                      depth=0.8, gap=10, roof=False, out_roof = False):
    """_summary_

    Args:
        lines (Array of int32): The coordinates of the detected lines 
                                      in elevation image, Size(n, 1, 4)
        plate_top (Array of int32): The coordinates of the plate top, Size(1, 4)
        plate_bottom(Array of int32): The coordinates of the plate bottom, Size(1, 4)
        
        plate_top1 (Array of int32): The coordinates of the upper plate top of floor
                                     which wall is detected at, Size(1, 4)
        plate_top2 (Array of int32): The coordinates of the lower plate top 
                                     of floor which wall is detected at, Size(1, 4)
        image(Array of int32): Elevation image
        th(float): The threshold value to be used in 
                   detection of right wall line
        depth(float): The size of the search region of left wall
        gap(int): Tolence gap value to be used in detection right wall
        roof(bool): Whether or not to detect the sidewall line of the roof
        

    Returns:
        right_wall(Array of int32): Array of the coordinates of right wall lines
    """
  
    # Line map of horizontal lines
    horizontal_line_map = get_line_map(lines, image, line_format = 'horizontal')
    
    # Vertical lines
    n = np.where(abs(lines[:, 0, 0] - lines[:, 0, 2]) < 4)[0]
    vertical_lines = lines[n]
    lower_floor = False
    right_wall_detection = False
    top = plate_top1.copy()
    bottom = plate_top2.copy()    
  
    # Incase the floor is lower flower 
    if (plate_top2 == plate_bottom).all():
        lower_floor = True
    
    # Iterate the right wall detection twice, it is needed    
    for k in range(2):
        
        if right_wall_detection:
            break
        right_vertical_lines = []
        right_vertical_len = []
        right_vertical_x = []
        right_wall = np.array([])
      
        # Detect candidate lines of right wall
        if lines is not None:
            for line in lines:
              x1, y1, x2, y2 = line[0]
             
              # Add candidate lines to the out roof 
              # Candidate lines must be vertical line and be in 50 % region from top
              if out_roof:
                  if (np.abs(x1 - x2) < 5 and x1 > image.shape[1]*depth and 
                      (y1 > top[0][1] + 50  and y1 < bottom[0][1] )  and 
                      (y2 > top[0][1] + 50  and y2 < bottom[0][1] + 20)):
                          
                    # Check the vertical line is dimension notation
                    len_horizon = len(np.where(
                        horizontal_line_map[y1:y2, x2 + 10:x2 + 20] == 1)[0])
                    if len_horizon > 10:
                       continue   
                    right_vertical_lines.append(line)
            
                    # Link lines with gap
                    # The horizantal distance of two lines must be smaller than 5
                    for line in lines:
                      x11, y11, x22, y22 = line[0]
                      if (y22 > y2 and (y11 - y2) < gap 
                          and np.abs(x11 - x22) < 5 and
                          np.abs(x1 - x22) < 5 and y11 > top[0][1]):
                      
                        y2 = y22
                        right_vertical_lines[-1][0][3] = y2
                    right_vertical_len.append(np.abs(y2-y1))
                    right_vertical_x.append(x2)
              else:
                  
                  # If line is vertical line and line is betwen top and between,
                  # then line is added to candidate wall line list       
                  if (np.abs(x1 - x2) < 5 and x1 > image.shape[1]*depth and 
                        ((y1 > top[0][1] - 35 and y1 < bottom[0][1] + 35 and
                        y2> top[0][1] - 35 and y2 < bottom[0][1] + 35) or 
                        (y1 < top[0][1] + 35  and y2 > bottom[0][1] - 35 ) or
                        (y1 > top[0][1] - 35 and 
                         y1 < (bottom[0][1] + top[0][1])/2 + 35))):               
                      right_vertical_lines.append(line)
                      
                      # Search lines with gap
                      for line in lines:
                        x11, y11, x22, y22 = line[0]
                        
                        # If the horizontal distance between 
                        # two lines is less than 5 and
                        # the coordinate of the second line's bottom y 
                        # is between top and bottom
                        # then the second line is linked to the first 
                        if (y22 > y2 and (y11 - y2) < gap and 
                            np.abs(x11 - x22) < 5 and 
                            np.abs(x1 - x22) < 5 and 
                            y22 < bottom[0][1] + 20):
                            y2 = y22
                            right_vertical_lines[-1][0][3] = y2
                      right_vertical_x.append(x1)
                      right_vertical_len.append(np.abs(y2-y1))

        if len(wall) > 0 and right_wall.size > 0:
            # If the detected wall is behind the previous wall 
            # when the detected wall is the lower wall,
            # it is set as empty
            if wall[-1][0] < right_wall[0][0]:
                right_wall = np.array([])
            
        # Detect the right wall line
        for i in range(len(right_vertical_x)):
            n = np.argmax(right_vertical_x)
            x1, y1, x2, y2 = right_vertical_lines[n][0]
            if right_vertical_len[n] > th * np.abs(top[0][1] - bottom[0][1]):
                right_wall = right_vertical_lines[n]
                
                # Detect cylinder
                if lower_floor :
                    len_horizon = len(np.where(horizontal_line_map[
                        int(top[0][1] + 0.25*(bottom[0][1] - top[0][1])):
                        int(top[0][1] + 0.5*(y2 - top[0][1])), 
                        x1 - 40:x1] == 1)[0])
                    if len_horizon > 10:
                        right_wall_detection = True
                        break
                    # The number of the vertical lines between -30 and 30 pixcel
                    n_vertical = np.where((abs(vertical_lines[:, 0, 0] - x1)< 25) &
                        (vertical_lines[:, 0, 1] < 
                        int(top[0][1] + 0.45*(bottom[0][1] - top[0][1]))) &
                        (vertical_lines[:, 0, 3] > 
                        int(top[0][1] + 0.25*(bottom[0][1] - top[0][1]))))
                    
                    if len(n_vertical[0]) < 3 or len(n_vertical[0]) > 5:
                        right_wall_detection = True
                        break    
                    # If the wall's position is same with the above wall
                    if wall.size > 0:
                        if abs(wall[-1][0] - x1) <6:
                            right_wall_detection = True
                            break                        
                else:  
                    right_wall_detection = True
                    break            
            right_vertical_x[n] = 0
        if out_roof:
            break
        th = th * 0.85
    return right_wall


def detect_roof_right(lines, lines2, roof_top, plate_top, image, th = 0.7, 
                      depth = 0.8, gap = 10, roof = True, plate_detect = False):
    """_summary_
    
    Args:
          lines (Array of int32): The coordinates of the detected lines 
                                  in elevation image, Size(n, 1, 4)
          roof_top (Array of int32): The coordinates of the roof top, Size(1, 4)
          plate_top(Array of int32): The coordinates of the plate_top, Size(1, 4)
          image(Array of int32): Elevation image
          th (float): The threshold value to be used in detection of 
                      sright wall line
          depth (float): The size of the search region of left wall
          gap(int): Tolence gap value to be used in detection right wall
          roof(bool): Whether or not to detect the sidewall line of the roof
          
    
    Returns:
          roof_right(Array of int32): List of the coordinates of right roof lines
    """
    # Line map of slant lines
    slant_line_map = get_line_map(
        lines2, image, line_format = 'slant', direction = 'right')
    threshold_slant_point = 0.05 * len(np.where(slant_line_map == 1)[0])
    
    #Line map of horizontal lines
    horizontal_line_map = get_line_map(lines, image, line_format = 'horizontal')
   
    #Revise y coordinate of roof top so that it is horizontal line
    if roof_top[0][1]<roof_top[0][3]:
        roof_top[0][3] = roof_top[0][1]
    else:
        roof_top[0][1] = roof_top[0][3]
      
    if lines is None:
      return []

    right_vertical_lines = []
    right_vertical_len = []
    right_vertical_x = []
    roof_right = []
    roof_right1 = []
    
    # Detect candidate lines of right wall
    if lines is not None:
      for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 >y2:
            yy = y2
            y2 = y1
            y1 = yy  
            line[0][1] = y1
            line[0][3] = y2
        
        # Add vertical line to candidate line list
        if (np.abs(x1 - x2) < 5 and x2 > image.shape[1] * depth and
            y1 > roof_top[0][1] -20 and
            y1 < plate_top[0][1] - 20 and y2 < plate_top[0][1] + 50):
          
          # Check slant lines in front of line
          len_slant = len(np.where(slant_line_map[y1: int(y1 + 0.8*(y2 -y1)), 
                                                  x1:image.shape[1] -1] == 1)[0])
          if len_slant > threshold_slant_point:
              continue
          
          if not plate_detect:        
              # Check the vertical line is dimension notation
              y1_ = max(0, y1 - 10)
              len_horizon = len(np.where(horizontal_line_map[y1_:
                                int(y1 + 0.25 * (y2 - y1)), x1 - 20:x1] == 1)[0])
              if len_horizon < 10:
                  continue
          
          right_vertical_lines.append(line) 
          for line in lines:
              x11, y11, x22, y22 = line[0]
           
              # Exchange it if top point's y is greater than bottom point's
              if y11 > y22:
                  yy = y22
                  y22 = y11
                  y11 = yy
                  
              # Linked vertical line with gap
              if (y22 > y2 and (y11 - y2) < gap and
                  np.abs(x11 - x22) < 5 and 
                  np.abs(x1 - x22) < 4 and 
                  y11 > roof_top[0][1]):
                  y2 = y22
                  right_vertical_lines[-1][0][3] = y2
          right_vertical_len.append(np.abs(y2-y1))
          right_vertical_x.append(x2)
    
    # Detect the right wall line
    for i in range(len(right_vertical_x)):
        n = np.argmax(right_vertical_x)
        if right_vertical_len[n] > th * np.abs(roof_top[0][1] - plate_top[0][1]):
            roof_right = right_vertical_lines[n] - 0
            break
        right_vertical_x[n] = 0
    
    if plate_detect:  
        try:
            if len(roof_right) == 0:
                roof_right = right_vertical_lines[np.argmax(right_vertical_len)]
        except:
            roof_right =[]
   
    # If line is not in range, it is set as []
    if len(roof_right) > 0:
        if min(roof_right[0][1], roof_right[0][3]) - plate_top[0][1] > 50:
            roof_right = []    
          
    # If left roof is at above 70% of image height
    if len(roof_right) > 0:
        if roof_right[0][3] > 0.75 * image.shape[0]:
              roof_right1 = roof_right
              roof_right = [] 
    
    if len(roof_right) == 0 and not plate_detect:
        try:
            # Detect horizontal lines between roof top and roof top and plate top
            nn = np.where((lines[:, 0, 1] > roof_top[0][1] - 5) &
                          (lines[:, 0, 1] < plate_top[0][1] + 5) &
                          (lines[:, 0, 3] > roof_top[0][3] - 5) &
                          (lines[:, 0, 3] < plate_top[0][3] + 5) &
                          (lines[:, 0, 2] > 50) &
                          lines[:, 0, 2] + 5 < image.shape[1] &
                              (abs(lines[:, 0, 1] - lines[:, 0, 3]) < 5))[0]
            roof_left_c = lines[nn]
            
            # Find line that y coordinate is biggest
            if len(roof_left_c) > 0:
                nn1 = np.where(roof_left_c[:,0,1] == 
                               max(roof_left_c[:, 0, 1]))[0][0]
                x2 = roof_left_c[nn1, 0, 0]
                y2 = roof_left_c[nn1, 0, 1]
                x1 = x2
                y1 = roof_top[0][1]
                
                # Vertical roof right 
                roof_right = np.array([[x1, y1, x2, y2]])
                
            # There are only slant lines    
            elif not plate_detect:
                for x in range(1, int((1-depth)*image.shape[1])):
                    x = image.shape[1] - x
                    y_r = roof_top[0][1]
                    y_p = plate_top[0][1]
                    
                    # The number of slant roof
                    n = np.where(slant_line_map[y_r:y_p + 20, x] == 1)[0]
                    if len(n) > 0:
                        if len(n) == 1:
                            roof_right = [[x, y_r + n[0],  x, y_r + n[0] + 10]]
                        else:
                            roof_right = [[x, y_r + n[0],  x, y_r + n[-1]]]
                        break
        except:
            roof_right = roof_right1      
    elif len(roof_right) ==0 and  plate_detect:      
        roof_right = roof_right1
        
        # Detect candidate points of roof right 
        # Find lines what y coordinates are between roof top and plate top
        y_roof = min(roof_top[0][1], roof_top[0][3])
        nn = np.where((lines[:, 0, 3] > y_roof - 5) &
                      (lines[:, 0, 3] < plate_top[0][3]+5) &
                      (lines[:, 0, 1] > y_roof - 5) & 
                      (lines[:, 0, 1] < plate_top[0][1]+5) &
                      (lines[:, 0, 2] + 5 < image.shape[1]))[0] 
        roof_right_c = lines[nn]
        
        # Detect x and y coordinate of  first point
        # Find point which have the maximum x coordinate
        nn1 = np.where(roof_right_c[:, 0, 2] == max(roof_right_c[:, 0, 2]))[0][0]
        y1 = roof_right_c[nn1, 0, 3]
        x1 = roof_right_c[nn1, 0, 2]
        
        # Detect x and y coordinate of second point
        # Find point which have the second maximum x coordinate
        nn2 = np.where(abs(roof_right_c[:, 0, 3] - y1)>5)
        roof_right_c2 = roof_right_c[nn2]
        
        nn21 = np.where(roof_right_c2[:, 0, 2] == 
                        max(roof_right_c2[:, 0, 2]))[0][0]
        y2 = roof_right_c2[nn21, 0, 3]
        x2 = roof_right_c2[nn21, 0, 2]
        
        # Right wall is vertical line
        x = max(x1, x2)
        roof_right = np.array([[x, y1, x, y2]])
        
    if plate_detect:
          # Detect slant roof lines
          for x in range(1, int((1 - depth)*image.shape[1])):
              x =  image.shape[1] -x            
              y_r = roof_top[0][1]
              y_p = plate_top[0][1]
              
              # The number of slant roof
              n = np.where(slant_line_map[y_r:y_p + 1, x] == 1)[0]
              # If there is slant roof, then plate line is revised
              if len(n) > 1:
                y_s = n[-1]  # y coordinate of slant roof
                if roof_right[0][1] + 30 > y_s and 120 > image.shape[1] - x:                  
                    if n[-1] - n[0] >10:
                        roof_right[0][3] = y_s + y_r 
                    else:
                        # Check the detected roof is up or blow
                        nn = [max(n)]
                        for k in range(10):
                            if x + k > image.shape[1]-1:
                                break
                            n_ = np.where(slant_line_map[y_r:y_p + 1, x + k] == 1)[0]
                            try:
                                nn.append(max(n_))
                            except:
                                continue
                        if max(n) +2  >= max(nn):
                           roof_right[0][3] = y_s + y_r + 15 
                        else:
                           roof_right[0][3] = y_s + y_r  
                    if roof_right[0][1] > roof_right[0][3]:
                        roof_right[0][1] = roof_right[0][3]                    
                    break
    return roof_right


def detect_bottom(lines, gray):
    """_summary_

    Args:
        lines (Array of int32): Coordinates of the detected lines, Size (n, 1 ,4)
        gray: gray elevation image


    Returns:
        bottom(array of int32): The coordinate of bottom line
    """    

    bottom_lines = []
    bottom_len = []
    bottom_y = []
    bottom = []
    bottom_lines90 = []
    bottom_len90 = []
    
    # Detect the candidate lines of bottom
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if np.abs(y1 - y2) < 5 and y1 > gray.shape[0]*0.65:
              bottom_lines.append(line)
              bottom_len.append(np.abs(x2-x1))
            if np.abs(y1 - y2) < 5 and y1 > gray.shape[0]*0.87:
              bottom_lines90.append(line)
              bottom_len90.append(np.abs(x2-x1))  
              bottom_y.append(y1)
    bottom_y1 = bottom_y
    bottom_y2 = bottom_y

    # Find the longest line in below region
    if len(bottom_lines90)>0:
        bottom = bottom_lines90[np.argmax(bottom_len90)]
 
        if abs(bottom[0][0] - bottom[0][2]) < gray.shape[1] * 0.7:
            bottom = bottom_lines[np.argmax(bottom_len)]
    else:
        bottom = bottom_lines[np.argmax(bottom_len)]

    # Revise the bottom line if it is in 10% of image bottom
    if bottom[0][1]>0.9*gray.shape[0] :
        for i in range(len(bottom_y1)):
            n = np.argmin(abs(bottom_y1 - bottom[0][1]))
            # 8 is threshold for spacing of two neighbor lines
            if (bottom_len90[n] > 0.4*gray.shape[1] and 
                abs(bottom_y1[n] - bottom[0][1]) > 8 and
                abs(bottom_y1[n] - bottom[0][1]) <38):
                bottom = bottom_lines90[n]
                break
            bottom_y1[n] = 0  
            
    # Detect horizontal lines
    horizontal_lines = lines[np.where(lines[:, 0, 1] - lines[:, 0, 3] < 3)]

    # Revise the bottom line 
    if bottom[0][1] > 0.9 * gray.shape[0]:
        for i in range(len(bottom_y2)):
            n = np.argmin(abs(bottom_y2 - bottom[0][1]))
            
            # 3 is threshold for spacing of two neighbor lines
            if (bottom_len90[n] > 0.6 * gray.shape[1] 
                and (bottom[0][1] - bottom_y2[n]) >3
                and ( bottom[0][1] - bottom_y2[n]) <20):
                bottom_candidate = bottom_lines90[n]
                               
                # Detect horizontal lines within 10 pixel with candidate bottom
                n_lines1 = np.where(abs(bottom_candidate[0][1] - 
                                        horizontal_lines[:, 0, 1]) < 10) 
                horizontal_lines2 = horizontal_lines[n_lines1]
                n_lines = np.where(abs(bottom_candidate[0][1] - 
                                       horizontal_lines2[:, 0, 1]) > 2)  
                candidate_horizontal_lines = horizontal_lines2[n_lines]
                
                # Calculate gray level between bottom candidate line 
                # and candidate horizontal line
                for line in candidate_horizontal_lines:
                    x1 =  max(bottom_candidate[0][0], line[0][0]) 
                    y1 =  min(bottom_candidate[0][1], line[0][1])
                    x2 = min(bottom_candidate[0][2], line[0][2]) 
                    y2 = max(bottom_candidate[0][1], line[0][1])
                    gray_level = np.mean(gray[y1:y2+1, x1:x2+1])
                    if gray_level < 100:
                        bottom = line
                        break
                break
            bottom_y2[n] = 0      
    return bottom


def detect_roof_top(lines, image):
    """_summary_
    
    Args:
          lines (Array of int32): Coordinates of the detected lines, Size (n, 1 ,4)
          image: elevation image
    
    
    Returns:
          roof_top(array of int32): The coordinate of roof top line
    """     
    # Line map of vertical lines
    vertical_line_map = get_line_map(lines, image, line_format = 'vertical')
    
    # Get y coordinates of description lines and candidate roof top
    y_des = []
    c_lines = []
    for line in lines:
          x1, y1, x2, y2 = line[0]
    
          # Find candidate roof top line in possible region
          if y1 > 0.4 * image.shape[0] or y2 > 0.4 * image.shape[1]:
              continue
          # Detect descriotion line,
          # If line is linked at left or right of image,
          # it is description line
          if (abs(x2 - image.shape[1]) < 4 or x1 < 4 or
              abs(x1 - image.shape[1]) < 4 or x2 <4): #description line 
              
              # If description line is crossed with vertical line,
              # it is the candidate roof line 
              for xx in range(int(x1 + 0.2 * (x2 - x1)), x2):
                  nn = np.where(vertical_line_map[y1:y1 + 20, xx] == 1)[0]
                  if len(nn) > 18:
                      line[0][2] = xx
                      c_lines.append(line)
                          
                  # If description line is not crossed with vertical line,    
                  # it is added description list
                  if xx == x2 - 1:    
                      y_des.append(y1)
          # Find the candidate roof top, vertical line is not roof top line    
          elif abs(x1- x2) > image.shape[1] * 0.1 :
              c_lines.append(line)
    
    c_lines = np.array(c_lines)
    # Detect the line roof top
    # Find line which y coordinate of first point is smallest
    n1 = np.where(c_lines[:, 0, 1] == min(c_lines[:, 0, 1]))[0][0]
    
    # Find line which y coordinate of second point is smallest
    n2 = np.where(c_lines[:, 0, 3] == min(c_lines[:, 0, 3]))[0][0]
    
    # Calculate min y values of two candidate top line
    line1 = c_lines[n1]
    line2 = c_lines[n2]
    lin1_y = min(line1[0][1], line1[0][3])
    lin2_y = min(line2[0][1], line2[0][3])
    
    # Of the two lines, the one with smaller y is the top line of the root.
    if lin1_y < lin2_y:
        roof_top = line1
    else:
        roof_top = line1
    return roof_top


def detect_plate_top(lines, lines2,  roof_top, bottom, 
                     image, ht1 = 0.25, ht2 = 0.4):
    """_summary_
    
    Args:
          lines (Array of int32): Coordinates of the detected lines, 
                                  Size (n, 1 ,4)
          lines2 (Array of int32): Coordinates of the detected lines, 
                                   Size (n, 1 ,4), it is used in detection of
                                   slant line
          image: Elevation image
          roof_top(array of int32): The coordinate of roof top line
          ht1(float): The top thresold value of plate height  
          ht2(float): The bottom thresold value of plate height    
    Returns:
          plate_top(array of int32): The coordinate of plate top line
    """  
   
    # Limit of plate top
    plate_top_limit = roof_top.copy()
    plate_top_limit[0][1] = int(image.shape[0] * 0.5) 
    plate_top_limit[0][3] = int(image.shape[0] * 0.5)
    
    # Detect the left of roof
    roof_left = detect_roof_left(
      lines.copy(), lines2.copy(), roof_top.copy(), plate_top_limit.copy(), 
      image, th = 0.3, gap = 10, roof = True, plate_detect = True)
    
    # Detect the right of roof
    roof_right = detect_roof_right(
      lines.copy(), lines2.copy(), roof_top.copy(), plate_top_limit.copy(), 
      image, th = 0.3, gap = 10, roof = True, plate_detect = True)  
    
    # Detect top plate point
    y11 = 0
    y12 = 0
    y21 = 0
    y22 = 0
    
    # y coordinates of the detected roof left line
    if roof_left is not None:
        y11 =  roof_left[0][1]
        y12 =  roof_left[0][3]
    try:    
        if roof_right is not None:
            y21 =  roof_right[0][1]
            y22 =  roof_right[0][3]
    except:
        pass
    
    y = min(max(y11, y12), max(y21, y22))
    
    # Exchange y coordinates so that left point's y coordinate is greater
    # than right point's y coordinate of roof top line
    if roof_top[0][1]<roof_top[0][3]:
        roof_top[0][3] = roof_top[0][1]
    else:
        roof_top[0][1] = roof_top[0][3]
    
    # Get y coordinates of description lines
    y_des = []
    for line in lines:
          x1, y1, x2, y2 = line[0]
          # If line is linked at left or right of image, it is description line
          if abs(x2 - image.shape[1]) < 4 or x1 < 4: #description line 
              y_des.append(y1)
              
    y_des = np.array(y_des)  
    
    # Candidate plate top lines list to be used in max legth search
    plate_top_lines_m = []
    plate_top_len_m = []
    
    # Candidate plate top lines list to be used in revising plate top line
    plate_top_lines = []
    plate_top_len = []
    plate_top_y = []
    plate_top = []
    
    # Detect of the candidate lines of plate top 
    if lines is not None:
      for line in lines:
        x1, y1, x2, y2 = line[0]
        if len(y_des) > 0:
            y_des_ = abs(y_des - y1)
            if y_des[np.argmin(y_des_)] < 10:
                continue
            
        # Add lines candidate lines list to be used in revising plate top line    
        if np.abs(y1 - y2) < 5 and y1 > y - 30 and y1 < y + 5:
          plate_top_lines.append(line)
          plate_top_len.append(np.abs(x2-x1))
          plate_top_y.append(y1)
        
        # Add lines in candidate line list to be used max length search   
        if np.abs(y1 - y2) < 5 and y1 > y - 20 and y1 < y + 5:
          plate_top_lines_m.append(line)
          plate_top_len_m.append(np.abs(x2 - x1))
    plate_top_y1 = plate_top_y.copy()  
    
    # Detect the line of first plate top
    if len(plate_top_len_m) > 0 :
        plate_top = plate_top_lines_m[np.argmax(plate_top_len_m)]
        yy = detect_horizontal_max_len_line(lines.copy(), image.copy(), 
                                            y - 12, y + 5)
        plate_top[0][1] = yy
        plate_top[0][3] = yy
    else:
        plate_top = [[int(0.2 * image.shape[1]), y - 13, 
                      int(0.8 * image.shape[1]), y - 13]]
       
    # Detect of  plate top in among candidate lines
    # plate_top_y is y coordinate of candidate plate top line
    min_diff = 3
    for k in range(2):
        if k == 1:
            plate_top_y = plate_top_y1.copy()
            min_diff = 2
        for i in range(len(plate_top_y)):
            n = np.argmin(abs(plate_top_y - plate_top[0][1]))
            if (plate_top_len[n] > 0.2*image.shape[1] and 
                abs(plate_top_y[n] - plate_top[0][1]) >min_diff and
                abs(plate_top_y[n] - plate_top[0][1]) < 5 and 
                plate_top_y[n] < plate_top[0][1]):
                
              plate_top = plate_top_lines[n]
              break
            plate_top_y[n] = 0
     
    return  plate_top 
  
    
def detect_horizontal_max_len_line(lines, image, detect_y1, detect_y2):
    """_summary_
    
    Args:
          lines (Array of int32): The coordinates of the detected lines 
                                  in elevation image, Size(n, 1, 4)
          image(Array of int32): Elevation image
          detect_y1(float): The start y coordinate to be detected
          detect_y2(float): The end y coordinate to be detected
    
    Returns:
          y(float): y coordinate of line that have maximum length in 
             the given range
    """   
    
    # Line map of horizontal lines
    line_map = np.zeros((image.shape[0], image.shape[1]))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        #If line is horizontal, it is added in line map
        if abs(x1 - x2) > 30 and abs(y1 - y2) < 3: 
            line_map[y1, x1 : x2+1] = 1
            
    # Revise with gap of 5 in horizontal direction and 5 in vertical direction
    for yy in range(20, image.shape[0] - 20):
      xx = np.where(line_map[yy, 20:image.shape[1] - 20] == 1)[0]
      line_map[yy-3:yy + 4, xx] = 1
      
    # Calculate  the length of lines between detect_y1 and detect_y2
    line_len = []
    for y_ in range(detect_y1, detect_y2):
      max_length = 0
      current_length = 0
      for element in line_map[y_, :]:
          if element == 1:
              current_length += 1
              max_length = max(max_length, current_length)
          else:
              current_length = 0
      line_len.append(max_length)
      
    # Detect y coordinate of line 
    n = np.argmax(line_len)
    y = detect_y1 + n
    return y


def detect_plate_top23(lines,  roof_top, plate_top, bottom, image, 
                       ht1 = 0.4, ht2 = 0.7, floor = 2, number_of_floor = 2):
    """_summary_
    
    Args:
          lines (Array of int32): Coordinates of the detected lines, Size (n, 1 ,4)
             
          image: Elevation image
          roof_top(array of int32): The coordinate of roof top line
          plate_top(array of int32): The coordinate of the upper plate top line
          bottom(array of int32): The coordinate of bottom line
          ht1(float): The top thresold value of plate height  
          ht2(float): The bottom thresold value of plate height  
          floor: 
          numOfFloor: The number of floors in the building
    Returns:
          plate_top2(array of int32): The coordinate of plate top line
    """  
   
    if roof_top[0][1]<roof_top[0][3]:
        roof_top[0][3] = roof_top[0][1]
    else:
        roof_top[0][1] = roof_top[0][3]
    
    # Determine plate_top23 line by the number of floor
    if floor ==2:
          yy = (plate_top[0][1] + (floor - 1) *
               (bottom[0][1] - plate_top[0][1])/number_of_floor)
          yy = int(yy)
          plate_top23 = [[bottom[0][0], yy, bottom[0][2], yy]]
          
    elif floor ==3:
          yy = (plate_top[0][1] + (floor - 1)*
               (bottom[0][1] - plate_top[0][1])/number_of_floor)
          yy = int(yy)
          
          # Find y coordinate of max length line  between yy - 10 and yy + 10
          yy = detect_horizontal_max_len_line(
                   lines.copy(), image.copy(), yy - 10, yy + 10)
          plate_top23 = [[bottom[0][0], yy, bottom[0][2], yy]]
    return plate_top23


def detect_roof_rest(lines, roof_top, roof_left, roof_right, 
                     plate_top, image, roof_rest_, nn):
    """_summary_
    
    Args:
        lines(Array of int32): The detectected lines in images, Size: (n,1,4)
        roof_top(Array of int32): The coordinates of roof top line, Size: (1,4)
        roof_left(Array of int32): The coordinates of roof left line, Size:  (1,4)
        roof_right(Array of int32): The coordninates of roof right line, Size: (1,4)
        plate_top(Array of int32): The coordinates of roof top line, Size: (1,4)
        image(Array of uint8): elevation image
        roof_rest_(list): The list of roof top lines except for main roof top line
        nn(int): the number of function calling
    
    Returns:
        roof_rest_(list): The list of roof top lines except for main roof top line
    """
    if roof_top[0][2] - roof_top[0][0] > 0.97*(roof_right[0][0] - roof_left[0][0]):
      return roof_rest_
    
    roof_rest_lines = []     # The candidate lines of roof rest
    roof_rest_lengths = []   # The list for length of the candidate lines of roof rest
    roof_rest_ys = []        # The list for the y coordinates of 
                             # he candidate lines of roof rest
    roof_rest = roof_rest_.copy() # Previous roof rest
   
    # Detect of the candidate lines of roof top lines 
    if lines is not None:
      for line in lines:
        x1, y1, x2, y2 = line[0]
       
        # If there is space between roof top and roof left, 
        # then the rest of roof top is searched.  
        if abs(roof_top[0][0] - roof_left[0][0]) > 20:
          if (np.abs(y1 - y2) < 5 and y1 > roof_top[0][1] and 
              y1 < plate_top[0][1] and x1 < roof_top[0][0] - 10 and
              x2 < roof_top[0][0] + 10):
            roof_rest_lines.append(line)
            roof_rest_lengths.append(np.abs(x2-x1))
            roof_rest_ys.append(y1)
      
        # If there is space between roof top and roof right, then the rest of 
        # roof top is searched.  
        if abs(roof_top[0][2] - roof_right[0][2]) > 20:
            if (np.abs(y1 - y2) < 5 and y1 > roof_top[0][1] and 
                y1 < plate_top[0][1] and
                x2 > roof_top[0][2] + 10 and x1 > roof_top[0][2] - 10):
              roof_rest_lines.append(line)
              roof_rest_lengths.append(np.abs(x2-x1))
              roof_rest_ys.append(y1)
   
    # Detect the roof top lines 
    for i in range(len(roof_rest_ys)):
        n = np.argmin(roof_rest_ys)
        if abs(roof_top[0][0] - roof_left[0][0]) > 20:
            if roof_rest_lengths[n] > 0.25*(abs(roof_top[0][0] - roof_left[0][0])):
                roof_rest.append(roof_rest_lines[n])
                break
        if abs(roof_top[0][2] - roof_right[0][2]) > 20:
            if roof_rest_lengths[n] > 0.25*(abs(roof_top[0][2] - roof_right[0][2])):
                roof_rest.append(roof_rest_lines[n])
                break
        roof_rest_ys[n] = image.shape[0]
        
    if len(roof_rest) == len(roof_rest_):
        if len(roof_rest_lines)>0:
            roof_rest.append(roof_rest_lines[np.argmax(roof_rest_lengths)])
    try:
        # Reset roof left and right x coordinates to continue to detect roof top lines
        if abs(roof_top[0][2] - roof_right[0][2]) > 20:
            if roof_rest[-1][0][2] > roof_top[0][2]:
                roof_top[0][2] = roof_rest[-1][0][2]
                
        if abs(roof_top[0][0] - roof_left[0][0]) > 20:
            if roof_rest[-1][0][0] < roof_top[0][0]:
                roof_top[0][0] = roof_rest[-1][0][0]
        nn = nn + 1
    except:
         return roof_rest

    # If detection depth is greater than 5, return the obtained roof top lines 
    if nn > 5:
        return roof_rest
    
    # If detection depth is smaller than 5, continue to detect the roof top ines
    else:
        roof_rest = detect_roof_rest(
            lines, roof_top, roof_left, roof_right, 
            plate_top, image, roof_rest, nn)
    return roof_rest


def get_area(roof_points, x1, x2, y):
    """_summary_
    
    Args:
        roof_points (Array of int32) : coordinates of points of roof top
        x1 (float): x1 coordinate of left point that the area is calculated
        x2 (float): x2 coordinate of right point that the area is calculated
        y (float): y coordinate of right point that the area is calculated
    
    Returns:
        area(float) : the area
        points(Array of int32) : the coordinates of top line points to be calculated
    """
    # Check the coordinates of roof points
    roof_points_new = np.array([roof_points[0]])
    
    for k in range(1, len(roof_points)):
        if roof_points_new[-1, 0] < roof_points[k,0]:
            roof_points_new = np.concatenate((roof_points_new, 
                                  roof_points[k].reshape(1,-1)))
        else:
            roof_points_new = np.concatenate((roof_points_new,
                              np.array([roof_points_new[-1, 0], 
                              roof_points[k, 1]]).reshape(1, -1)))
    
    roof_points = roof_points_new  
    n1 = np.where(roof_points[:, 0] <= x1)[0][-1]
    y1 = (roof_points[n1, 1] + (roof_points[n1+1, 1] - 
                               roof_points[n1, 1])*(x1 - roof_points[n1, 0])/
         (roof_points[n1+1, 0] - roof_points[n1, 0]))
    n2 = np.where(roof_points[:, 0] <= x2)[0][-1]
    try:
      y2 = (roof_points[n2, 1] + (roof_points[n2+1, 1] -
                                  roof_points[n2, 1])*(x2 - roof_points[n2, 0])/
            (roof_points[n2+1, 0] - roof_points[n2, 0]))
    except:
      y2 = roof_points[n2, 1]
    
    points = []
    points.append([x1, y1])
    for k in range(n1+1, n2+1):
      points.append([roof_points[k][0], roof_points[k][1]])
    
    points.append([x2, y2])
    points = np.array(points)
    area = 0
    for k in range(len(points)-1):
      area = area + (points[k+1][0] - points[k][0]) * \
        (y - (points[k+1][1] + points[k][1])/2)
    return area, points


def get_area_BD(roof_points, roof_top, roof_left, roof_right,
                plate_top, lines, gray, x_st, x_end, y, B=False):
    """_summary_
    
    Args:
        roof_points (Array of int32): The coordinates of roof points, Size(n,2)
        roof_top(Array of int32): The coordinates of roof top, Size(1,4)
        roof_left(Array of int32): The coordinates of roof left, Size(1,4)
        roof_right(Array of int32): The coordinates of roof right, Size(1,4)
        plate_top(Array of int32): The coordinates of plate top, Size(1,4)
        
        lines(Array of int32): The coordinates of the detected line, Size(n,1,4)
        x_st (float): x1 coordinate of left point that the area is calculated
        x_end (float): x2 coordinate of right point that the area is calculated
        y (float): y coordinate of top plate that the area is calculated
    
    Returns:
        roof_D_area(float) : The area of D
        roof_C_area(float) : The area of C included in D 
        points(Array of int32): The coordinates of upper line points of roof D, Size(n,2): 
    """
    # Slant line map
    slant_line_map = np.zeros(gray.shape).astype(np.uint8)
    
    points = [] # Coordinates of roof top
    for line in lines:
          x1, y1, x2, y2 = line[0]
          if abs(y1 - y2) <70 or abs(x1 - x2) < 70:
              continue
          for xx in range(x1, x2+1):
              yy = y1 + (y2 -y1) * (xx - x1) / (x2 - x1)
              slant_line_map[int(yy), int(xx)] = 1
    
    # Horizontal line map
    horizontal_line_map = np.zeros(gray.shape).astype(np.uint8)
     
    for line in lines:
         x1, y1, x2, y2 = line[0]
         if abs(y1- y2) < 5:
             horizontal_line_map[y1, x1:x2 + 1] =1
    
    # Coordinates of D area points
    D_area_xy = []
    
    # If roof top line is not horizontal line
    if abs(roof_top[0][1] - roof_top[0][3]) > 5:
        
        # y coordinate of upper point of roof top line
        y_top = min(roof_top[0][1], roof_top[0][3]) 
        area = (plate_top[0][1] - y_top)*(x_end - x_st) #total area in roof region      
              
        # Calcaulate D and C area to be included in roof region
        D_area = 0
        C_area = 0
        point_record = 0
        slant_line_start = 0 # The variable to display that the slant line starts
        region_index = 0 # If it is 1, then D area, if it is 2 then C area
        
        # Caculate the D area and C area to be inclued in D
        for xx in range(int(x_st), int(x_end)):
          point_record = 0
          slant_line_start = 0
          region_index = 0
          
          for yy in range(y_top, int(y)):
              if horizontal_line_map[yy, xx] ==1 and slant_line_start ==0:
                  region_index = 1 
                  if point_record == 0:
                      points.append([xx, yy])
                      point_record =1
              
              if slant_line_map[yy, xx] ==1:
                  region_index = 2
                  slant_line_start = 1
                  if point_record == 0:
                      points.append([xx, yy])
                      point_record =1
          
              if region_index == 1:
                  D_area = D_area + 1
                  D_area_xy.append([xx, yy])
              if region_index == 2:
                 C_area = C_area + 1 
                 
        if D_area < 100:
            D_area = 0
        if B:
            C_area = 1.15*C_area   
            
        # Calculate center of D_area
        if D_area_xy:
            D_area_xy = np.array(D_area_xy)
            
            D_area_x = np.mean(D_area_xy[:, 0])
            D_area_y = np.mean(D_area_xy[:, 1])
            D_area_center =[D_area_x, D_area_y]
        else:
            D_area_center = [0, 0]
           
        return  D_area, C_area,  points, D_area_center    
        
    # If roof top line is horizontal    
    else:
        area, points = get_area(roof_points.copy(), x_st, x_end, y)
    
    # Calulate roof C area to be included in roof
    roof_D_area = 0
    for k in range(len(points) -1): # Points are coordinates of top roof in D region
          x1 , y1 = points[k]
          x2 , y2 = points[k + 1]
          
          # If top line roof is not horizontal, this is region of roof C
          if abs(y1 - y2)>5: 
              continue
          for x in range(int(x1), int(x2)):
              roof_D_area = roof_D_area  + 3
              for yy in range(int(y1 + 3), int(y)): #detect D area
                  # If the value of gray image is less than 230, 
                  # then it is line point
                  if slant_line_map[yy, x] == 1: 
                      break
                  roof_D_area  = roof_D_area + 1
    
    roof_C_area = area - roof_D_area
    if abs(roof_top[0][1] - roof_top[0][3]) > 5:
        roof_D_area = 0
    
    # Calculate the area of B
    if B :
        roof_C_area1 = roof_C_area
        roof_C_area = 0.55*roof_C_area1
        if roof_D_area > 10:
            if roof_C_area1 / roof_D_area > 0.25:
                roof_D_area = roof_D_area - 0.05 * roof_C_area1
            else:
                roof_D_area = roof_D_area + 0.15 * roof_C_area1
    # Calculate the area of D
    else:
        roof_C_area1 = roof_C_area
        roof_C_area = 0.8 * roof_C_area1 
        if roof_D_area > 10:
            roof_D_area = roof_D_area 
    return roof_D_area, roof_C_area,  points, [-1, -1] 
 
   
def get_area_roof2(plate_top2, left_wall, right_wall,  lines, image, x_1, x_2):
    """_summary_

    Args:
        plate_top2 (Array of int32): Line coordinates of plate top2
        left_wall (Array of int32) : Line coordinates of left wall
        right_wall (Array of int32): Line coordinates of right wall
        
        lines(Array of int32): Coordinates of lines to be detected in image,
                                Size(n,1,4)
        image(Array of uint8): Elevation image
        x_1(float): The start x coordinate of D region
        x_2(float): The end x coordinates of D region 

    Returns:
        B_area(float) : The area of B
        D_area(float) : The area of D
 
    """
    B_area = 0
    D_area = 0
    
    # Line map
    line_map = np.zeros((image.shape[0], image.shape[1]))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if abs(x2 - image.shape[1]) < 4 or x1 < 4: # Description line 
            continue
        if abs(y1-y2) < 5:
            line_map[y1, x1:x2] = 1
        elif abs(x1 - x2) < 5 and x_1 < x1 and x_1 < x2:
            line_map[y1:y2, x1 - 10:x1 + 10] = 2
        else:
            for x in range(x1, x2):
                y = y1 + (y2 - y1) * ((x - x1) / (x2-x1))
                line_map[int(y), int(x)] = 1
    
    # Calculate left B area
    B1 = 0 
    for x in range(left_wall[0], int(x_1)):
        B1 = B1 + 10
        for y in range(0, plate_top2[0][1] - 10):
            yy = plate_top2[0][1] - 10 - y
            if line_map[yy,x] == 1:
                break
        B1 = B1 + (plate_top2[0][1] - yy)
    
    if B1/(int(x_1)-left_wall[0]) < 30:
        B1 = 0
        
    # Calculate right B area    
    B2 = 0 
    for x in range(int(x_2), right_wall[0]):
        B2 = B2 
        for y in range(0, plate_top2[0][1] - 10):
            yy = plate_top2[0][1]-10 - y
            if line_map[yy,x] == 1:
                break
        B2 = B2 + (plate_top2[0][1] - yy)   
    if B2 / (right_wall[0] - int(x_2) ) < 30:
        B2 = 0  
        
    # Calculate D area    
    D_area = 0 
    for x in range(int(x_1), int(x_2)):
        D_area = D_area 
        for y in range(0, plate_top2[0][1] - 10):
            yy = plate_top2[0][1] - 10 - y
            if line_map[yy,x] == 1 or line_map[yy,x] == 2:
                break
        D_area = D_area + (plate_top2[0][1] - yy) 
        
    if D_area / (int(x_2) - int(x_1) ) < 30:
        D_area = 0
    B_area = B1    
    return B_area, D_area


def get_sum_square_distance(lines, plate_top, bottom, yy1, yy2, image):
    """_summary_
    
    Args:
          lines(Array of int32): The coordinates of the detected lines 
                                 in elevation image, Size(n, 1, 4)
          plate_top(Array of int32): The coordinates of the plate top, Size(1, 4)
          bottom(Array of int32): The coordinates of the bottom line, Size(1, 4)
          image(Array of int32): Elevation image
          yy1(float): The start y coordinate of range
          yy2(float): The end y coordinate of range
    
    Returns:
          sq_dist(float): Sum of the square distance in the given range
    """    
        
    # The sum of squared distances
    sq_dist = 0
    
    # Center of range
    range_cen = (yy1 + yy2) / 2
       
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x1 > 0.06 * image.shape[1] and x1 < 0.94 * image.shape[1]:
            
            # Find vertical lines between plate top and bottom and 
            # Calculate the sum of squared distances
            if (abs(x1 - x2) < 4 and  y1 > plate_top[0][1]  and
                y1 < bottom[0][1]  and 
                y2 > plate_top[0][1] - 10 and  y2 < bottom[0][1] + 10):
                    
                cc = (y1 + y2)/2    
                if yy1 <= cc and cc < yy2:
                    sq_dist = sq_dist + (cc -range_cen )**2
    return sq_dist
    

def get_number_of_floor(lines, roof_top, plate_top, bottom, image):
    """_summary_
    
    Args:
          lines(Array of int32): The coordinates of the detected lines 
                                  in elevation image, Size(n, 1, 4)
          roof_top(Array of int32): The coordinates of the roof top, Size(1, 4)
          plate_top(Array of int32): The coordinates of the plate_top, Size(1, 4)
          bottom(Array of int32): The coordinates of the bottom line, Size(1, 4)
          image(Array of int32): Elevation image
    
    Returns:
          number_of_floor(int): The number of floors of story building
    """    
    
    # Determine the number of floor
    ht1 = plate_top[0][1] - roof_top[0][1]
    ht2 = bottom[0][1] - plate_top[0][1]    
    
    if ht1/ht2 > 0.65:
        return 1
    elif ht1/ht2 > 0.39:
        return 2 
    elif ht1/ht2 < 0.28 :
        return 3
            
    # Determine the optimal number of clusters 
    ss_list = []
    for i in range(1, 4):
        
        # Calculate the sum of squared disctances of one floor
        if i == 1:
            yy1 =  plate_top[0][1] 
            yy2 =  bottom[0][1]
            
            # The sum of squared distances
            ss =  get_sum_square_distance(lines.copy(), plate_top.copy(),
                                          bottom.copy(), yy1, yy2, image)
            ss_list.append(ss)
            
        # Calculate the sum of squared disctances of two floors
        if i ==2:
            yy1 =  plate_top[0][1] 
            yy2 =  (plate_top[0][1] + bottom[0][1]) / 2
            
            # The sum of squared distances
            ss1 =  get_sum_square_distance(
                       lines.copy(), plate_top.copy(),
                       bottom.copy(), yy1, yy2, image)
            
            yy1 = (plate_top[0][1] + bottom[0][1]) / 2
            yy2 =  bottom[0][1] 
            
            # The sum of squared distances
            ss2 =  get_sum_square_distance(
                       lines.copy(), plate_top.copy(),
                       bottom.copy(), yy1, yy2 ,image)            
            ss_list.append(ss1 + ss2)
            
        # Calculate the sum of squared disctances of three floors    
        if i == 3:
            yy1 =  plate_top[0][1] 
            yy2 =  plate_top[0][1]  + (bottom[0][1] - plate_top[0][1]) / 3
            
            # The sum of squared distances
            ss1 =  get_sum_square_distance(
                       lines.copy(), plate_top.copy(),
                       bottom.copy(), yy1, yy2, image)
            yy1 = plate_top[0][1]  + (bottom[0][1]- plate_top[0][1]) / 3
            yy2 =  plate_top[0][1]  + 2 * (bottom[0][1]- plate_top[0][1]) / 3 
            
            # The sum of squared distances
            ss2 =  get_sum_square_distance(
                       lines.copy(), plate_top.copy(),
                       bottom.copy(), yy1, yy2, image)
            yy1 = plate_top[0][1]  + 2 * (bottom[0][1] - plate_top[0][1]) / 3
            yy2 =  bottom[0][1] 
            
            # The sum of squared distances
            ss3 =  get_sum_square_distance(lines.copy(), plate_top.copy(),
                                           bottom.copy(), yy1, yy2, image)
            ss_list.append(ss1 + ss2 + ss3)        
    
    if ss_list[1] / ss_list[2] > 2.5:
        return 3
    else:
        return 2


def draw(ratio, p0, p1, p2, B11, B12, B2, D, D2,  D_area_center, D_point, C_roof,
         roof_top, plate_top, plate_top2, plate_top3, bottom, left_bottom,
         right_bottom, left_out_roof, right_out_roof, number_of_floor,
         resized_image, original_image, output_path, image_path,
         additional_lines = False):
    """_summary_
    
    Args:
          ratio(float): The ratio of resized image to original image
          p0(Array of int32): The coordinate for start point of interest region
          p1(Array of int32): The coordinate of 20% point in the bottom
          p1(Array of int32): The coordinate of 80% point in the bottom
          B11(float): The area of left B in the roof
          B12(float): The area of right B in the roof
          B2(float): The area of B in the upper floor
          D(float): The area of D in the roof
          D2(float): The area of D in the upper floor 
          D_area_center(float): The center of D region in the roof
          D_point(Array of int32): The coordinate of points in the upper line of D
          C_roof(float): The area of C region to be included in the D
          bottom(Array of int32): The coordinate of bottom line
          left_bottom(Array of int32): The coordinate of the left bottom point
          right_bottom(Array of int32): The coordinate of the right bottom point
          left_out_roof(Array of int32): The coordinate of the left out roof
          rihgt_out_roof(Array of int32): The coordinate of the right out roof
          roof_top (Array of int32): The coordinates of the roof top, Size(1, 4)
          plate_top(Array of int32): The coordinates of the upper plate_top,
                                     Size(1, 4)
          plate_top2(Array of int32): The coordinates of the middle plate_top,
                                     Size(1, 4) 
          plate_top3(Array of int32): The coordinates of the below plate_top,
                                     Size(1, 4)
          number_of_floor(int): The number of floor of the story building                           
          resized_image(Array of int32): Resized elevation image
          original_image(Array of int32): Original elevation image
          output_path(String): Path of output
          image_path(String): Path of the input image
          
    
    Returns:
          Void: write drawing images 
    """    
    
    # Coordinates of vertical lines
    vert1 = [p1, [p1[0], 0]]
    vert2 = [p2, [p2[0], 0]]
    vert1 = np.array(vert1)
    vert2 = np.array(vert2)
    
    # Coordinates coordinate of plate tops
    plate_tops = plate_top
    if len(plate_top2) > 0:
        plate_tops = np.concatenate((plate_tops, plate_top2))
    if len(plate_top3) > 0:
        plate_tops = np.concatenate((plate_tops, plate_top3))
        
    plate_tops = np.concatenate((plate_tops, bottom))
    plate_tops[:, 0] = 0
    plate_tops[:, 2] = resized_image.shape[1]
    
    vertical_lines = []
    # Convert first vertical line's coordinate to original image's coordinate
    vert1 = (vert1/ratio).astype(int)
    vert1[:, 0] = vert1[:, 0] + p0[0]
    vert1[:, 1] = vert1[:, 1] + p0[1]
    vertical_lines.append([vert1[0, 0], vert1[0, 1], vert1[1, 0], vert1[1, 1]])
    
    # Convert second vertical line's coordinate to original image's coordinate
    vert2 = (vert2/ratio).astype(int)
    vert2[:, 0] = vert2[:, 0] + p0[0]
    vert2[:, 1] = vert2[:, 1] + p0[1]
    vertical_lines.append([vert2[0, 0], vert2[0, 1], vert2[1, 0], vert2[1, 1]])
    
    # Drawing the vertical lines and horizontal lines
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # Coordinates of horizontal lines
    horizon_lines = []
    for k in range(len(plate_tops) - 1):
          yy = (plate_tops[k][1] + plate_tops[k + 1][1]) / 2
          horizon_lines.append([0, yy, resized_image.shape[1], yy])
    horizon_lines = np.array(horizon_lines)      
    
    # Convert second horizontal line's coordinate to original image's coordinate
    horizon_lines = (horizon_lines/ratio).astype(int)
    horizon_lines[:, 0] = horizon_lines[:, 0] + p0[0]
    horizon_lines[:, 1] = horizon_lines[:, 1] + p0[1] 
    horizon_lines[:, 2] = horizon_lines[:, 2] + p0[0]
    horizon_lines[:, 3] = horizon_lines[:, 3] + p0[1]
    
    # Convert second horizontal line's coordinate to original image's coordinate
    plate_tops = (plate_tops/ratio).astype(int)
    plate_tops[:, 0] = plate_tops[:, 0] + p0[0]
    plate_tops[:, 1] = plate_tops[:, 1] + p0[1] 
    plate_tops[:, 2] = plate_tops[:, 2] + p0[0]
    plate_tops[:, 3] = plate_tops[:, 3] + p0[1]
    
    if additional_lines:
    
        # Drawing  top lines of the plates
        for k in range(len(plate_tops)):
            x1, y1, x2, y2 = plate_tops[k]
            cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    # Drawing the  horizontal lines
    for line in horizon_lines:
        x1, y1, x2, y2 = line
        cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    original_image = cv2.circle(
      original_image, (vert1[0, 0], vert1[0, 1]),
      radius=5, color=(0, 0, 255), thickness=-1)
    
    original_image = cv2.circle(
      original_image, (vert2[0, 0], vert2[0, 1]), 
      radius=5, color=(0, 0, 255), thickness=-1)
    
    ######## Write texts in roof area ##########
    # Write text in left B
    if B11 > 0:
        x = int((left_bottom[0] + p1[0]) * 0.5)
        y = int((roof_top[0][1] + plate_top[0][1]) / 2) 
        x = int(x/ratio) + p0[0]  - 5
        y = int(y/ratio) + p0[1] + 10
        
        cv2.putText(
            original_image, "B" , (x, y),
            cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)  
        
    # Write text in roof D
    if D > 0 and D_area_center[0] > 0:
        x = int(D_area_center[0]/ratio + p0[0]) # x coordinate of center for D area
        y = int(D_area_center[1]/ratio + p0[1] + 10) # y coordinate of center for D area
        cv2.putText(original_image, "D" , (x, y),
            cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2) 
    
    elif D > 0:
        
        if C_roof/D > 0.2:
            ct = 1 / original_image.shape[1] * 1000
            if C_roof > D * 0.8:
                x = int((p1[0])) + 30 * ct
                y = int((D_point[0][1])) + 20 * ct
                x = int(x/ratio) + p0[0] - 5
                y = int(y/ratio) + p0[1] + 10
                cv2.putText(
                    original_image, "D", (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2) 
    
            else:
                x = int(p1[0] + (p2[0] - p1[0]) * 0.1)
                y = int((roof_top[0][1] + (plate_top[0][1]  - roof_top[0][1]) * 0.2)) 
                x = int(x/ratio) + p0[0] - 5
                y = int(y/ratio) + p0[1] + 5
                cv2.putText(
                    original_image, "D" , (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)           
            
        else:
            x = int((p1[0] + p2[0]) * 0.5)
            y = int((roof_top[0][1] + plate_top[0][1])/2) 
            x = int(x/ratio) + p0[0] - 5
            y = int(y/ratio) + p0[1] + 10
            cv2.putText(
                original_image, "D", (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
       
    # write text in right B
    if B12 > 10:
        x = int((p2[0] + right_bottom[0]) * 0.5)
        y = int((roof_top[0][1] + plate_top[0][1]) / 2) 
        x = int(x/ratio) + p0[0] - 5
        y = int(y/ratio) + p0[1] + 10
        cv2.putText(
            original_image, "B", (x, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
   
    # write text in left upper roof B
    if number_of_floor > 1:
        if B2 > 0 :
            x = int((left_bottom[0] + p1[0]) * 0.5)
            y = plate_top2[0][1] - (bottom[0][1] - plate_top2[0][1]) * 0.1
            x = int(x/ratio) + p0[0] - 5
            y = int(y/ratio) + p0[1] + 10
            cv2.putText(
                original_image, "B", (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2) 
            
            # write text in right upper B
            x = int((p2[0] + right_bottom[0]) * 0.5)
            y = plate_top2[0][1] - (bottom[0][1] - plate_top2[0][1]) * 0.1
            x = int(x/ratio) + p0[0] - 5
            y = int(y/ratio) + p0[1] + 10
            cv2.putText(
                original_image, "B", (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)  
            
        if D2 > 0:
            # write text in upper D
            x = int((p1[0] + p2[0]) * 0.5)
            y = plate_top2[0][1] - (bottom[0][1] - plate_top2[0][1]) * 0.1
            x = int(x/ratio) + p0[0] - 5
            y = int(y/ratio) + p0[1] + 10
            cv2.putText(
                original_image, "D", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)      
    
    ####################
    # Write text in left roof A
    x = int((left_bottom[0] + p1[0]) * 0.5)
    x = int(x/ratio) + p0[0]- 5
    y = plate_tops[0,1] + (plate_tops[1, 1] - plate_tops[0, 1]) * 0.25
    y = int(y) + 10
    cv2.putText(
        original_image, "A", (x, y), 
        cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)
    
    for k in range(1, len(plate_tops) - 1):
        y = plate_tops[k, 1] 
        y = int(y) + 20
        cv2.putText(
            original_image, "A" , (x, y),
            cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)
        
    # write text in right roof A
    x = int((p2[0] + right_bottom[0]) * 0.5)
    x = int(x/ratio) + p0[0]- 5
    y = plate_tops[0,1] + (plate_tops[1, 1] -plate_tops[0, 1]) * 0.25
    y = int(y) + 10
    cv2.putText(
        original_image, "A", (x, y),
        cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)
     
    for k in range(1, len(plate_tops) - 1):
         y = plate_tops[k, 1] 
         y = int(y) + 20
         cv2.putText(original_image, "A", (x, y), 
                     cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2) 
         
    # write text in roof C
    x = int((p1[0] + p2[0]) * 0.5)
    x = int(x/ratio) + p0[0] - 5
    y = plate_tops[0,1] + (plate_tops[1,1] - plate_tops[0, 1]) * 0.25
    y = int(y) + 10
    cv2.putText(
        original_image, "C", (x, y), 
        cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)
     
    for k in range(1, len(plate_tops) - 1):
         y = plate_tops[k, 1] 
         y = int(y) + 20
         cv2.putText(
             original_image, "C", (x, y), 
             cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)      
    
    # write text in left upper B
    if len(left_out_roof) > 0:
        y = (left_out_roof[0][3] + left_out_roof[0][1])/2
        x = left_out_roof[0][0]
        x = int(x/ratio) + p0[0] + 10
        y = int(y/ratio) + p0[1] + 5
        cv2.putText(
            original_image, "B", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)       
     
    # write text in right upper B
    if len(right_out_roof)>0:
        y = (right_out_roof[0][3] + right_out_roof[0][1])/2
        x = right_out_roof[0][0]
        x = int(x/ratio) + p0[0] - 30
        y = int(y/ratio) + p0[1] + 5
        cv2.putText(
            original_image, "B", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
    
    # write the leading text of roof
    x = min( max(p0[0] - 100, 0), p0[0])
    y = plate_tops[0, 1]
    
    cv2.putText(
        original_image, "Load to" , (x, y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)
    
    cv2.putText(
        original_image, "roof" , (x, y + 10), 
        cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2) 
    
    cv2.putText(
        original_image, "plate:" , (x, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2) 
    
    if number_of_floor > 1:
        # Write the leading text of upper
        x = min(max(p0[0]- 100, 0), p0[0])
        y1 = int((horizon_lines[0, 1]  + horizon_lines[1, 1])/2)
        
        cv2.putText(
            original_image, "Load to", (x, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)
        
        cv2.putText(
            original_image, "upper", (x, y1 + 10), 
            cv2.FONT_HERSHEY_SIMPLEX ,0.7, (0, 0, 255), 2)
        
        cv2.putText(
            original_image, "plate:", (x, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX ,  0.7, (0, 0, 255), 2) 
        
    if number_of_floor == 3:
        # Write the leading text of upper
        x = min(max(p0[0]- 100, 0), p0[0])
        y1 = int((horizon_lines[1, 1]  + horizon_lines[2, 1])/2)
        
        cv2.putText(
            original_image, "Load to", (x, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0, 0, 255), 2)
        
        cv2.putText(
            original_image, "lower", (x, y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        
        cv2.putText(
            original_image, "plate:", (x, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)       
        
    # Save image file
    filename = output_path + "/" + image_path.split("/")[-1]
    cv2.imwrite(filename, original_image)   
    
def detect_left_right_out_roof(
        image, number_of_floor, lines, left_walls, 
        plate_top, plate_top2, bottom, right_walls):
    """_summary_
    
    Args:

          image(Array of int32): Elevation image
          image(Array of int32): Elevation image
          number_of_floor(int): The number of floor of the story buidling
          lines (Array of int32): The coordinates of the detected lines 
                                  in elevation image, Size(n, 1, 4)
          plate_top(Array of int32): The coordinates of the upper plate_top,
                                    Size(1, 4)
          plate_top2(Array of int32): The coordinates of the middle plate_top,
                                    Size(1, 4)                                                                            
          bottom(Array of int32): The coordinates of the bottom line,
                                    Size(1, 4)           
          left_walls(Array of int32): The coordinate of the left walls
          right_walls(Array of int32): The coordinate of the right walls          
    Returns:
          left_out_roof(Array of int32): The coordinate of left out roof
          Right_out_roof(Array of int32): The coordinate of right out roof
    """    
    
    # ====== Detect left out roof and right out roof
    left_out_roof = np.array([])
    right_out_roof = np.array([])
    
    # If left wall is placed at position that is greater than 8%(0.08)
    # of image width, then detect left roof
    if left_walls[0][0] > 0.08 * image.shape[1]:
        try:
            # Threshod value is 0.15, detecting the vertical lines that is 
            # greater than 0.15 *(plate top y - bottom y)
            # depth value is 0.05, detecting the vertical lines 
            # in 5% region of left of image
            if number_of_floor > 1:
                left_out_roof = detect_left_wall(
                    lines, plate_top.copy(), plate_top2.copy(), 
                    plate_top.copy(), bottom.copy(), image, th=0.15, 
                    depth=0.15, roof=True, out_roof = True)
                
            # If left out roof is place in impossible position, it is set as empty
            if len(left_out_roof)>0:
                if left_out_roof[0][0] > left_walls[1][0] - 30:
                    left_out_roof = np.array([])
        except:
          pass
      
    # If right wall is placed at position that is smaller than 92%(1-0.08
    # of image width, then detect right roof
    if right_walls[0][0] < 0.92 * image.shape[1]:
      
        # threshod value is 0.15, detecting the vertical lines that is greater 
        # than 0.15 *(plate top y - bottom y)
        # depth value is 0.95, detecting the vertical lines 
        # in 5% region of right image        
        if number_of_floor > 1:
            right_out_roof = detect_right_wall(
                lines, plate_top.copy(), plate_top2.copy(), plate_top.copy(),
                bottom.copy(), image, th=0.3, depth=0.85, roof=True,
                out_roof = True)
            
            # if right out roof is place in impossible position, it is set as empty
            if len(right_out_roof)>0:
                if right_out_roof[0][0] < right_walls[1][0] + 30:
                    right_out_roof = np.array([])
    return  left_out_roof, right_out_roof 

def detect_reft_walls(number_of_floor, lines, plate_top, plate_top2,
                      plate_top3,  bottom, image):
    """_summary_
    
    Args:
          number_of_floor(int): the number of floor of story building
          lines (Array of int32): The coordinates of the detected lines 
                                  in elevation image, Size(n, 1, 4)
          plate_top(Array of int32): The coordinates of 
                                     the upper plate top, Size(1, 4)
          plate_top2(Array of int32): The coordinates of 
                                     the middle plate top, Size(1, 4)  
          plate_top3(Array of int32): The coordinates of 
                                     the below plate top, Size(1, 4)
          bottom(Array of int32): The coordinates of bottom lines                                                               
          image(Array of int32): Elevation image
   
    Returns:
          left_walls(Array of int32): The coordinates of left walls
    """    
    
    # ====== Detect left walls =====
    left_walls = np.array([])
    if number_of_floor == 1:
    
        left_walls = detect_left_wall(
          lines, plate_top.copy(), bottom.copy(), plate_top.copy(), 
          bottom.copy(), image, gap = 30)

    elif number_of_floor == 2:
        # Left wall of upper floor
        left_walls = detect_left_wall(
          lines, plate_top.copy(), plate_top2.copy(), plate_top.copy(), 
          bottom.copy(), image, gap = 30)
        
        # Left wall of lower floor
        left_walls2 = detect_left_wall(
          lines, plate_top2.copy(), bottom.copy(), plate_top.copy(), 
          bottom.copy(), image, wall = left_walls, gap = 30)
        
        if left_walls.size == 0:
            left_walls = left_walls2
        if left_walls2.size == 0:
            left_walls2 = left_walls
        left_walls = np.concatenate((left_walls, left_walls2), axis = 0)

    elif number_of_floor == 3:
        # Left wall of upper floor
        left_walls = detect_left_wall(
          lines, plate_top.copy(), plate_top2.copy(), plate_top.copy(), 
          bottom.copy(), image, gap = 30)
        
        # Left wall of middle floor
        left_walls2 = detect_left_wall(
          lines, plate_top2.copy(), plate_top3.copy(), plate_top.copy(), 
          bottom.copy(), image, gap = 10)
       
        if len(left_walls) == 0:
           left_walls = left_walls2  
        if len(left_walls2) == 0:
           left_walls2 = left_walls 
        left_walls12 = np.concatenate((left_walls, left_walls2), axis = 0)
        
        # Left wall of lower floor
        left_walls3 = detect_left_wall(
          lines, plate_top3.copy(), bottom.copy(), plate_top.copy(), 
          bottom.copy(), image, wall = left_walls12, gap = 30)   
        
        if left_walls.size == 0:
            if left_walls2.size > 0:
                left_walls = left_walls2
            else:
                left_walls = left_walls3
            
            left_walls2 = left_walls      
        
        if left_walls2.size == 0:
            left_walls2 = left_walls
        if left_walls3.size == 0:
            left_walls3 = left_walls2
        left_walls = np.concatenate((left_walls, left_walls2, 
                                     left_walls3), axis = 0)
    return left_walls


def detect_right_walls(number_of_floor, lines, plate_top, plate_top2, 
                       plate_top3,  bottom, image):  
    """_summary_
    
    Args:
          number_of_floor(int): the number of floor of story building
          lines (Array of int32): The coordinates of the detected lines 
                                  in elevation image, Size(n, 1, 4)
          plate_top(Array of int32): The coordinates of 
                                     the upper plate top, Size(1, 4)
          plate_top2(Array of int32): The coordinates of 
                                     the middle plate top, Size(1, 4)  
          plate_top3(Array of int32): The coordinates of 
                                     the below plate top, Size(1, 4)
          bottom(Array of int32): The coordinates of bottom lines                                                               
          image(Array of int32): Elevation image
   
    Returns:
          right_walls(Array of int32): The coordinates of right walls
    """       
    
    right_walls = np.array([])
    if number_of_floor == 1:
        
        right_walls = detect_right_wall(
            lines, plate_top.copy(), bottom.copy(), plate_top.copy(),
            bottom.copy(), image, gap = 30)
        
    elif number_of_floor == 2:
        # Right wall of upper floor
        right_walls = detect_right_wall(
            lines, plate_top.copy(), plate_top2.copy(), plate_top.copy(),
            bottom.copy(), image,  gap = 30)
        
        # Right wall of lower floor
        right_walls2 = detect_right_wall(
            lines, plate_top2.copy(), bottom.copy(), plate_top.copy(),
            bottom.copy(), image, wall = right_walls.copy(), gap = 30)
        
        if len(right_walls) == 0:
            right_walls = right_walls2
        if len(right_walls2) == 0:
            right_walls2 = right_walls          
        
        right_walls = np.concatenate((right_walls, right_walls2), axis = 0)
        
    elif number_of_floor == 3:
        # Right wall of upper floor
        right_walls = detect_right_wall(
            lines, plate_top.copy(), plate_top2.copy(), plate_top.copy(),
            bottom.copy(), image, gap = 30)
        
        # Right wall of middle floor
        right_walls2 = detect_right_wall(
            lines, plate_top2.copy(), plate_top3.copy(), plate_top.copy(),
            bottom.copy(), image, gap = 30)
            
        if len(right_walls) == 0:
           right_walls = right_walls2  
        if len(right_walls2) == 0:
           right_walls2 = right_walls         
        
        right_walls12 = np.concatenate((right_walls, right_walls2), axis = 0)
        
        # Right wall of lower floor
        right_walls3 = detect_right_wall(
            lines, plate_top3.copy(), bottom.copy(), plate_top.copy(),
            bottom.copy(), image, 
            wall = right_walls12.copy(), gap = 30)     
        
        if len(right_walls) == 0:
            if len(right_walls2) > 0:
                
                right_walls = right_walls2
            else:
                right_walls = right_walls3
        
        if len(right_walls2) == 0:
            right_walls2 = right_walls
        if len(right_walls3) == 0:
                right_walls3 = right_walls2   
           
        right_walls = np.concatenate((right_walls, right_walls2,
                                      right_walls3), axis = 0)     
    return right_walls

def get_horizontal_pressure_areas(image_path: str, output_path: str,
                                  image_length: float, image_height: float,
                                  additional_lines: bool = True) -> areas:
    """Gets the sum of all the pressure areas for each floor.
    
    Args:
      image_path: A path to the image being detected on a local machine
        (full path). The image is always in the form of .png.
      image_length: The length of the image in real life scale in feet (from x==0
        to x==len(x-1)).
      image_height: The length of the image in real life scale in feet
      output_path : A path to be saved.
    
    Returns:
      An object holding dictionaries mapping each floors index to their
        respective a, b, c, and d areas.
    """
    # Load the image
    original_image = cv2.imread(image_path)
    
    # Detect interested region
    interested_region_image, image_len, image_ht, p0 = detect_interest_region(
        original_image.copy(), image_length, image_height)
    
    width = interested_region_image.shape[1]
    height = interested_region_image.shape[0]
    
    # Image resize image's width to 1050 to detect lines easy
    ratio = 1050 / width
    height2 = int(height * ratio)
    resized_image = cv2.resize(interested_region_image, (1050, height2))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
    # Apply edge detection to find the edges of the building
    edges = cv2.Canny(gray, 50, 150)
    
    # Apply line detection
    lines = cv2.HoughLinesP(
        edges.copy(), 1, np.pi / 180, threshold = 20,
        minLineLength=20, maxLineGap = 10)
   
    # Exchange y coordinates of horizonal lines so that first point's 
    # y coordinate is smaller than the second point's y coordinate
    for n, line in enumerate(lines):
      x1, y1, x2, y2 = line[0]
      if y1 > y2 and abs(x2-x1)<5:
        line[0][1] = y2
        line[0][3] = y1
        lines[n] = line
   
    # ==== Detect the top of roof ===
    roof_top = detect_roof_top(lines, resized_image)
    x1, y1, x2, y2 = roof_top[0]
    cv2.line(resized_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    # ==== Detect the bottom ===
    lines2 = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold = 20, minLineLength = 100, 
                            maxLineGap = 10)
    
    # Draw the detected lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            cv2.line(resized_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    bottom = detect_bottom(lines2.copy(), gray)
    x1, y1, x2, y2 = bottom[0]
    cv2.line(resized_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    # ==== Detect the top of plate ===
    # Apply edge detection to find the edges of the building
    edges = cv2.Canny(gray, 180, 200)  #50, 150, 
    
    lines_for_plate_top = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold = 20, minLineLength = 30, 
                            maxLineGap = 10) 
    
    # Apply edge detection to find the edges of the building
    edges = cv2.Canny(gray, 50, 150)  #50, 150, 
    lines31 = cv2.HoughLinesP(edges, 1, np.pi / 180,
                              threshold = 20, minLineLength = 30, 
                              maxLineGap = 10)
    lines3 = np.concatenate((lines_for_plate_top, lines31))
    
    plate_top = detect_plate_top(lines.copy(), lines3.copy(), 
                                 roof_top.copy(), bottom.copy(), resized_image)
    x1, y1, x2, y2 = plate_top[0]
    cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # ==== Determine the number of floor ====
    number_of_floor = get_number_of_floor(lines.copy(), roof_top.copy(), 
                                          plate_top.copy(), bottom.copy(),
                                          resized_image)
    
    # ==== Detect the second top of plate ===
    plate_top2 = []
    if number_of_floor > 1:
        plate_top2 = detect_plate_top23(
            lines, roof_top.copy(), plate_top.copy(), bottom.copy(),
            resized_image, floor = 2,  number_of_floor = number_of_floor)
    try:
        x1, y1, x2, y2 = plate_top2[0]
        cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    except:
        pass
    
    # ==== Detect the third top of plate ===
    plate_top3 = []
    if number_of_floor == 3:
        plate_top3 = detect_plate_top23(
            lines, roof_top.copy(), plate_top.copy(), bottom.copy(), 
            resized_image, floor = 3, number_of_floor = 3)
    try:
        x1, y1, x2, y2 = plate_top3[0]
        cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    except:
        pass
      
    # =====  Detect roof left and roof right wall =====
    roof_left = detect_roof_left(
      lines.copy(), lines3.copy(), roof_top.copy(), plate_top.copy(), 
      resized_image, th=0.25, roof = True)
     
    try:
        x1, y1, x2, y2 = roof_left[0]
        cv2.line(resized_image, (x1, y1), (x2, y2), (200, 200, 0), 2)
    except:
        pass
    
    # Right
    roof_right = detect_roof_right(
      lines.copy(), lines3.copy(), roof_top.copy(), 
      plate_top.copy(), resized_image, th = 0.25, roof = True)
    
    try:
        x1, y1, x2, y2 = roof_right[0]
        cv2.line(resized_image, (x1, y1), (x2, y2), (200, 200, 0), 2)
    except:
        pass
     
    # === Detect the rest of roof ===
    # Apply edge detection to find the edges of the building
    edges4 = cv2.Canny(gray, 50, 150)
    
    # Apply line detection
    lines4 = cv2.HoughLinesP(edges4, 1, np.pi/180,
                            threshold = 20, minLineLength = 30, maxLineGap = 5)  
    roof_rest = detect_roof_rest(lines4, roof_top.copy(), roof_left.copy(),
                                 roof_right.copy(), plate_top.copy(), 
                                 resized_image, [], 0)
    
    for k in range(len(roof_rest)):
      try:
          x1, y1, x2, y2 = roof_rest[k][0]
          cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
      except:
          pass
    
    # ====== Detect left walls =====
    left_walls = detect_reft_walls(number_of_floor, lines, plate_top, plate_top2,
                      plate_top3, bottom, resized_image)    

    for k in range(len(left_walls)):
        x1, y1, x2, y2 = left_walls[k]
        cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)          
    
    # ====== Detect right wall =====
    right_walls = detect_right_walls(number_of_floor, lines, plate_top, plate_top2, 
                           plate_top3, bottom, resized_image)

    for k in range(len(right_walls)):
        x1, y1, x2, y2 = right_walls[k]
        cv2.line(resized_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # ====== Detect left out roof and right out roof
    left_out_roof, right_out_roof= detect_left_right_out_roof(
            resized_image, number_of_floor, lines, left_walls, 
            plate_top, plate_top2, bottom, right_walls)
    
    if len(left_out_roof) > 0:
        x1, y1, x2, y2 = left_out_roof[0]
        cv2.line(resized_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  
        
    if len(right_out_roof) >0:
        x1, y1, x2, y2 = right_out_roof[0]
        cv2.line(resized_image, (x1, y1), (x2, y2), (255, 0, 0), 2)    
    roof_list = roof_rest.copy()
    try:
        roof_list.append(roof_top)
    except:
        pass
    
    # Calculate the total square footage of the area labeled for each letter
    # get the coordinates of  points for roof top
    roof_points = []
    try:
        roof_points.append([roof_left[0][0], roof_left[0][1]])
      
        while len(roof_list) > 0:
          dist = []
          for k in range(len(roof_list)):
            d = np.sqrt((roof_points[-1][0]-roof_list[k][0][0])
                        **2 + (roof_points[-1][1] - roof_list[k][0][1])**2)
            d = np.sqrt((roof_points[-1][0]-roof_list[k][0][0])
                        **2 )
            dist.append(d)
          n = np.argmin(dist)
          roof_points.append([roof_list[n][0][0], roof_list[n][0][1]])
          roof_points.append([roof_list[n][0][2], roof_list[n][0][3]])
          roof_list.pop(n)
      
        roof_points = np.array(roof_points)
    except:
        pass
    roof_points = np.array(roof_points)
    
    left_bottom = [left_walls[-1][0], bottom[0][1]]
    right_bottom = [right_walls[-1][0], bottom[0][1]]
    plate_top_left = [left_walls[-1][0], plate_top[0][1]]
    plate_top_right = [right_walls[-1][0], plate_top[0][1]]  
    
    p1 = [left_bottom[0] + 0.2 *
          (right_bottom[0] - left_bottom[0]), left_bottom[1]]
    p2 = [left_bottom[0] + 0.8 *
          (right_bottom[0] - left_bottom[0]), bottom[0][1]]
     
    # Constat to convert from image width resolution to foot width
    conv1 = (image_len/resized_image.shape[1])
    # Constat to convert from image height resolution to foot height
    conv2 = (image_ht/resized_image.shape[0])
    
    # Correction constant , this is used to correcting areas of  roof and upper area
    # because real plate line is in bit above place than the detected plate line
    if ((plate_top[0][1] - roof_top[0][1])/(0.5*(bottom[0][1] - plate_top[0][1]))< 1 and
        (plate_top[0][1] - roof_top[0][1])/(0.5*(bottom[0][1] - plate_top[0][1]))>0.75):
        cc = 0.02
    else:
        cc = 0.07 * ((plate_top[0][1] - roof_top[0][1])/
                    (0.5*(bottom[0][1] - plate_top[0][1])))
    
    # Calulate areas of roof region
    # Area of left B
    x1 = roof_left[0][0]
    y = plate_top[0][1] - cc * (plate_top[0][1] - roof_top[0][1])
    x2 = p1[0]
    
    try:
          B11, A_roof1, _, _ = get_area_BD(
              roof_points.copy(),roof_top, roof_left, roof_right, plate_top,
              lines3.copy(), gray.copy(), x1, x2, y, B =True)
    except:
          B11 = 0
          A_roof1 = 0
        
    # Area of right B
    x1 = p2[0]
    x2 = roof_right[0][0]
    try:
         B12, A_roof2, _, _ = get_area_BD(
             roof_points.copy(), roof_top, roof_left, roof_right,plate_top,
             lines3.copy(), gray.copy(), x1, x2, y, B =True)
    except:
          B12 = 0
          A_roof2 = 0
        
    # Area of D
    x1 = p1[0]
    x2 = p2[0]
    try:
          D, C_roof, D_point, D_area_center = get_area_BD(
              roof_points.copy(),roof_top, roof_left, roof_right,plate_top,
              lines3.copy(), gray.copy(), x1, x2, y)
    except:
          D = 0
          C_roof = 0
    
    if D * conv1 * conv2 < 2:
        D = 0
    if B11 * conv1 * conv2 < 2:
        B12 = 0
    if B12 * conv1 * conv2 <2:
        B12 = 0
    if number_of_floor == 1:
         
        B2 = 0
        D2 = 0
        # Areas of left roof A
        A11 = (p1[0] - left_walls[0][0]) *\
        (0.47 * ((bottom[0][1] - plate_top_left[1]) +
               cc * (plate_top[0][1] - roof_top[0][1]))) 
        
        # Areas of right roof A
        A12 =  (right_walls[0][0] - p2[0]) * \
        (0.47 * ((bottom[0][1] - plate_top_right[1]) +
               cc * (plate_top[0][1] - roof_top[0][1])))
        
        # Area of roof C
        C1 =(p2[0] - p1[0]) * \
        (0.47 * ((bottom[0][1] - plate_top[0][1]) +
               cc * (plate_top[0][1] - roof_top[0][1])))
              
    else:    
        # Areas of left roof A
        A11 = (p1[0] - left_walls[0][0]) * \
        (0.467 * ((plate_top2[0][1] - plate_top_left[1]) + 
               cc * (plate_top[0][1] - roof_top[0][1])) ) 
        
        # Areas of right roof A
        A12 =  (right_walls[0][0] - p2[0]) * \
        (0.467 * ((plate_top2[0][1] - plate_top_right[1]) + 
               cc * (plate_top[0][1] - roof_top[0][1])))
     
        # Area of roof C
        C1 =(p2[0] - p1[0]) * \
        (0.467 * ((plate_top2[0][1] - plate_top[0][1]) + 
               cc * (plate_top[0][1] - roof_top[0][1])))
        
        if number_of_floor ==2: 
            
            # Area of left upper A
            A21 = A11 +  (p1[0] - left_walls[1][0]) * \
            (0.57 * ((left_bottom[1] - plate_top2[0][1])))
            
            # Area of right upper A
            A22 = A12 +  (right_walls[1][0] - p2[0]) * \
            (0.57 * ((left_bottom[1] - plate_top2[0][1])))
            
            # Area of upper C
            C2 = C1 +  (p2[0] - p1[0]) * \
            (0.57 * ((left_bottom[1] - plate_top2[0][1])))
            
            B13 = 0
            # Area of left upper B
            if len(left_out_roof) > 0:
                B13 = B13 + (left_out_roof[0][3] - left_out_roof[0][1]) * \
                    (left_bottom[0] - left_out_roof[0][0])
            # Area of right upper B
            if len(right_out_roof)>0:
              B13 = B13 + (right_out_roof[0][3] - right_out_roof[0][1]) * \
                (right_out_roof[0][0] - right_bottom[0])
          
            # Calculate B2 and D2
            B2 = 0
            D2 = 0    
            x1 = p1[0]
            x2 = p2[0]
        
            if len(plate_top2)>0:
                B2, D2 = get_area_roof2(
                    plate_top2, left_walls[0], right_walls[0], lines.copy(),
                    resized_image.copy(), x1, x2)
    
                if D2 * conv1 * conv2 < 25:
                    D2 = 0
                    B2 = 0
                    
        if number_of_floor == 3:    
            # Area of left upper A
            A21 = A11 +  (p1[0] - left_walls[1][0]) * \
            (0.57 * ((plate_top3[0][1] - plate_top2[0][1])))
            
            # Area of right upper A
            A22 = A12 +  (right_walls[1][0] - p2[0]) * \
            (0.57 * ((plate_top3[0][1] - plate_top2[0][1])))          
            
            A31 = A21 - A11  +  (p1[0] - left_walls[2][0]) * \
            (0.5 * ((bottom[0][1] - plate_top3[0][1])))
            
            # Area of right upper A
            A32 = A22 - A12 +  (right_walls[2][0] - p2[0]) * \
            (0.5 * ((bottom[0][1] - plate_top3[0][1])))           
            
            # Area of upper C
            C2 = C1 +  (p2[0] - p1[0]) * \
            (0.57 * ((plate_top3[0][1] - plate_top2[0][1])))
          
            C3 = C2 - C1 +  (p2[0] - p1[0]) * \
            (0.5 * ((bottom[0][1] - plate_top3[0][1])))
            
            B13 = 0
            # Area of left upper B
            if len(left_out_roof) > 0:
                B13 = B13 + (left_out_roof[0][3] - left_out_roof[0][1]) * \
                    (left_bottom[0] - left_out_roof[0][0])
            # Area of right upper B
            if len(right_out_roof)>0:
              B13 = B13 + (right_out_roof[0][3] - right_out_roof[0][1]) * \
                (right_out_roof[0][0] - right_bottom[0])
          
            # Calculate B2 and D2
            B2 = 0
            D2 = 0    
            x1 = p1[0]
            x2 = p2[0]
        
            if len(plate_top2)>0:
              
                B2, D2 = get_area_roof2(
                    plate_top2, left_walls[0], right_walls[0], lines.copy(),
                    resized_image.copy(), x1, x2)
                
                if B2 * conv1 * conv2 < 20:
                    B2 = 0
                if D2 * conv1 * conv2 < 25:
                    D2 = 0   

    # roof is 1 and upper is 2. Below is 3
    if number_of_floor == 1:
              
        a_area = {
            0: (A11 + A12 + A_roof1 + A_roof2) * conv1 * conv2
            }
        
        b_area = {
            0: ( B11 + B12 ) * conv1 * conv2
            }
        
        c_area = {
            0: (C1 + C_roof) * conv1 * conv2
            }
        d_area = {
            0: D * conv1 * conv2
            }
        
    elif number_of_floor == 2:
        
        a_area = {
            0: (A11 + A12 + A_roof1 + A_roof2) * conv1 * conv2, 
            1: (A21 + A22- B2) * conv1 * conv2
            }
        
        b_area = {
            0: ( B11 + B12 + B13) * conv1 * conv2, 
            1: B2 * conv1 * conv2
            }
        
        c_area = {
            0: (C1 + C_roof) * conv1 * conv2, 
            1: (C2- D2) * conv1 * conv2
            }
        
        d_area = {
            0: D * conv1 * conv2, 
            1: D2 * conv1 * conv2
            }

    elif number_of_floor == 3:

        a_area = {
            0: (A11 + A12 + A_roof1 + A_roof2) * conv1 * conv2, 
            1: (A21 + A22 - B2) * conv1 * conv2,
            2: (A31 + A32) * conv1 * conv2,
            }
        
        b_area = {
            0: (B11 + B12 + B13) * conv1 * conv2, 
            1: B2 * conv1 * conv2, 
            }
        
        c_area = {
            0: (C1 + C_roof) * conv1 * conv2, 
            1: (C2 - D2) * conv1 * conv2, 
            2: C3 * conv1 * conv2,
            }
        
        d_area = {
            0: D * conv1 * conv2, 
            1: D2 * conv1 * conv2, 
            }      

    ## Drawing
    draw(
        ratio, p0, p1, p2, B11,B12, B2, D, D2, D_area_center,D_point, C_roof,
        roof_top, plate_top, plate_top2, plate_top3, bottom, left_bottom,
        right_bottom, left_out_roof, right_out_roof, number_of_floor,
        resized_image, original_image, output_path, image_path, additional_lines)

    return areas(
      a_area=a_area,
      b_area=b_area,
      c_area=c_area,
      d_area=d_area
    )
