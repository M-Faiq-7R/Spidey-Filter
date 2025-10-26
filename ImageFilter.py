import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_replace = cv2.imread('smn.png')
cam = cv2.VideoCapture(0)
def bg_remove_filter(image, replacement_face, return_image = 0 , expand = 45 , thresh = 40):
    faces= face_cascade.detectMultiScale(image, 1.3, 3)
    for x,y,w,h in faces:
        w = w+expand
        h = h+expand
        x = int(x- expand/2)
        y = int(y - expand/2)
        
        replacement_face_copy = cv2.resize(replacement_face ,(w,h))
        roi = image[y : y+h , x:x+w]
        img_gray = cv2.cvtColor(replacement_face_copy , cv2.COLOR_BGR2GRAY)
        _ , mask= cv2.threshold(img_gray , thresh , 255 ,cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        try:
            img_bg =  cv2.bitwise_and(roi ,roi , mask=mask)
        except:
            continue
        img_fg = cv2.bitwise_and(replacement_face_copy,replacement_face_copy,mask=mask_inv)
        combined = cv2.add(img_bg,img_fg)
        image[y : y+h , x:x+w] = combined
        
    if return_image:
        return image
    else:
        cv2.imshow('Spidey Filter' , image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

while 1:
    trash , img = cam.read()
    img = bg_remove_filter(img, image_replace ,1, thresh= 210)
    img = cv2.flip(img, 1)
    cv2.imshow('Image Spidy ' ,img)
    k = cv2.waitKey(1)
    
    if k & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
