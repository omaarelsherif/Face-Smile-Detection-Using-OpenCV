### Face smile Detection ###
"""
Description :
			  Simple face smile detection using haarcascade classifiers in opencv
"""

# 1 | Import modules
import cv2
import sys

# 2 | Create cascade classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# 3 | Detect simle
def detect_smile(img, gray):
    """Function to detect smile in face within an image"""
    
    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Loop over faces and draw bounding box over it
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect smile in the face        
        smile = smileCascade.detectMultiScale(roi_gray, 1.7, 20)
        
        print(f"Here : {type(smile)}")
        
        # Set smile state as text (Smiling or not)
        if len(smile) != 0:
            cv2.putText(img, "Smiling", (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Not smiling", (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255), 2, cv2.LINE_AA)
            
    # Show the image
    cv2.imshow("Smile Detection", img)
    cv2.waitKey(0)
    
# 4 | Load image by the path from cmd
path = sys.argv
img = cv2.imread(path[1])

# 5 | Convert the image to grey scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 6 | Apply smile detection and show the result image
detect_smile(img, img_gray)
