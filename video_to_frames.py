import cv2

capture = cv2.VideoCapture(r"C:/Users/dmitr/neural/videos/false.mov")
 
frameNr = 0
 
while (True):
 
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(f"c:/Users/dmitr/neural/videos/converted_images/false/1frame_{frameNr}.jpg", frame)
 
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()
