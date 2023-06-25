import cv2

face_detec = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_detec = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
video = cv2.VideoCapture(0)
while True: 
    succ_frame,frame = video.read()
    frame_gra = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detec.detectMultiScale(frame_gra)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        the_face = frame[y:y+h,x:x+w]
        face_grascale= cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
        eyes = eye_detec.detectMultiScale(face_grascale,scaleFactor=1.7,minNeighbors=20)
        for(x_,y_,w_,h_) in eyes:
            cv2.rectangle(the_face,(x_,y_) ,(x_+w_ ,y_+h_),(255,255,255),2)
    cv2.imshow("Face_Detection",frame)
    key =cv2.waitKey(1)
    if key == ord("q"):
        break
    
video.release()
cv2.destroyAllWindows()