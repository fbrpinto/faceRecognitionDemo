import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(1):
	ret, img = cap.read()
	if(ret == 1):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)
		for (x,y,w,h) in faces:
		    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
		cv2.imshow('Face detection',img)
		if(cv2.waitKey(30) & 0xFF == 27):
			break

cv2.destroyAllWindows()
