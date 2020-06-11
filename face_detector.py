# waitKey(delayValue) , delayValue<=0 is a special value means forever
# delayValue = any milliseconds ,then the method returns ASCII code[ord('key pressed')] of any key pressed or
# -1 if no key is pressed during delayed time.
import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #face cascade object is created
img = cv2.imread("face.jpg")
gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img,
scaleFactor= 1.05,                         #scaling factor to create image pyramid smaller the image lesser the work for detector large face converted to smaller
minNeighbors=5)              # gives 4 values: top corner x,y  and width,height

for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)  # updating img object already created

resized = cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
cv2.imshow("gray",resized)
k=cv2.waitKey(1000)
cv2.destroyAllWindows()
print(k)
