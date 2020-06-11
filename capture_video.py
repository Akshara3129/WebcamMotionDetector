import cv2,time,pandas
from datetime import datetime

video= cv2.VideoCapture(0)
#captures video using webcam. if another external camera used then 0,1 and so on
first_frame= None
#first frame is static background with which each frame is compared
status_list = [None,None]
df = pandas.DataFrame(columns=["Start","End"])
times = []

while True:
    check,frame = video.read();
    # video.read() gives out 2 parameters (true/false defining whether video turned on or not, ndarray)
    status = 0

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    # gaussianblur reduces the noise produced in captured frame- Image smoothing .
    #(21,21) - must be odd values - defines width and height of gaussian kernel for plotting data points in space.
    # 0 is standard deviation

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame = cv2.absdiff(first_frame ,gray)
    # produces image that gives difference between first frame and the current frame
    thres_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    # threshlod(frame, threshold , if greater than threshold apply colur(eg:255),) returns 2 values
    # 1st - threshold value used(eg:30) , 2nd - thresholded image(numpy array)
    thres_frame = cv2.dilate(thres_frame,None,iterations = 2)
    # smoothing threshold frame - to remove black holes from white areas
    #(frame, kernel array value, number of times image should iterated for smoothing)

    (cnts,_) = cv2.findContours(thres_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # returns (contours,heirarchy) heirarchy - contours in images are somehow related to one another
    # eg. some shapes are inside another shape then outer shape is taken as parent
    # curve joining continuous points along boundary
    # findContours()- like finding white object from black background- modifies the source image hence copy of it is passed
    # cv2.RETR_EXTERNAL - retrieves only the extreme outer contours
    # cv2.CHAIN_APPROX_NONE - all the boundary points are stored
    # cv2.CHAIN_APPROX_SIMPLE - removes all redundant points and compresses the contour, thereby saving memory.

    for contour in cnts:
        if(cv2.contourArea(contour) < 1000):
            continue
        status = 1

        (x,y,w,h) = cv2.boundingRect(contour)
        # boundingRect() returns coordinates of rectangle (top left (x,y) , width, height)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # creates rectangluar frame (frame, top left co-ord , bottom right co-ord, colour ,width)
    status_list.append(status)

    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Gaussian",gray)
    cv2.imshow("delta_frame",delta_frame)
    cv2.imshow("threshold frame",thres_frame)
    cv2.imshow("Rectangle",frame)

    key=cv2.waitKey(1)
    # generates new window for every 1 millisecond
    # waitKey(delayValue) , delayValue<=0 is a special value means forever
    # delayValue = any milliseconds ,then the method returns ASCII code[ord('key pressed')] of any key pressed or
    # -1 if no key is pressed during delayed time.

    if(key == ord('q')):
        if status == 1:
            times.append(datetime.now())
        break
print(times)
for i in range(0,len(times),2):
    df = df.append({"Start": times[i] ,"End": times[i+1]} ,ignore_index=True)

df.to_csv("Times.csv")
print(status_list)
video.release()
cv2.destroyAllWindows()
