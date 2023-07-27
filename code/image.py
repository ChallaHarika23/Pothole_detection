#importing necessary libraries
import cv2
import os

#reading test image
img = cv2.imread("C:/Users/hp/Desktop/POTHOLE_MAIN_DETECTION/pothole_coordinates/test/img-472_jpg.rf.d71e2cae627685f2ad46e4182bbfb68a.jpg") #image name

#reading label name from obj.names file
with open(os.path.join("model/project_files",'C:/Users/hp/Desktop/POTHOLE_MAIN_DETECTION/model/project_files/obj.names'), 'r') as f:
    classes = f.read().splitlines()

#importing model weights and config file
net = cv2.dnn.readNet('C:/Users/hp/Desktop/POTHOLE_MAIN_DETECTION/model/project_files/yolov4_tiny.weights', 'C:/Users/hp/Desktop/POTHOLE_MAIN_DETECTION/model/project_files/yolov4_tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

#detection 
#for (classId, score, box) in zip(classIds, scores, boxes):
for (classId, score, box) in zip(classIds.flatten(), scores.flatten(), boxes):
    
    label = f"{classes[classId]}: {score * 100:.2f}%"
    x, y, w, h = box
    cv2.rectangle(img, (x,y), (x + w, y + h),
                  color=(0, 0, 255), thickness=1)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
 
cv2.imshow("pothole",img)
cv2.imwrite("resultimage"+".jpg",img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
