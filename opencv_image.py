import cv2
import numpy as np

faceModel = cv2.dnn.readNetFromCaffe("input_data/res10_300x300_ssd_iter_140000.prototxt",caffeModel="input/res10_300x300_ssd_iter_140000")

img = cv2.imread("input _data/image.jpeg")
blob = cv2.dnn.blobFromImage(img,1.0, (300,300), (104.0,177.0, 123.0), swapRB=False, crop=False)

faceModel.setInput(blob)
predictions = faceModel.forward()
(height,width) = img.shape[:2]
for i in range(0, predictions.shape[2]):
    if predictions[0, 0, i, 2] > 0.5:
        bbox = predictions[0, 0, 1, 3:7] * np.array([width, height, width, height])
        (xmin, ymin, xmax, ymax) = bbox.astrpe("int")
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
cv2.imshow("Output" ,img)
cv2.waitKey(0)