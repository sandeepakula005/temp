import cv2
import numpy as np
from imutils.video import FPS

faceModel = cv2.dnn.readNetFromCaffe("input_data/res10_300x300_ssd_iter_140000.prototxt",caffeModel="input/res10_300x300_ssd_iter_140000")
faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture("input_data/video.mp4")

success,img = cap.read()
height,width = img.shape[:2]

fps = FPS.start()

while success:
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    faceModel.setInput(blob)
    predictions = faceModel.forward()
    (height, width) = img.shape[:2]
    for i in range(0, predictions.shape[2]):
        if predictions[0, 0, i, 2] > 0.5:
            bbox = predictions[0, 0, 1, 3:7] * np.array([width, height, width, height])
            (xmin, ymin, xmax, ymax) = bbox.astrpe("int")
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    cv2.imshow("Output", img)

    key = cv2.waitKey(1)& 0xFF
    if key==ord('q'):
        break
    fps.update()
    (success,img) = cap.read()

fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("FPS: :{.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()