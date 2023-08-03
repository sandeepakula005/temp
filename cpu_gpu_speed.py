import cv2
import numpy as np
import time

npTmp = np.random.random((1824, 1624)).astype (np. float32)

npMat1 = np.stack([npTmp,npTmp],axis=2)
npMat2 = npMat1

cuMat1 = cv2.cuda_GpuMat()
cuMat2 = cv2.cuda_GpuMat()
cuMat1.upload(npMat1)
cuMat2.upload(npMat2)
start_time = time.time()
cv2.cuda.gemm(cuMat1, cuMat2, 1, None, 0, None, 1)
print("Using GPU -- %s seconds • -" %(time. time ()-start_time))
start_time = time.time()
cv2.gemm(npMat1, npMat2, 1, None, 0, None , 1)
print ("Using CPU -- %5 seconds - -" %(time. time ()-start_time))