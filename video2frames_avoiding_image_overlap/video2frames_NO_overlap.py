#deteccion de moviemineto en videos
import cv2
import os
import time

# Use VideoCapture para capturar video, aquí usamos video local
cap = cv2.VideoCapture("./IC14_CATEDRAL_0421_T2_vuelta_b.MP4")

# Crea un archivo para guardar el fotograma del video
save_path = " "

save_path_fotograma_bueno = " "

flag = 0
if cap.isOpened():
    flag = 1
else:
    flag = 0
 

i = 0
imgPath = ""

base = None
num_magico=100
cont=0

if flag == 1:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        i += 1
        
        if i==1  or i%(num_magico)==0:
            imgPath = save_path_fotograma_bueno+"%s.jpg" % str(i)
            if i==1:
                cv2.imwrite(imgPath, frame)
                
            if base is None:
                base = frame
                continue
                
            # check for similarities
            #sift = cv2.xfeatures2d.SIFT_create()
            sift = cv2.SIFT_create()
            
            # check keypoints and descriptions of images
            kp_1,desc_1 = sift.detectAndCompute(base,None)
            kp_2,desc_2 = sift.detectAndCompute(frame,None)

            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(desc_1, desc_2, k=2)
            
            good_points = []
            ratio = 0.6
            for m, n in matches:
                if m.distance < ratio*n.distance:
                    good_points.append(m) 
            print(len(good_points))
                
            if len(good_points)<5:
                cont+=1

                if cont==1:
                    num_magico=round(i*(2/3),-2)
                base=frame
                
                cv2.imwrite(imgPath, frame)
                base=frame
