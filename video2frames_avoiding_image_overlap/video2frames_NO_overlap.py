import cv2
import os
import time

# Use VideoCapture to capture video, here we use local video
cap = cv2.VideoCapture("./IC14_CATEDRAL_0421_T2_vuelta_b.MP4")

# Create a file to save the frame of the video
save_path = ""

save_good_frame_path = ""

flag = 0
if cap.isOpened():
    flag = 1
else:
    flag = 0

frame_count = 0
img_path = ""

base_frame = None
magic_number = 100
cont = 0

if flag == 1:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count == 1 or frame_count % magic_number == 0:
            img_path = save_good_frame_path + "%s.jpg" % str(frame_count)
            if frame_count == 1:
                cv2.imwrite(img_path, frame)

            if base_frame is None:
                base_frame = frame
                continue

            # check for similarities
            sift = cv2.SIFT_create()

            # check keypoints and descriptions of images
            kp_1, desc_1 = sift.detectAndCompute(base_frame, None)
            kp_2, desc_2 = sift.detectAndCompute(frame, None)

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

            if len(good_points) < 5:
                cont += 1

                if cont == 1:
                    magic_number = round(frame_count * (2 / 3), -2)
                base_frame = frame

                cv2.imwrite(img_path, frame)
                base_frame = frame
