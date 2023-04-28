from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from skimage import morphology
import math
import glob
import os
import csv

        
register_coco_instances("ropes",{},"./cuerdas.json", "")
all_especies_metadata = MetadataCatalog.get("ropes")
dataset_dicts = DatasetCatalog.get("ropes")

cfg = get_cfg()

# load weights
cfg.MODEL.WEIGHTS = "C:/Users/ser/detectron2/pesos_cuerdas_mas_ajustadas/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65   # set the testing threshold for this model
cfg.merge_from_file("C:/Users/ser/detectron2/pesos_cuerdas_mas_ajustadas/config.yaml")
cfg.DATASETS.TEST=("ropes",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.SOLVER.BASE_LR=0.02
cfg.MODEL.ROI_HEADS.NUM_CLASSES=1

# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

#image
im_path = "./IC14_CATEDRAL_0421_T1_VUELTA/"
transect="IC14_CATEDRAL_0421_T1_VUELTA"


with open('./10species/'+transect+"/"+transect+'.csv', mode='w',newline='') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',')
    
    employee_writer.writerow(['Transect','Image','Specie','area(m2)','Number of specimen','Cobertura(%)','Cobertura(m2)'])
    for image_path in glob.glob(im_path+"*.jpg"):
        name=imagen_path.split("/"+transecto+"\\")
        name=name[1]
        
        #inference
        im=cv2.imread(image_path)
        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1],
                           metadata=all_especies_metadata, 
                           scale=1, 
                           instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
            )
        v,imagen,nuevo_negro = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        ret,thres = cv2.threshold(nuevo_negro,10,255,cv2.THRESH_BINARY)
        thres=np.uint8(thres)
        one_channel=thres[:,:,1]

        mask=one_channel
        result1=im.copy()
        result1[mask==0] = 0
        result1[mask != 0]=im[mask!=0]

        low_blue=np.array([120,70,70])
        high_blue=np.array([200,110,110])
        
        #detect pixel between low and high_blue
        mask=cv2.inRange(result1,low_blue,high_blue)

        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        noZero=cv2.findNonZero(mask)
        noZero_x=[]
        noZero_y=[]

        if noZero is not None:
            for i in noZero:
                noZero_x.append(i[0][0])
                noZero_y.append(i[0][1])   
                
                mask=cv2.circle(mask,(i[0][0],i[0][1]),10,(255,255,255),-1)

            
            contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            centers=[]
            for i in contours:
               
                moments=cv2.moments(i)
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
             
                
                mask=cv2.circle(mask,(cx, cy), 1, (0,0,255), -1)
                font=cv2.FONT_HERSHEY_SIMPLEX
                
                
                cv2.putText(mask,"(x: " + str(cx) + ", y: " + str(cy) + ")",(cx+10,cy+10), font, 0.5,(255,255,255),1)
                centre=(cx,cy)
                centers.append(center)
   
            MAX_distance=0
            MAX_distances=[]
            for i in range(len(centers)-1):
                point1=resultado_centros[i]
                point2=resultado_centros[i+1]
                x=point1[0]-point2[0]
                y=point1[1]-point2[1]
                distance_px=math.sqrt(pow(x,2)+pow(y,2))   
                if MAX_distance<distance_px:
                    MAX_distance=distance_px
                
                MAX_distances.append(distance_px)
            mean_distances=[]
            for j in range(len(MAX_distances)):
                if MAX_distances[j]>3/4*MAX_distance:
                    mean_distances.append(MAX_distances[j])
            MAX_distance=np.mean(mean_distances)        

            height, width = im.shape[0:2]
            real_distance_laser=0.2#m
            
            if maxima_distancia>0:
                area=(height*real_distance_laser/MAX_distance)*(width*real_distance_laser/MAX_distance)
                area=round(area,2)

        else:
            area=0
              
        
