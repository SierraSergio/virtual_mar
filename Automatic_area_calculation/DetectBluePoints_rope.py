#inferimos con detectron2 para conseguir las mascaras(binarias) y aplicamos cv2.inpaint para tratar de suavizar el color.
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from PIL import Image
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from skimage import morphology
import math
import glob
import os
import csv

        
register_coco_instances("all_especies",{},"C:/Users/ser/detectron2/pesos_cuerdas_mas_ajustadas/cuerdas.json", "C:/Users/ser/detectron2/pesos_cuerdas_mas_ajustadas/")
all_especies_metadata = MetadataCatalog.get("all_especies")
dataset_dicts = DatasetCatalog.get("all_especies")

cfg = get_cfg()

# load weights
cfg.MODEL.WEIGHTS = "C:/Users/ser/detectron2/pesos_cuerdas_mas_ajustadas/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65   # set the testing threshold for this model
cfg.merge_from_file("C:/Users/ser/detectron2/pesos_cuerdas_mas_ajustadas/config.yaml")
cfg.DATASETS.TEST=("all_especies",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.SOLVER.BASE_LR=0.02
cfg.MODEL.ROI_HEADS.NUM_CLASSES=1

# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

#image
im_path = "C:/Users/ser/detectron2/IC14_CATEDRAL_0421_T1_VUELTA/"
transecto="IC14_CATEDRAL_0421_T1_VUELTA"


with open('./11especies/'+transecto+"/"+transecto+'.csv', mode='w',newline='') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',')
    
    employee_writer.writerow(['Transect','Image','Specie','area(m2)','Number of specimen','Cobertura(%)','Cobertura(m2)'])
    for imagen_path in glob.glob(im_path+"*.jpg"):
        nombre=imagen_path.split("/"+transecto+"\\")
        nombre=nombre[1]
        
        #inference
        im=cv2.imread(imagen_path)
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
        vector_noZero_x=[]
        vector_noZero_y=[]
        print(type(noZero))
        if noZero is not None:
            for i in noZero:
                vector_noZero_x.append(i[0][0])
                vector_noZero_y.append(i[0][1])   
                
                mask=cv2.circle(mask,(i[0][0],i[0][1]),10,(255,255,255),-1)

            
            contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            resultado_centros=[]
            for i in contours:
               
                momentos=cv2.moments(i)
                cx = int(momentos['m10']/momentos['m00'])
                cy = int(momentos['m01']/momentos['m00'])
             
                
                mask=cv2.circle(mask,(cx, cy), 1, (0,0,255), -1)
                font=cv2.FONT_HERSHEY_SIMPLEX
                
                
                cv2.putText(mask,"(x: " + str(cx) + ", y: " + str(cy) + ")",(cx+10,cy+10), font, 0.5,(255,255,255),1)
                centro=(cx,cy)
                resultado_centros.append(centro)

            
            maxima_distancia=0
            distancias_maximas=[]
            for i in range(len(resultado_centros)-1):
                punto1=resultado_centros[i]
                punto2=resultado_centros[i+1]
                x=punto1[0]-punto2[0]
                y=punto1[1]-punto2[1]
                distancia_px=math.sqrt(pow(x,2)+pow(y,2))   
                if maxima_distancia<distancia_px:
                    maxima_distancia=distancia_px
                
                distancias_maximas.append(distancia_px)
            distancias_medias=[]
            for j in range(len(distancias_maximas)):
                if distancias_maximas[j]>3/4*maxima_distancia:
                    distancias_medias.append(distancias_maximas[j])
            maxima_distancia=np.mean(distancias_medias)        

            height, width = im.shape[0:2]
            distancia_puntos_real=0.2#m
            
            if maxima_distancia>0:
                area=(height*distancia_puntos_real/maxima_distancia)*(width*distancia_puntos_real/maxima_distancia)
                area=round(area,2)
                print(area)
        else:
            area=0
              
        