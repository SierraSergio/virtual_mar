from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from PIL import Image
import numpy as np
import os
import glob
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import csv
import errno
import math


counter = 0
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x,y
        counter = counter + 1
        
def distancia_laser(img_real,a,b,col_real,row_real,laseres_x,laseres_y):
    pixel_real=img_real[a,b]
          
    if pixel_real[0]>245 and pixel_real[1]==255 and pixel_real[2]>240 :
        #print("he entrado", pixel_real, b,a)
        laseres_x.append(b)
        laseres_y.append(a)
        #opciones: guardar un vector con todos los valores que esten entre estos puntos y ver cual es el mayor y el menor y entonces restar.
    return laseres_x,laseres_y

def cropper_varios(org_image_path, mask_array,nombre,csv_imagenes_path,area,transecto,employee_writer):
        print(mask_array.shape[0])
        img = cv2.imread(org_image_path)
        height, width = img.shape[0:2]
        pixeles_totales=height*width
        
        area_conjunta=0
        cont_demos=cont_salma=cont_acant=cont_axinella=cont_schi=cont_parazo=cont_spira=cont_rete=cont_myri=cont_frondi=cont_dide=cont_agelas=0
        area_demos=area_salma=area_acant=area_axinella=area_schi=area_parazo=area_spira=area_rete=area_myri=area_frondi=area_dide=area_agelas=0
        for i in range(mask_array.shape[0]):
            
            mask_array1 = np.moveaxis(mask_array, 0, -1)
            mask_array2 = mask_array1[:,:,i:i+1]
            
            instances=outputs.get('instances')
            classes_tensor=instances.pred_classes
            classes_tensor=classes_tensor.cpu()
            classes=classes_tensor.numpy()
            clase=classes[i]
                        
            output = np.where(mask_array2 == False, 0, (np.where(mask_array2 == True, 255, img)))
            im = Image.fromarray(output)

            gray=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)                              
            pixeles=cv2.countNonZero(gray)
            area_abarcada=pixeles/pixeles_totales

            area_conjunta=area_conjunta+pixeles
            if clase==0:
               'Agelas oroides'
               cont_agelas=cont_agelas+1
               area_agelas=area_agelas+(area_abarcada*100)
               
            if clase==1:
               'Didenmnum sp.'
               cont_dide=cont_dide+1
               area_dide=area_dide+(area_abarcada*100)

               
            if clase==2:
               'Froindipora verrucosa'
               cont_frondi=cont_frondi+1
               area_frondi=area_frondi+area_abarcada*100
              
            if clase==3:
               'Myriapora truncata'
               cont_myri=cont_myri+1
               area_myri=area_myri+(area_abarcada*100)
               
            if clase==4:
               'Reteporella sp'
               cont_rete=cont_rete+1
               area_rete=area_rete+(area_abarcada*100)
               
            if clase==5:
               'Spirastrella cunctatrix'
               cont_spira=cont_spira+1
               area_spira=area_spira+(area_abarcada*100)
               
            if clase==7:
               "Parazoanthus axinellae" 
               cont_parazo=cont_parazo+1
               area_parazo=area_parazo+(area_abarcada *100)                   
                               
            if clase==8:
               "schizoretepora serratima"
               cont_schi=cont_schi+1
               area_schi=area_schi+(area_abarcada*100)
               
            if clase==9:
               "axinella sp"
               cont_axinella=cont_axinella+1
               area_axinella=area_axinella+(area_abarcada*100)
              
            if clase==10:
               "SP025C_Acanthella acuta"
               cont_acant=cont_acant+1
               area_acant=area_acant+(area_abarcada*100)
               
            if clase==11:
               "SP040B_Salmancina incrustans"
               cont_salma=cont_salma+1
               area_salma=area_salma+(area_abarcada*100)
               
            if clase==12:
               "SP019B_Demospongia sp."
               cont_demos=cont_demos+1
               area_demos=area_demos+(area_abarcada*100)
               
        
        if cont_agelas>0:
            employee_writer.writerow([transecto,nombre,'Agelas oroides',area,cont_agelas,area_agelas,round(area*area_agelas/100,2)])
        if cont_dide>0:
            employee_writer.writerow([transecto,nombre,'Didenmnum sp.',area,cont_dide,area_dide,round(area*area_dide/100,2)])
        if cont_frondi>0:
            employee_writer.writerow([transecto,nombre,'Froindipora verrucosa',area,cont_frondi,area_frondi,round(area*area_frondi/100,2)]) 
        if cont_myri>0:
            employee_writer.writerow([transecto,nombre,'Myriapora truncata',area,cont_myri,area_myri,round(area*area_myri/100,2)])
        if cont_rete>0:
            employee_writer.writerow([transecto,nombre,'Reteporella sp',area,cont_rete,area_rete,round(area*area_rete/100,2)])
        if cont_spira>0:
            employee_writer.writerow([transecto,nombre,'Spirastrella cunctatrix',area,cont_spira,area_spira,round(area*area_spira/100,2)])
        if cont_parazo>0:
            employee_writer.writerow([transecto,nombre,"Parazoanthus axinellae" ,area,cont_parazo,area_parazo,round(area*area_parazo/100,2)])  
        if cont_schi>0:
            employee_writer.writerow([transecto,nombre,"schizoretepora serratima",area,cont_schi,area_schi,round(area*area_schi/100,2)])  
        if cont_axinella>0:
            employee_writer.writerow([transecto,nombre,"axinella sp",area,cont_axinella,area_axinella,round(area*area_axinella/100,2)])
        if cont_acant>0:
            employee_writer.writerow([transecto,nombre,"SP025C_Acanthella acuta",area,cont_acant,area_acant,round(area*area_acant/100,2)])
        if cont_salma>0:
            employee_writer.writerow([transecto,nombre,"SP040B_Salmancina incrustans",area,cont_salma,area_salma,round(area*area_salma/100,2)])          
        if cont_demos>0:
            employee_writer.writerow([transecto,nombre,"SP019B_Demospongia sp.",area,cont_demos,area_demos,round(area*area_demos/100,2)])
        
        area_no_etiquetada=100-((area_conjunta/pixeles_totales)*100)
        employee_writer.writerow([transecto,nombre,"No identificado",area,"1",area_no_etiquetada,round(area*area_no_etiquetada/100,2)])


#register
register_coco_instances("paredes",{},"C:/Users/ser/Desktop/virtual_mar_entrega/deteccion_13especies/train_160921_13especies.json", "C:/Users/ser/detectron2/10especies_cuevas/images/")
paredes_metadata = MetadataCatalog.get("paredes")
dataset_dicts = DatasetCatalog.get("paredes")

cfg = get_cfg()

# load weights
cfg.merge_from_file("C:/Users/ser/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "C:/Users/ser/Desktop/virtual_mar_entrega/deteccion_13especies/2700_160921/model_final.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 

cfg.SOLVER.BASE_LR=0.02
cfg.MODEL.ROI_HEADS.NUM_CLASSES=13
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.3

# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

transectos=["IC14_CATEDRAL_0421_T1_VUELTA"]
for nombre_transecto in transectos:
    try:
        os.mkdir('./'+nombre_transecto)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
    #image
    im_path = "./"+nombre_transecto+"/"
    with open('./'+nombre_transecto+"/"+nombre_transecto+'.csv', mode='w',newline='') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(['Transecto','Imagen','Especie','area(m2)','Unidades de la especie','Cobertura(%)','Cobertura(m2)'])
        #employee_writer.writerow(['Transect', 'Photo', 'Agelas oroides','Didenmnum sp.','Froindipora verrucosa','Myriapora truncata','Reteporella sp','Spirastrella cunctatrix',"Parazoanthus axinellae","schizoretepora serratima","axinella sp", "SP025C_Acanthella acuta","SP040B_Salmancina incrustans","SP019B_Demospongia sp.",'Area','Dens_Agelas_oroides','Dens_Didemnum','Dens_Froindipora_verrucosa','Dens_Myriapora_truncata','Dens_Reteporella','Dens_Spirastrella cunctatrix','Dens_Parazoanthus axinellae','Dens_schizoretepora_serratima','Dens_axinella','Dens_SP025C_Acanthella acuta','Dens_SP040B_Salmancina_incrustans','Dens_SP019B_Demospongia'])
        for imagen in glob.glob(im_path+"*.jpg"):
            nombre=imagen.split("/"+nombre_transecto+"\\")
            nombre=nombre[1]
            print(nombre)
            im=cv2.imread(imagen)
            
            #inference
            outputs = predictor(im)
            print(outputs)
            csv_imagenes_path='./'+nombre_transecto+'/'
            mask_array = outputs["instances"].pred_masks.cpu().numpy() 
                        
            # Create point matrix get coordinates of mouse click on image
            point_matrix = np.zeros((2,2))
            
            # Showing original image
            cv2.imshow("Original Image ", im)
            
            # Mouse click event on original image
            cv2.setMouseCallback("Original Image ", mousePoints)
            cv2.waitKey()
            counter=0
            
            # Printing updated point matrix
            punto1=point_matrix[0]
            punto2=point_matrix[1]

            punto1_x=punto1[0]
            punto1_y=punto1[1]

            punto2_x=punto2[0]
            punto2_y=punto2[1]

            x=punto1_x-punto2_x
            y=punto1_y-punto2_y
            print(x,y)
            
            distancia_px=math.sqrt(pow(x,2)+pow(y,2))
            print(distancia_px)
            
            height, width = im.shape[0:2]

            distancia_puntos_real=0.2#m
                             
            instances=outputs.get('instances')
            classes_tensor=instances.pred_classes
            classes_tensor=classes_tensor.cpu()
            classes=classes_tensor.numpy()
                     
            if distancia_px==0:
                area=0
                
            else:
                area=(height*distancia_puntos_real/distancia_px)*(width*distancia_puntos_real/distancia_px)
                area=round(area,2)
            
            cropper_varios(imagen, mask_array,nombre,csv_imagenes_path,area,nombre_transecto,employee_writer)
            v = Visualizer(im[:, :, ::-1],
                               metadata=paredes_metadata, 
                               scale=1, 
                               instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
                )
            
            v,imagen,nuevo_negro = v.draw_instance_predictions(outputs["instances"].to("cpu"))
               
        
            v.save("./"+nombre_transecto+"/"+nombre)