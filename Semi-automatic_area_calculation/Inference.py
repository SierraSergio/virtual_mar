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
cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "./model_final.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 

cfg.SOLVER.BASE_LR=0.02
cfg.MODEL.ROI_HEADS.NUM_CLASSES=13
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.3

# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
import numpy as np
import cv2
import csv
import math
import os

mouse_click_counter = 0
point_matrix = np.zeros((100, 2))

def mouse_points_handler(event, x, y, flags, params):
    global mouse_click_counter, point_matrix
    # Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[mouse_click_counter] = x, y
        mouse_click_counter += 1

def get_laser_coordinates(img_real, a, b, col_real, row_real, laseres_x, laseres_y):
    pixel_real = img_real[a, b]
    if pixel_real[0] > 245 and pixel_real[1] == 255 and pixel_real[2] > 240:
        laseres_x.append(b)
        laseres_y.append(a)
    return laseres_x, laseres_y
    
def crop_multiple(org_image_path, mask_array, name, csv_images_path, area, transect, employee_writer):
    print(mask_array.shape[0])
    img = cv2.imread(org_image_path)
    height, width = img.shape[:2]
    total_pixels = height * width

    joint_area = 0
    demos_count = salma_count = acant_count = axinella_count = schi_count = parazo_count = spira_count = rete_count = myri_count = frondi_count = dide_count = agelas_count = 0
    demos_area = salma_area = acant_area = axinella_area = schi_area = parazo_area = spira_area = rete_area = myri_area = frondi_area = dide_area = agelas_area = 0

    for i in range(mask_array.shape[0]):
        mask = np.moveaxis(mask_array, 0, -1)
        mask = mask[:,:,i:i+1]
        
        instances = outputs.get('instances')
        classes_tensor = instances.pred_classes
        classes_tensor = classes_tensor.cpu()
        classes = classes_tensor.numpy()
        category = classes[i]

        output = np.where(mask == False, 0, (np.where(mask == True, 255, img)))
        image = Image.fromarray(output)

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        non_zero_pixels = cv2.countNonZero(gray)
        area_covered = non_zero_pixels / total_pixels
        joint_area += non_zero_pixels
        
        if category == 0:
            # Agelas oroides
            agelas_count += 1
            agelas_area += area_covered * 100

        if category == 1:
            # Froindipora verrucosa
            frondi_count += 1
            frondi_area += area_covered * 100

        if category == 2:
            # Myriapora truncata
            myri_count += 1
            myri_area += area_covered * 100

        if category == 3:
            # Reteporella sp
            rete_count += 1
            rete_area += area_covered * 100

        if category == 4:
            # Spirastrella cunctatrix
            spira_count += 1
            spira_area += area_covered * 100

        if category == 5:
            # Parazoanthus axinellae
            parazo_count += 1
            parazo_area += area_covered * 100

        if category == 6:
            # schizoretepora serratima
            schi_count += 1
            schi_area += area_covered * 100

        if category == 7:
            # axinella sp
            axinella_count += 1
            axinella_area += area_covered * 100

        if category == 8:
            # SP025C_Acanthella acuta
            acant_count += 1
            acant_area += area_covered * 100

    if agelas_count > 0:
        employee_writer.writerow([transect, name, 'Agelas oroides', area, agelas_count, agelas_area, round(area * agelas_area / 100, 2)])
    if frondi_count > 0:
        employee_writer.writerow([transect, name, 'Froindipora verrucosa', area, frondi_count, frondi_area, round(area * frondi_area / 100, 2)])
    if myri_count > 0:
        employee_writer.writerow([transect, name, 'Myriapora truncata', area, myri_count, myri_area, round(area * myri_area / 100, 2)])
    if rete_count > 0:
        employee_writer.writerow([transect, name, 'Reteporella sp', area, rete_count, rete_area, round(area * rete_area / 100, 2)])
    if spira_count > 0:
        employee_writer.writerow([transect, name, 'Spirastrella cunctatrix', area, spira_count, spira_area, round(area * spira_area / 100, 2)])
    if parazo_count > 0:
        employee_writer.writerow([transect, name, 'Parazoanthus axinellae', area, parazoi_count, parazo_area, round(area * parazo_area / 100, 2)])
    if schi_count > 0:
        employee_writer.writerow([transect, name, 'Schizoretepora serratima', area, schi_count, schi_area, round(area * schi_area / 100, 2)])
    if axinella_count > 0:
        employee_writer.writerow([transect, name, 'axinella sp', area, axinella_count, axinella_area, round(area * axinella_area / 100, 2)])
    if acant_count > 0:
        employee_writer.writerow([transect, name, 'Acanthella acuta', area, acant_count, acant_area, round(area * acant_area / 100, 2)])
    
    area_no_etiquetada = 100 - ((joint_area / total_pixels) * 100)
    employee_writer.writerow([transecto,nombre,"No identify",area,"1",area_no_etiquetada,round(area*area_no_etiquetada/100,2)])

def create_directory(directory):
    try:
        os.mkdir(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
#register
register_coco_instances("paredes",{},"C:/Users/ser/Desktop/virtual_mar_entrega/deteccion_13especies/train_160921_13especies.json", "C:/Users/ser/detectron2/10especies_cuevas/images/")
paredes_metadata = MetadataCatalog.get("paredes")
dataset_dicts = DatasetCatalog.get("paredes")

cfg = get_cfg()

# load weights
cfg.merge_from_file("./mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "./model_final.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 

cfg.SOLVER.BASE_LR=0.02
cfg.MODEL.ROI_HEADS.NUM_CLASSES=13
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.3

# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

transects=["IC14_CATEDRAL_0421_T1_VUELTA"]

for transect_name in transects:
    # Create directory for transect
    create_directory('./'+transect_name)

    # Image
    im_path = "./"+transect_name+"/"
    with open('./'+transect_name+"/"+transect_name+'.csv', mode='w',newline='') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(['Transect','Image','Species','Area (m2)','Units of species','Coverage (%)','Coverage (m2)'])

        for image_path in glob.glob(im_path+"*.jpg"):
            image_name = image_path.split("/"+transect_name+"\\")[1]
            image = cv2.imread(image_path)

            # Inference
            predictor = DefaultPredictor(cfg)
            outputs = predictor(image)
            print(outputs)

            csv_image_path='./'+transect_name+'/'
            mask_array = outputs["instances"].pred_masks.cpu().numpy()

            # Create point matrix get coordinates of mouse click on image
            point_matrix = np.zeros((2,2))

            # Showing original image
            cv2.imshow("Original Image", image)

            # Mouse click event on original image
            cv2.setMouseCallback("Original Image", mousePoints)
            cv2.waitKey()
            counter=0

            # Printing updated point matrix
            point1=point_matrix[0]
            point2=point_matrix[1]

            point1_x=point1[0]
            point1_y=point1[1]

            point2_x=point2[0]
            point2_y=point2[1]

            x=point1_x-point2_x
            y=point1_y-point2_y

            distance_px=math.sqrt(pow(x,2)+pow(y,2))

            height, width = image.shape[0:2]

            real_distance = 0.2 # m

            instances=outputs.get('instances')
            classes_tensor=instances.pred_classes
            classes_tensor=classes_tensor.cpu()
            classes=classes_tensor.numpy()

            if distancia_px == 0:
                area = 0
            else:
                area = (height * real_distance / distance_px) * (width *real_distance / distance_px)
                area = round(area, 2)

            cropper_varios(image_path, mask_array, image_name, csv_image_path, area, transect_name, employee_writer)

            v = Visualizer(image[:, :, ::-1],
                           metadata=metadata, 
                           scale=1, 
                           instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
                )

            v, image, nuevo_negro = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            v.save("./"+transect_name+"/"+image_name)
