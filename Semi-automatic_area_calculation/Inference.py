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
    return laseres_x,laseres_y

def crop_multiple(org_image_path, mask_array, name, csv_images_path, area, transect, employee_writer):
    print(mask_array.shape[0])
    img = cv2.imread(org_image_path)
    height, width = img.shape[0:2]
    total_pixels = height * width

    joint_area = 0
    demos_area = salma_area = acant_area = axinella_area = schi_area = parazo_area = spira_area = rete_area = myri_area = frondi_area = dide_area = agelas_area = escuadra_area = 0
    demos_count = salma_count = acant_count = axinella_count = schi_count = parazo_count = spira_count = rete_count = myri_count = frondi_count = dide_count = agelas_count = escuadra_count = 0
    
    for i in range(mask_array.shape[0]):
        mask_array1 = np.moveaxis(mask_array, 0, -1)
        mask_array2 = mask_array1[:, :, i:i+1]
            
        instances = outputs.get('instances')
        classes_tensor = instances.pred_classes
        classes_tensor = classes_tensor.cpu()
        classes = classes_tensor.numpy()
        class_num = classes[i]
                        
        output = np.where(mask_array2 == False, 0, (np.where(mask_array2 == True, 255, img)))
        im = Image.fromarray(output)

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)                              
        pixels = cv2.countNonZero(gray)
        area_covered = pixels / total_pixels
           
        joint_area += pixels
        if class_num == 0:
            'Agelas oroides'
            agelas_count += 1
            agelas_area += (area_covered * 100)
  
        if class_num == 1:
            'Froindipora verrucosa'
            frondi_count += 1
            frondi_area += (area_covered * 100)
              
        if class_num == 2:
            'Myriapora truncata'
            myri_count += 1
            myri_area += (area_covered * 100)
               
        if class_num == 3:
            'Reteporella sp'
            rete_count += 1
            rete_area += (area_covered * 100)
               
        if class_num == 4:
            'Spirastrella cunctatrix'
            spira_count += 1
            spira_area += (area_covered * 100)

        if class_num == 5:
            "Parazoanthus axinellae" 
            parazo_count += 1
            parazo_area += (area_covered * 100)                   
                               
        if class_num == 6:
            "schizoretepora serratima"
            schi_count += 1
            schi_area += (area_covered * 100)
               
        if class_num == 7:
            "axinella sp"
            axinella_count += 1
            axinella_area += (area_covered * 100)
              
        if class_num == 8:
            "Acanthella acuta"
            acant_count += 1
            acant_area += (area_covered * 100)
               
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
        employee_writer.writerow([transecto,name,"No identify",area,"1",area_no_etiquetada,round(area*area_no_etiquetada/100,2)])

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

            crop_multiple(image_path, mask_array, image_name, csv_image_path, area, transect_name, employee_writer)

            v = Visualizer(image[:, :, ::-1],
                           metadata=metadata, 
                           scale=1, 
                           instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
                )

            v, image, nuevo_negro = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            v.save("./"+transect_name+"/"+image_name)
