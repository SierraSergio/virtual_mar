
def crop_multiple(org_image_path, mask_array, name, csv_images_path, area, transect, employee_writer):
    print(mask_array.shape[0])
    img = cv2.imread(org_image_path)
    height, width = img.shape[0:2]
    total_pixels = height * width

    joint_area = 0
    demos_area = salma_area = acant_area = axinella_area = schi_area = parazo_area = spira_area = rete_area = myri_area = frondi_area = dide_area = agelas_area = escuadra_area = 0
    demos_count = salma_count = acant_count = axinella_count = schi_count = parazo_count = spira_count = rete_count = myri_count = frondi_count = dide_count = agelas_count = escuadra_count = 0
    
    for i in range(mask_array.shape[0]):
        print("entering if statement 1")
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
            "SP025C_Acanthella acuta"
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
        
       
