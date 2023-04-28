
def cropper_varios(org_image_path, mask_array,nombre,csv_imagenes_path,area,transecto,employee_writer):
        print(mask_array.shape[0])
        img = cv2.imread(org_image_path)
        height, width = img.shape[0:2]
        pixeles_totales=height*width

        area_conjunta=0
        area_demos=area_salma=area_acant=area_axinella=area_schi=area_parazo=area_spira=area_rete=area_myri=area_frondi=area_dide=area_agelas=area_escuadra=0
        cont_demos=cont_salma=cont_acant=cont_axinella=cont_schi=cont_parazo=cont_spira=cont_rete=cont_myri=cont_frondi=cont_dide=cont_agelas=cont_escuadra=0
        for i in range(mask_array.shape[0]):
            print("entro al if1")
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
               'Froindipora verrucosa'
               cont_frondi=cont_frondi+1
               area_frondi=area_frondi+area_abarcada*100
              
            if clase==2:
               'Myriapora truncata'
               cont_myri=cont_myri+1
               area_myri=area_myri+(area_abarcada*100)
               
            if clase==3:
               'Reteporella sp'
               cont_rete=cont_rete+1
               area_rete=area_rete+(area_abarcada*100)
               
            if clase==4:
               'Spirastrella cunctatrix'
               cont_spira=cont_spira+1
               area_spira=area_spira+(area_abarcada*100)

            if clase==5:
               "Parazoanthus axinellae" 
               cont_parazo=cont_parazo+1
               area_parazo=area_parazo+(area_abarcada *100)                   
                               
            if clase==6:
               "schizoretepora serratima"
               cont_schi=cont_schi+1
               area_schi=area_schi+(area_abarcada*100)
               
            if clase==7:
               "axinella sp"
               cont_axinella=cont_axinella+1
               area_axinella=area_axinella+(area_abarcada*100)
              
            if clase==8:
               "SP025C_Acanthella acuta"
               cont_acant=cont_acant+1
               area_acant=area_acant+(area_abarcada*100)
               
            if clase==9:
               "SP040B_Salmancina incrustans"
               cont_salma=cont_salma+1
               area_salma=area_salma+(area_abarcada*100)
               
            if clase==10:
               "SP019B_Demospongia sp."
               cont_demos=cont_demos+1
               area_demos=area_demos+(area_abarcada*100)
               
        
        if cont_agelas>0:
            employee_writer.writerow([transecto,nombre,'Agelas oroides',area,cont_agelas,area_agelas,round(area*area_agelas/100,2)])
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
        area_no_etiquetada=100-(area_demos+area_salma+area_acant+area_axinella+area_schi+area_parazo+area_spira+area_rete+area_myri+area_frondi+area_dide+area_agelas+area_escuadra)
        employee_writer.writerow([transecto,nombre,"No identificado",area,"1",area_no_etiquetada,round(area*area_no_etiquetada/100,2)])