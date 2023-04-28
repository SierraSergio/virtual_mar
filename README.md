# Virtual mar
Monitoring marine biodiversity is a challenge in some vulnerable and difficult-to-access habitats, such as underwater caves. Underwater caves are a great focus of biodiversity, concentrating a large number of species in their environment. But, most of the sessile species that live on the rocky walls are very vulnerable, and they are often threatened by different pressures. The use of these spaces as a destination for recreational divers can cause different impacts on the benthic habitat. In this work, we propose a methodology based on video recordings of cave walls and image analysis with deep learning algorithms to estimate the spatial density of structuring species in a study area. We propose a combination of automatic frame overlap detection, estimation of the actual extent of surface cover, and semantic segmentation of the main 10 species of corals and sponges to obtain species density maps. These maps can be the data source for monitoring biodiversity over time. In this paper, we analyzed the performance of three different semantic segmentation algorithms and backbones for this task and found that the Mask-RCNN model with the Xception101 backbone achieves the best accuracy, with an average segmentation accuracy of 82%.


<img src="https://user-images.githubusercontent.com/58831974/235145530-27011fd9-07cb-4f4b-9d6e-e6513b89ef74.jpg" alt="Descripción de la imagen" width="600">
Image 1
<img src=https://user-images.githubusercontent.com/58831974/235145567-e49c61f0-8c71-445d-a356-2350e0b36de7.jpg alt="Descripción de la imagen" width="600">
Image 1 inference
<img src=https://user-images.githubusercontent.com/58831974/235145536-964333a2-5c82-492f-9589-f856c456ed6e.jpg alt="Descripción de la imagen" width="600">
Image 2
<img src=https://user-images.githubusercontent.com/58831974/235145572-b647ba92-a889-483a-b362-87f6916dd026.jpg alt="Descripción de la imagen" width="600">
Image 2 inference
