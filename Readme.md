#Description
This code can simulate soil loss for any watershed based on RUSLE model. Using the specified inputs soil loss can be estimated.
   
RUSLE equation is Revised Universal Soil Loss Equation
   		 A = R*K*LS*C*P
 where A = Estimated Soil Loss (tonnes/ha/year)
       R = Rainfall Erosivity factor (MJ-mm/ha/hr/yr)
       K = Soil Erodibility factor (tonnes-hr/MJ/mm)
       LS = Topographic factor considering slope length and slope steepness
       C = Vegetative Cover Factor
       P = Conservation Practice Factor

#Running the project
This project uses various inputs: 
Geojason file of the watershed boundary
Spatial distribution of Monthly precipitation
SSURGO data for percent sand, clay, silt and organic matter of the top soil
DEM of the watershed
Sentinel images of the watershed to calculate NDVI
Land use land cover map of the watershed

The sources to the input file are mentioned in the python file at the place of their use.
Most inputs have been added in the input folder. Below inputs will need to be downloaded in order to succesfully run the project: 
For running the code you need to download the DEM of the watershed using the link https://earthexplorer.usgs.gov. The files used in the code are mentioned as DEM1, DEM2 and DEM3 which covers the whole watershed and merged using the python code.
Also you need to download the land use and land cover tiff file using the link https://www.mrlc.gov/viewer/ for the watershed which is named as land_use