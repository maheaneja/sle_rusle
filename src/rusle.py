#!/usr/bin/env python
# coding: utf-8

"""
Created on Wed Aug 21 16:22:23 2024

@author: mahekpreet@ksu.edu
"""

"""
   Soil Loss estimation using RUSLE equation
   RUSLE equation is Revised Universal Soil Loss Equation
            A = R*K*LS*C*P
         where A = Estimated Soil Loss (tonnes/ha/year)
               R = Rainfall Erosivity factor (MJ-mm/ha/hr/yr)
               K = Soil Erodibility factor (tonnes-hr/MJ/mm)
               LS = Topographic factor considering slope length and slope steepness
               C = Vegetative Cover Factor
               P = Conservation Practice Factor
"""

# # Import modules

!pip install geopandas
!pip install xarray
!pip install rasterio
!pip install rioxarray
!pip install whitebox


import geopandas as gpd
import rioxarray
import numpy as np
import xarray as xr
from shapely.geometry import polygon
import matplotlib.pyplot as plt
from matplotlib import colors
from rioxarray.merge import merge_arrays
import rasterio

# Import module for hydrological tools
import whitebox 
import os


# # Import the files required

# In[6]:
# please change to relevant working directory
os.chdir("/Users/user_name/RUSLE/")

# Loading the watershed file for lower big blue
# The shapefile of big blue was downloaded from National Map Viewer for HUC 10(https://apps.nationalmap.gov/downloader/)
big_blue = gpd.read_file('input/big_blue.geojson')

#Read the raster file for rainfall data
# The data is downloaded from NCEP NOAA (https://ftp.cpc.ncep.noaa.gov/GIS/USDM_Products/precip/total/monthly/)
raster_array_jan = xr.open_dataarray('input/p.full.202301.tif')
raster_array_feb = xr.open_dataarray('input/p.full.202302.tif')
raster_array_march = xr.open_dataarray('input/p.full.202303.tif')
raster_array_april = xr.open_dataarray('input/p.full.202304.tif')
raster_array_may = xr.open_dataarray('input/p.full.202305.tif')
raster_array_june = xr.open_dataarray('input/p.full.202306.tif')
raster_array_july = xr.open_dataarray('input/p.full.202307.tif')
raster_array_aug = xr.open_dataarray('input/p.full.202308.tif')
raster_array_sept = xr.open_dataarray('input/p.full.202309.tif')
raster_array_oct = xr.open_dataarray('input/p.full.202310.tif')
raster_array_nov = xr.open_dataarray('input/p.full.202311.tif')
raster_array_dec = xr.open_dataarray('input/p.full.202312.tif')

# Import data files for the soil properties
# This data is taken from the SSURGO data
sand_lbb = xr.open_dataarray('input/Percent_sand_BBL.tif')
silt_lbb = xr.open_dataarray('input/Percent_silt_BBL.tif')
clay_lbb = xr.open_dataarray('input/Percent_clay_BBL.tif')
org_matt_lbb = xr.open_dataarray('input/Percent_Organic_mattter_BBL.tif')

# Adding DEM ie elevation data
# DEM is downloaded from USGS earth explorer
DEM_1 = rioxarray.open_rasterio("input/DEM_1.tif")
DEM_2 = rioxarray.open_rasterio("input/DEM_2.tif")
DEM_3 = rioxarray.open_rasterio("input/DEM_3.tif")
DEM_merged = merge_arrays([DEM_1, DEM_2, DEM_3])

#adding the sentinel images for NDVIcalculation
# Images are downloaded from Sentinel database (https://apps.sentinel-hub.com/eo-browser/?zoom=9&lat=39.39694&lng=-97.86896&themeId=DEFAULT-THEME)
image_red = xr.open_dataarray("input/2023-12-10_Sentinel-2_L2A_B04.tiff")
image_IR = xr.open_dataarray("input/2023-12-10_Sentinel-2_L2A_B08.tiff")

#Adding land use land cover for P factor calculation
#Land use land cover data is downloaded from https://www.mrlc.gov/viewer/
land_use = xr.open_dataarray("input/NLCD_2021_Land_Cover.tiff")


# In[7]:


#Reading the shapefile of watershed
print(big_blue.head())
print(big_blue.crs)


# In[8]:


#Visualizing the lower big blue watershed from HUC8
tuttle_creek = big_blue.loc[[0], 'geometry']
big_blue['geometry'][0]


# In[9]:


# Viewing the Lower big blue
plt.figure(figsize = (5,5))
tuttle_creek.plot(facecolor='purple', edgecolor = 'k')
plt.title('Lower Big Blue Watershed')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

"""
     R FACTOR (Rainfall Erosivity Factor)
     It is the ability of the rain to cause erosion, measured in MJ-mm/ha-hr-yr
     Formula: R = -823.8 + 5.213 P
                 where, P = annual rainfall (mm) (Fernadez et al, 2003 originally developed by USDA_ARS in 2002 and used for USA))
"""

# In[11]:


#Finding the annual rainfall
raster_array_annual = (raster_array_jan + raster_array_feb + raster_array_march + raster_array_april +
                       raster_array_may + raster_array_june + raster_array_july + raster_array_aug +
                       raster_array_sept + raster_array_oct + raster_array_nov + raster_array_dec)
#Defining a R factor function 
def R_factor_fn(annual_rainfall, crs, x_size, y_size):
    # Convert negative values to nan
    idx_missing = annual_rainfall.data < 0
    annual_rainfall.data[idx_missing] = np.nan
    #Calculation of R Factor
    R_factor_values = (-823.8 + (5.213 * annual_rainfall))
    R_factor = R_factor_values.rio.reproject(dst_crs= crs, shape=(x_size,y_size))
    return R_factor


# In[12]:


#Calculation of R Factor
R_factor = R_factor_fn(raster_array_annual, 'EPSG:4326', 5000,5000)
R_factor


# In[13]:


#To check the projection of raster file
R_factor.rio.crs


# In[14]:


#To visualize the Rainfall and R factor
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
raster_array_annual.plot(cmap='Blues', vmin=100, vmax=1000)
plt.title('Annual Rainfall (mm)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
 

plt.subplot(1,2,2)
R_factor.plot(cmap='Blues', vmin=100, vmax=4000)
plt.title('R Factor (MJ-mm/ha-hr-yr)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[15]:


#Define function for cliping raster file to tuttle creek
clip_fn = lambda polygon, R: R.rio.clip( [polygon.geometry], 
                                        crs = R.rio.crs,
                                       all_touched = True)


# In[16]:


#Clip the R_factor to Lower big blue 
big_blue['clipped_R_factor'] = big_blue.apply(lambda row: clip_fn(row,R_factor), axis =1)
big_blue.head(3)


# In[17]:


# Removing the no data values created due to reprojection from the array
idx_R_factor = big_blue['clipped_R_factor'][0].data >5000
big_blue['clipped_R_factor'][0].data[idx_R_factor] = np.nan


# In[18]:


# Plotting the R factor for Tuttle creek
fig,ax = plt.subplots(figsize = (10,8))
big_blue.loc[ [0], 'geometry'].boundary.plot(ax=ax, edgecolor = 'k')
big_blue.loc[0, 'clipped_R_factor'].plot(ax=ax)
ax.set_title('Lower big blue R Factor')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

"""
     K FACTOR (Soil Erodibility Factor)
     Soil erodibility is defined as the susceptibility of soil particles to detachment and transport by raindrop impact and runoff
     It is measured in (tonnes-ha-hr/MJ-ha-mm) and depends on percentage of sand, silt, clay and organic matter present in soil
     Formula: K = 0.1317 * (0.2 + (0.3* 𝑒𝑥𝑝(0.0256 ∗ 𝑆𝑎𝑛𝑑 (1 −(𝑠𝑖𝑙𝑡/100)))) ∗ ((𝑠𝑖𝑙𝑡/(𝐶𝑙𝑎𝑦+𝑆𝑖𝑙𝑡))^0.3 ∗ 
                 (1 − ((0.25*organic_matter)/(organic_matter+ exp(3.72−2.95*organic_matter)))) ∗ (1 −(0.7*𝑆𝑛₁/(𝑆𝑛₁+ exp(−5.51+22.9𝑆𝑛₁))
                 where Sn1 = 1 - (sand/100)
                (Williams J.R and Renard K.G, 1983)
"""
# In[20]:


# Analysing the size of different raster datasets
# Data is downloaded from SSURGO Database for Kansas State
print(sand_lbb.shape)
print(silt_lbb.shape)
print(clay_lbb.shape)
print(org_matt_lbb.shape)


# In[21]:


# Defining funtion for K Factor
def K_factor_fn(sand, silt, clay, org_matt, crs, x_size, y_size):
    Sn = 1-(sand/100)
    K_factor_values = 0.1317 * (0.2 + ((0.3*np.exp(0.0256*sand* (1-(silt/100)))) * ((silt/(silt+clay))**0.3) * (1 -((0.25*org_matt)/(org_matt+np.exp(3.72 - 2.95*org_matt)))) * (1 -((0.7*Sn)/(Sn+np.exp(-5.51+ 22.9*Sn))))))
    K_factor = K_factor_values.rio.reproject(dst_crs= crs, shape=(x_size,y_size))
    return K_factor


# In[22]:


#Calculating the K Factor for the lower big blue watershed
K_factor = K_factor_fn(sand_lbb, silt_lbb, clay_lbb, org_matt_lbb, 'EPSG:4326', 3000, 3000)
K_factor


# In[23]:


# Clipping the K Factor
big_blue['clipped_K_factor'] = big_blue.apply(lambda row: clip_fn(row,K_factor), axis =1)
big_blue.head(3)


# In[24]:


# Removing the no data values created due to reprojection from the array
idx_K_factor = big_blue['clipped_K_factor'][0].data >1
big_blue['clipped_K_factor'][0].data[idx_K_factor] = np.nan


# In[25]:


# Plotting the K factor for Tuttle creek
fig,ax = plt.subplots(figsize = (10,8))
big_blue.loc[ [0], 'geometry'].boundary.plot(ax=ax, edgecolor = 'k')
big_blue.loc[0, 'clipped_K_factor'].plot(ax=ax)
ax.set_title('Lower Big Blue K Factor')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

"""
     LS FACTOR (Topographic Factor)
     The topographic factor (LS) is the ratio of soil loss per unit area from a field to the loss from a field of 22.13 m length and a uniform slope of 9%, other things being constant.
     Formula: LS = (1+m) * (((FA*Resolution)/22.1)^m) * (((Sin(Slope*0.01745))/0.09)^n)  (Moore and Burch, 1985)
     where = m and n are adjustable values m = 0.6, n=1.4    
"""
# In[27]:


# Reading the Digital Elevation for the area
print(DEM_merged)
idx_DEM = DEM_merged.data <0
DEM_merged.data[idx_DEM] = np.nan
DEM = DEM_merged.rio.reproject(dst_crs= 'EPSG:4326', shape=(3000,3000))
DEM


# In[28]:


#Saving the merged DEM as tif file
from rasterio.transform import from_origin
data = DEM.values.squeeze()
x_res = (DEM['x'].values[1] - DEM['x'].values[0]) 
y_res = (DEM['y'].values[0] - DEM['y'].values[1]) 
transform = from_origin(DEM['x'].min(), DEM['y'].max(), x_res, y_res)
with rasterio.open('DEM.tif', 'w', driver= 'GTiff', height= data.shape[0], width= data.shape[1],
                    count= 1, dtype= data.dtype, crs= 'EPSG:4326', transform = transform ) as dst:
    dst.write(data,1)  # The `1` refers to the first band


# In[29]:


DEM_new = xr.open_dataarray('DEM.tif')
plt.figure(figsize=(8,6))
DEM_new.plot(cmap='terrain')
plt.title('Digital Elevation Model')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[30]:


#Visualizing the DEM of the area
plt.figure(figsize=(8,6))
DEM.plot(cmap='terrain')
plt.title('Digital Elevation Model')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[31]:


# Calculating slope of the region
def calculate_slope(elevation_array, cell_size, crs, x_size, y_size):
    # Calculate slope using central difference method
    dz_dx = np.gradient(elevation_array, axis=1) / cell_size
    dz_dy = np.gradient(elevation_array, axis=0) / cell_size
    
    # Calculate slope magnitude
    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_xarray = xr.DataArray(slope, dims= ('y','x'), coords = {'x': elevation_array['x'], 'y': elevation_array['y']})
    slope_xarray.rio.write_crs(crs, inplace=True)
    DEM_slope = slope_xarray.rio.reproject(dst_crs=crs, shape=(x_size, y_size))
    return DEM_slope


# In[32]:


#Calculation Slope for the Lower Big Blue
DEM_slope = calculate_slope(DEM[0,:,:], 30, 'EPSG:4326', 3000, 3000)
DEM_slope


# In[33]:


# Creatingn the flow accumulation function
wbt = whitebox.WhiteboxTools()
path = os.getcwd()
wbt.set_working_dir(path)
def flow_accumalation(DEM_raster, output_path):
    fill_raster = wbt.fill_depressions(DEM_raster, output= "fil.tif")
    flow_acc = wbt.fd8_flow_accumulation("fil.tif", "flow_acc.tif")
    return flow_acc


# In[34]:


# Calculating the flow accumulation from DEM
flow_acc = flow_accumalation('DEM.tif', path)


# In[35]:


# Reading the flow accumulation file and check the projection
flow_acc_raster = xr.open_dataarray("flow_acc.tif")
print(flow_acc_raster.rio.crs)
flow_acc_raster


# In[36]:


#Function for the LS FACTOR
def LS_factor (Res, Flow_accu, slope, crs, x_size, y_size):
    LS_raster = 1.6 * (((Flow_accu*Resolution)/22.13)**0.6) * (((np.sin((np.degrees(np.arctan(slope)))*0.01745))/0.09)**1.4)
    LS_factor = LS_raster.rio.reproject(dst_crs= crs, shape=(x_size,y_size))
    return LS_factor


# In[37]:


#Calculating
Resolution = 30
LS_factor_BBL = LS_factor(Resolution, flow_acc_raster, DEM_slope.values, 'EPSG:4326', 3000, 3000)
LS_factor_BBL


# In[38]:


#Visualizing the slope, flow accumulation and LS Factor
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
DEM_slope.plot(cmap='YlOrBr', vmax=0.15)
plt.title('Slope Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(1,3,2)
flow_acc_raster.plot(cmap='YlOrBr', vmax=10)
plt.title('Flow Accumalation')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(1,3,3)
LS_factor_BBL.plot(cmap='YlOrBr', vmax=0.5)
plt.title('LS Factor')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()


# In[39]:


#Clip the LS factor to tuttle creek
big_blue['clipped_LS_factor'] = big_blue.apply(lambda row: clip_fn(row,LS_factor_BBL), axis =1)
big_blue.head(3)


# In[40]:


# LS Factor for the Tuttle creek
fig,ax = plt.subplots(figsize = (10,8))
big_blue.loc[ [0], 'geometry'].boundary.plot(ax=ax, edgecolor = 'k')
big_blue.loc[0, 'clipped_LS_factor'].plot(ax=ax, vmax=1)
ax.set_title('Big Blue LS Factor')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

"""
    C FACTOR (Vegetative Cover Factor)
     The crop management factor is the ratio of soil loss from cultivated land under certain conditions to soil loss from pure fallow on the same soil and slope under the same rainfall conditions.
     This factor is claculated using the NDVI (Normalized Difference Vegetation Index)
     Formula: C = exp(−2.5(𝑁𝐷𝑉𝐼/(1−𝑁𝐷𝑉𝐼))   (Tamene and Le, 2015; Van der Knijff et al., 2000)
"""
# In[42]:


#Function to calculate C factor
def C_factor(IR_image, red_image, crs, x_size, y_size):
    ndvi = ((IR_image - red_image)/(IR_image + red_image))
    C_fact = np.exp(-2.5*(ndvi/(1-ndvi)))
    C_factor = C_fact.rio.reproject(dst_crs= crs, shape=(x_size,y_size))
    return C_factor


# In[43]:


#Calculate the C factor
red = image_red[0,:,:]
IR = image_IR[0,:,:]
C_factor_BBL = C_factor( IR, red, 'EPSG:4326', 1000, 1000)
C_factor_BBL


# In[44]:


#Define Custom colormap
hex_palette = ['#CE7E45', '#DF923D', '#F1B555', '#FCD163', '#99B718', '#74A901','#66A000', 
               '#529400', '#3E8601', '#207401', '#056201', '#004C00', '#023B01','#012E01', '#011D01', '#011301']
NDVI_cmap = colors.ListedColormap(hex_palette)
NDVI_cmap.set_under('#0000FF')

#Registering the new colormap
plt.colormaps.register(cmap = NDVI_cmap, name ='NDVI')
NDVI_cmap


# In[45]:


#Visualize the C Factor
plt.figure(figsize= (15,5))
C_factor_BBL.plot(cmap='Spectral', vmax=1)
plt.title('C Factor')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()


# In[46]:


#Clip the LS factor to turtle creek
big_blue['clipped_C_factor'] = big_blue.apply(lambda row: clip_fn(row, C_factor_BBL), axis =1)
big_blue.head(3)


# In[47]:


# Removing the nodata values
idx_C_factor = big_blue['clipped_C_factor'][0].data >1
big_blue['clipped_C_factor'][0].data[idx_C_factor] = np.nan


# In[48]:


fig,ax = plt.subplots(figsize = (10,8))
big_blue.loc[ [0], 'geometry'].boundary.plot(ax=ax, edgecolor = 'k')
big_blue.loc[0, 'clipped_C_factor'].plot(ax=ax)
ax.set_title('Big Blue C Factor')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

"""
   P Factor
   The support practice factor (P) is defined as the ratio of soil loss under a specific soil conservation practice (e.g. contouring, terracing) to that of a field with upslope and downslope tillage (Renard et al., 1997). The P factor accounts for management practices that affect soil erosion through modifying the flow pattern, such as contouring, strip cropping, or terracing (Renard et al., 1997). The estimation of the p value in this study is realized by assigning values to land use types. 
       Land Use Types:
           #### Cropland: 0.35
           #### Forest: 1.0
           #### Grassland: 1.0
           #### Water: 0.0 
           #### Settlement Place: 0.0
           #### Unused Land:  0.0
"""
# In[60]:


# reclassify the land use values
def P_factor(land_use, crs, x_size, y_size):
    def reclassify(value):
        if value >= 0 and value < 35:
            return 0
        elif value >= 35 and value < 82:
            return 1
        elif value >= 82 and value < 88:
            return 0.35
        else:
            return 0
    reclassify_vectorized = np.vectorize(reclassify)
    reclassified_data = reclassify_vectorized(land_use)
    data = reclassified_data[0,:,:]
    x_resl = (land_use['x'].values[1] - land_use['x'].values[0]) 
    y_resl = (land_use['y'].values[0] - land_use['y'].values[1]) 
    transform_land = from_origin(land_use['x'][0], land_use['y'][0], x_resl, y_resl)
    with rasterio.open('P_factor.tiff', 'w', driver= 'GTiff', height= land_use.shape[1], width= land_use.shape[2],
                    count= 1, crs = crs, dtype= land_use.dtype, transform = transform_land ) as dst:
        dst.write(data,1)
    P_factor_values = xr.open_dataarray("P_factor.tiff")
    P_factor = P_factor_values.rio.reproject(dst_crs= crs, shape=(x_size,y_size))
    return P_factor


# In[56]:


land_use_BBL = land_use.rio.reproject('epsg:4326')
land_use_BBL


# In[62]:


P_factor_BBL = P_factor(land_use_BBL, 'EPSG:4326', 3000, 3000)
P_factor_BBL


# In[54]:


plt.figure(figsize=(8,6))
P_factor_BBL.plot(cmap='terrain')
plt.title('Big Blue P Factor')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[55]:


#Clip the P factor to turtle creek
big_blue['clipped_P_factor'] = big_blue.apply(lambda row: clip_fn(row, P_factor_BBL), axis =1)
big_blue.head(3)


# In[ ]:


fig,ax = plt.subplots(figsize = (10,8))
big_blue.loc[ [0], 'geometry'].boundary.plot(ax=ax, edgecolor = 'k')
big_blue.loc[0, 'clipped_P_factor'].plot(ax=ax, cmap='terrain')
ax.set_title('Big Blue P Factor')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# # Soil Loss Estimation (RUSLE Equation)
#      A =R*K*LS*C*P

# In[ ]:


#Reproject all the layers according K Factor raster layer
R_factor_reproject = big_blue['clipped_R_factor'][0].rio.reproject_match(big_blue['clipped_K_factor'][0])
LS_factor_reproject = big_blue['clipped_LS_factor'][0].rio.reproject_match(big_blue['clipped_K_factor'][0])
C_factor_reproject = big_blue['clipped_C_factor'][0].rio.reproject_match(big_blue['clipped_K_factor'][0])
P_factor_reproject = big_blue['clipped_P_factor'][0].rio.reproject_match(big_blue['clipped_K_factor'][0])


# In[ ]:


#Estimation of soil loss for Tuttle Creek
Soil_loss = (R_factor_reproject * big_blue['clipped_K_factor'][0] * LS_factor_reproject * C_factor_reproject * P_factor_reproject)
Soil_loss


# In[ ]:


#Ploting the soil loss from tuttle creek
fig,ax = plt.subplots(figsize = (10,8))
big_blue.loc[ [0], 'geometry'].boundary.plot(ax=ax, edgecolor = 'k')
Soil_loss.plot(ax=ax, vmax=30)
ax.set_title('Soil Loss Estimation (tonnes/ha/year)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[ ]:


#Mean soil Loss from Tuttle creek
Mean_soil_loss = Soil_loss.mean()

print(f'The Simulated Mean Soil Loss from Tuttle creek is {Mean_soil_loss:.2f} tonnes/ha/year')



