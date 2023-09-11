"""

Morphometric feature extraction features for sub-compartments within segmented FTUs


"""

import numpy as np
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.color import rgb2gray, rgb2hsv

from skimage.color import rgb2hsv
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import exposure

import girder_client
from PIL import Image
import requests
from io import BytesIO
from skimage.draw import polygon
import pandas as pd
import json
import argparse
from tqdm import tqdm
import sys
import os
from matplotlib import pyplot as plt
from skimage.color import label2rgb
# create figure

import shutil
filenames = []
class FeatureExtractor:
    def __init__(self,
                 girder_api_url: str,
                 user_token: str,
                 slide_item_id: str,
                 sub_seg_params: list,
                 feature_list: list,
                 output_path: str):

        # Initializing properties of FeatureExtractor object
        self.gc = girder_client.GirderClient(apiUrl=girder_api_url)
        self.user_token = user_token
        self.sub_seg_params = sub_seg_params
        self.slide_item_id = slide_item_id
        self.feature_list = feature_list
        self.output_path = output_path

        # Making feature extract list
        self.feature_extract_list = {} 
        

        for f in self.feature_list:
            if f=='Distance Transform Features':
                self.feature_extract_list[f] = lambda comp: self.calculate_distance_transform_features(comp)
                
            elif f=='Color Features':
                self.feature_extract_list[f] = lambda image,comp: self.calculate_color_features(image,comp)

            elif f=='Texture Features':
                self.feature_extract_list[f] = lambda image,comp: self.calculate_texture_features(image,comp)
                
            elif f=='Morphological Features':
                self.feature_extract_list[f] = lambda comp: self.calculate_morphological_features(comp)
                
        # Getting the names of the sub-compartments
        self.sub_comp_names = [i['name'] for i in self.sub_seg_params]

        # Getting annotations
        self.annotations = self.gc.get(f'annotation/item/{self.slide_item_id}')
        # Iterating through annotations and extracting features
        for a_idx, ann in tqdm(enumerate(self.annotations)):
            if 'annotation' in ann:
                if not 'interstitium' in ann['annotation']['name'] and not ann['annotation']['name'] == 'Spots':

                    # Initialize annotation/compartment dictionary, keys for each feature category specified in self.feature_list
                    compartment_feature_dict = {i:[] for i in self.feature_list}

                    # Iterating through elements in annotation
                    for c_idx,comp in tqdm(enumerate(ann['annotation']['elements'])):
                        # Extract image, mask, and sub-compartment mask
                        image, mask = self.grab_image_and_mask(comp['points'])
                        sub_compartment_mask = self.sub_segment_image(image, mask)
                        
                        # Gettining rid of structures with areas less than the minimum size for each subcompartment
                        if np.sum(np.sum(sub_compartment_mask,axis=-1))>0:
                            if 'user' not in comp:
                                comp['user'] = {}

                            # Iterating through feature extraction function handles
                            for feat in self.feature_extract_list:
                                try:
                                    cat_feat = self.feature_extract_list[feat](image,sub_compartment_mask)
                                except:
                                    cat_feat = self.feature_extract_list[feat](sub_compartment_mask)

                                # Adding extracted category of features to compartment_feature_dict
                                compartment_feature_dict[feat].append(cat_feat)

                                # Adding extracted category of features to element user metadata
                                for c_f in cat_feat:
                                    comp['user'][c_f] = np.float64(cat_feat[c_f])
                                
                                self.annotations[a_idx]['annotation']['elements'][c_idx] = comp

                    # Outputting compartment features to excel file (one sheet per feature category)
                    with pd.ExcelWriter(self.output_path+'/'+f'{ann["annotation"]["name"]}_Features.xlsx') as writer:
                        filenames.append(self.output_path+'/'+f'{ann["annotation"]["name"]}_Features.xlsx')
                        for feat_cat in compartment_feature_dict:
                            feat_df = pd.DataFrame.from_records(compartment_feature_dict[feat_cat])
                            feat_df.to_excel(writer,sheet_name = feat_cat)

        # Posting updated annotations to slide
        self.post_annotations()


    def grab_image_and_mask(self,coordinates):

        coordinates = np.squeeze(np.array(coordinates))

        # Defining bounding box:
        min_x = np.min(coordinates[:,0])
        min_y = np.min(coordinates[:,1])
        max_x = np.max(coordinates[:,0])
        max_y = np.max(coordinates[:,1])

        # Getting image from bbox region
        image = np.uint8(np.array(Image.open(BytesIO(requests.get(self.gc.urlBase+f'/item/{self.slide_item_id}/tiles/region?token={self.user_token}&left={min_x}&top={min_y}&right={max_x}&bottom={max_y}').content))))

        scaled_coordinates = coordinates.tolist()
        scaled_coordinates = [[int(i[0]-min_x), int(i[1]-min_y)] for i in scaled_coordinates]

        x_coords = [i[0] for i in scaled_coordinates]
        y_coords = [i[1] for i in scaled_coordinates]

        height = int(max_y-min_y)
        width = int(max_x-min_x)
        mask = np.zeros((height,width))
        cc,rr = polygon(y_coords,x_coords,(height,width))
        mask[cc,rr] = 1

        return image, mask

    def sub_segment_image(self,image,mask):

        # Output format is one-hot encoded sub-compartment masks

        # Initializing sub-compartment mask and remaining pixels mask
        sub_comp_image = np.zeros((np.shape(image)[0],np.shape(image)[1],len(self.sub_seg_params)))
        remainder_mask = np.ones_like(mask)

        # Converting image to HSV space (using saturation channel for thresholding)
        hsv_image = np.uint8(255*rgb2hsv(image)[:,:,1])

        # Applying histogram equalization
        hsv_image = np.uint8(255*exposure.equalize_hist(hsv_image))
        
        # Iterating through sub-compartment parameters
        for idx, param in enumerate(self.sub_seg_params):

            remainder_mask = np.multiply(hsv_image,remainder_mask)
            masked_remaining_pixels = np.multiply(remainder_mask,mask)

            # Applying manually set threshold
            masked_remaining_pixels[masked_remaining_pixels<=param['threshold']] = 0
            masked_remaining_pixels[masked_remaining_pixels>0] = 1

            # Check if the current sub-compartment is nuclei (additional special processing)
            if param['name'].lower()=='nuclei':

                # Filling holes
                sub_mask = remove_small_holes(masked_remaining_pixels>0,area_threshold=10)
                sub_mask = sub_mask>0
                # Watershed implementation from: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
                distance = ndi.distance_transform_edt(sub_mask)
                labeled_mask, _ = ndi.label(sub_mask)
                coords = peak_local_max(distance,footprint=np.ones((3,3)),labels = labeled_mask)
                watershed_mask = np.zeros(distance.shape,dtype=bool)
                watershed_mask[tuple(coords.T)] = True
                markers, _ = ndi.label(watershed_mask)
                sub_mask = watershed(-distance,markers,mask=sub_mask)
                sub_mask = sub_mask>0

                # Filtering out small objects again
                sub_mask = remove_small_objects(sub_mask,param['min_size'])
            else:

                # Filtering by minimum size
                small_object_filtered = (1/255)*np.uint8(remove_small_objects(masked_remaining_pixels>0,param['min_size']))
                sub_mask = small_object_filtered

            sub_comp_image[sub_mask>0,idx] = 1
            remainder_mask -= sub_mask>0
        
        # Assigning remaining pixels within the boundary mask to the last sub-compartment
        remaining_pixels = np.multiply(mask,remainder_mask)
        sub_comp_image[remaining_pixels>0,idx] = 1

        return sub_comp_image

    def calculate_distance_transform_features(self,subcompartment_mask):
        # Function to calculate distance transform features for each compartment
        feature_values = {}

        # For scaling by compartment area
        object_mask = np.sum(subcompartment_mask,axis=-1)
        object_area = np.sum(object_mask)

        for sc in range(len(self.sub_comp_names)):  # As there are 3 compartments
            compartment_binary_mask = (subcompartment_mask[:,:,sc]).astype(np.uint8)
            sub_compartment_area = np.sum(compartment_binary_mask>0)

            distance_transform = cv2.distanceTransform(compartment_binary_mask, cv2.DIST_L2, 5)
            distance_transform[distance_transform==0] = np.nan

            sum_distance = np.nansum(distance_transform)
            mean_distance = np.nanmean(distance_transform)
            max_distance = np.nanmax(distance_transform)

            # Sum Distance Transform By Object Area
            if not np.isnan(sum_distance) and object_area>0:
                sum_distance_by_object_area = sum_distance / object_area
            else:
                sum_distance_by_object_area = 0
            feature_values[f"Sum Distance Transform By Object Area {self.sub_comp_names[sc]}"] = sum_distance_by_object_area

            # Sum Distance Transform By Sub-compartment Area
            if sub_compartment_area>0:
                sum_distance_by_subcompartment_area = sum_distance / sub_compartment_area
            else:
                sum_distance_by_subcompartment_area = 0

            feature_values[f"Sum Distance Transform By {self.sub_comp_names[sc]} Area"] = sum_distance_by_subcompartment_area

            # Sum Distance Transform
            if not np.isnan(sum_distance):
                feature_values[f"Sum Distance Transform {self.sub_comp_names[sc]}"] = sum_distance
            else:
                feature_values[f"Sum Distance Transform {self.sub_comp_names[sc]}"] = 0


            # Mean Distance Transform By Object Area
            if not np.isnan(mean_distance) and object_area>0:
                mean_distance_by_object_area = mean_distance / object_area
            else:
                mean_distance_by_object_area = 0
            feature_values[f"Mean Distance Transform By Object Area {self.sub_comp_names[sc]}"] = mean_distance_by_object_area

            # Mean Distance Transform By Compartment Area
            if sub_compartment_area>0 and not np.isnan(mean_distance):
                mean_distance_by_subcompartment_area = mean_distance / sub_compartment_area
            else:
                mean_distance_by_subcompartment_area = 0

            feature_values[f"Mean Distance Transform By {self.sub_comp_names[sc]} Area"] = mean_distance_by_subcompartment_area

            # Mean Distance Transform
            if not np.isnan(mean_distance):
                feature_values[f"Mean Distance Transform {self.sub_comp_names[sc]}"] = mean_distance
            else:
                feature_values[f"Mean Distance Transform {self.sub_comp_names[sc]}"] = 0

            # Max Distance Transform By Object Area
            if not np.isnan(max_distance):
                max_distance_by_object_area = max_distance / object_area
            else:
                max_distance_by_object_area = 0
            feature_values[f"Max Distance Transform By Object Area {self.sub_comp_names[sc]}"] = max_distance_by_object_area

            # Max Distance Transform By Compartment Area
            if not np.isnan(max_distance) and sub_compartment_area>0:
                max_distance_by_subcompartment_area = max_distance / sub_compartment_area
            else:
                max_distance_by_subcompartment_area = 0

            feature_values[f"Max Distance Transform By {self.sub_comp_names[sc]} Area"] = max_distance_by_subcompartment_area

            # Max Distance Transform
            if not np.isnan(max_distance):
                feature_values[f"Max Distance Transform {self.sub_comp_names[sc]}"] = max_distance
            else:
                feature_values[f"Max Distance Transform {self.sub_comp_names[sc]}"] = 0

        return feature_values

    def calculate_color_features(self,image, subcompartment_mask):
        # Function to calculate color features for each compartment
        feature_values = {}

        for sc in range(len(self.sub_comp_names)):  # As there are 3 compartments
            compartment_pixels = image[subcompartment_mask[:,:,sc]>0]

            if len(compartment_pixels) > 0:
                # Mean Color
                mean_color = np.mean(compartment_pixels, axis=0)
                for i, channel_value in enumerate(mean_color):
                    feature_values[f"Mean {['Red', 'Green', 'Blue'][i]} {self.sub_comp_names[sc]}"] = channel_value

                # Standard Deviation Color
                std_dev_color = np.std(compartment_pixels, axis=0)
                for i, channel_value in enumerate(std_dev_color):
                    feature_values[f"Standard Deviation {['Red', 'Green', 'Blue'][i]} {self.sub_comp_names[sc]}"] = channel_value
            else:
                # If compartment has no pixels, set values to zero
                for i in range(3):
                    feature_values[f"Mean {['Red', 'Green', 'Blue'][i]} {self.sub_comp_names[sc]}"] = 0.0
                    feature_values[f"Standard Deviation {['Red', 'Green', 'Blue'][i]} {self.sub_comp_names[sc]}"] = 0.0

        return feature_values

    def calculate_texture_features(self,image, subcompartment_mask):
        # Function to calculate texture features for each compartment
        feature_values = {}
        texture_feature_names = ['Contrast', 'Homogeneity', 'Correlation', 'Energy']

        for sc in range(len(self.sub_comp_names)):  # As there are 3 compartments
            compartment_pixels = (subcompartment_mask[:,:,sc]>0).astype(np.uint8)
            compartment_image = cv2.bitwise_and(image, image, mask=compartment_pixels)
            compartment_image_gray = rgb2gray(compartment_image)
            compartment_image_gray_uint = (compartment_image_gray * 255).astype(np.uint8)
            texture_matrix = graycomatrix(compartment_image_gray_uint, [1], [0], levels=256, symmetric=True, normed=True)

            for i, texture_name in enumerate(texture_feature_names):
                texture_feature_value = graycoprops(texture_matrix, texture_name.lower())
                feature_values[f"{texture_name} {self.sub_comp_names[sc]}"] = texture_feature_value[0][0]

        return feature_values

    def calculate_morphological_features(self,subcompartment_mask):
        # Function to calculate morphological features for each compartment
        feature_values = {}

        # Region properties for total object
        object_mask = np.sum(subcompartment_mask,axis=-1).astype(int)
        object_props = regionprops(object_mask)[0]
        
        # Compartment Area By Object Area and Compartment Area
        object_area = np.sum(subcompartment_mask)
        for sc in range(len(self.sub_comp_names)):  # As there are 3 compartments
            subcompartment_area = np.sum(subcompartment_mask[:,:,sc])
            feature_values[f"{self.sub_comp_names[sc]} Area By Object Area "] = subcompartment_area / object_area
            feature_values[f"{self.sub_comp_names[sc]} Area "] = subcompartment_area

        # Calculate Nuclei Number
        nuclei_number = np.max(label(subcompartment_mask[:,:,self.sub_comp_names.index('Nuclei')]))
        feature_values[f"Nuclei Number {self.sub_comp_names[2]}"] = nuclei_number

        # Calculate Mean Aspect Ratio and Standard Deviation Aspect Ratio for the nuclei compartment
        nuclei_label = label(subcompartment_mask[:,:,self.sub_comp_names.index('Nuclei')])
        nuclei_props = regionprops(nuclei_label)
        aspect_ratios = [i.axis_major_length / i.axis_minor_length for i in nuclei_props]

        if not np.isnan(np.nanmean(aspect_ratios)):
            feature_values[f"Mean Aspect Ratio {self.sub_comp_names[2]}"] = np.nanmean(aspect_ratios)
        else:
            feature_values[f"Mean Aspect Ratio {self.sub_comp_names[2]}"] = 0
        if not np.isnan(np.nanstd(aspect_ratios)):
            feature_values[f"Standard Deviation Aspect Ratio {self.sub_comp_names[2]}"] = np.nanstd(aspect_ratios)
        else:
            feature_values[f"Standard Deviation Aspect Ratio {self.sub_comp_names[2]}"] = 0

        # Calculate Mean Nuclear Area for the nuclei compartment
        if nuclei_number>0:
            feature_values[f"Mean Nuclei Area {self.sub_comp_names[2]}"] = np.sum(subcompartment_mask[:,:,self.sub_comp_names.index('Nuclei')]) / nuclei_number
        else:
            feature_values[f"Mean Nuclei Area {self.sub_comp_names[2]}"] = 0

        # Total Object Area
        feature_values["Total Object Area Total compartment"] = object_area

        # Total Object Perimeter
        feature_values["Total Object Perimeter Total compartment"] = object_props.perimeter

        # Total Object Aspect Ratio
        if not object_props.axis_minor_length==0:
            feature_values["Total Object Aspect Ratio Total compartment"] = object_props.axis_major_length / object_props.axis_minor_length
        else:
            feature_values["Total Object Aspect Ratio Total compartment"] = 0

        # Major Axis Length
        feature_values["Major Axis Length Total compartment"] = object_props.axis_major_length

        # Minor Axis Length
        feature_values["Minor Axis Length Total compartment"] = object_props.axis_minor_length

        return feature_values

    def post_annotations(self):

        # Deleting old annotations
        self.gc.delete(f'annotation/item/{self.slide_item_id}?token={self.user_token}')
        
        # Updating with new annotations
        self.gc.post(f'/annotation/item/{self.slide_item_id}?token={self.user_token}',
                     data = json.dumps(self.annotations),
                     headers={
                         'X-HTTP-Method':'POST',
                         'Content-Type':'application/json'
                         }
                    )




parser = argparse.ArgumentParser()
parser.add_argument('--basedir')
parser.add_argument('--girderApiUrl')
parser.add_argument('--girderToken')
parser.add_argument('--input_image')
parser.add_argument('--threshold_nuclei')
parser.add_argument('--minsize_nuclei')
parser.add_argument('--threshold_PAS')
parser.add_argument('--minsize_PAS')
parser.add_argument('--threshold_LAS')
parser.add_argument('--minsize_LAS')
args = parser.parse_args()

gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
gc.setToken(args.girderToken)
#getting file_id
folder = args.basedir
girder_folder_id = folder.split('/')[-2]
_ = os.system("printf 'Using data from girder_client Folder: {}\n'".format(folder))
file_name = args.input_image.split('/')[-1]
files = list(gc.listItem(girder_folder_id))
item_dict = dict()
for file in files:
    d = {file['name']: file['_id']}
    item_dict.update(d)
print(item_dict)
print(item_dict[file_name])

file_id = item_dict[file_name]


        
threshold_nuclei = int(args.threshold_nuclei)
minsize_nuclei =  int(args.minsize_nuclei)
threshold_PAS = int(args.threshold_PAS)
minsize_PAS = int(args.minsize_PAS)
threshold_LAS = int(args.threshold_LAS)
minsize_LAS = int(args.minsize_LAS)
    

sub_seg_params = [
        {
            'name':'Nuclei',
            'threshold':threshold_nuclei,
            'min_size':minsize_nuclei,
            'color':[0,0,255],
            'marks_color':'rgb(0,0,255)'
        },            
        {
            'name':'PAS',
            'threshold':threshold_PAS,
            'min_size':minsize_PAS,
            'color':[255,0,0],
            'marks_color':'rgb(255,0,0)'
        },
        {
            'name':'Luminal Space',
            'threshold':threshold_LAS,
            'min_size':minsize_LAS,
            'color':[0,255,0],
            'marks_color':'rgb(0,255,0)'
        }
    ]

feature_list = ['Distance Transform Features','Color Features','Texture Features','Morphological Features']

output_path = args.basedir + '/tmp'
os.makedirs(args.basedir + "/tmp", exist_ok=True)

FeatureExtractor(args.girderApiUrl, args.girderToken, file_id, sub_seg_params, feature_list, output_path)

for path in filenames:
    gc.uploadFileToItem(file_id, path, reference=None, mimeType=None, filename=None, progressCallback=None)

shutil.rmtree(output_path)
print("done")

'''
def main():

    import os

    # Using DSA as base directory for storage and accessing files=
    dsa_url = 'http://ec2-3-230-122-132.compute-1.amazonaws.com:8080/api/v1/'
    #username = os.environ.get('DSA_USER')
    #p_word = os.environ.get('DSA_PWORD')

    gc = girder_client.GirderClient(apiUrl=dsa_url)
    #gc.authenticate(username,p_word)
    #user_token = gc.get('token/session')['token']
    user_token = "6Rvod0MFrzXJWK5XDxxMO9qy63zMzzXWLGrml3rpZBiVQcgQsl7c2R7PppdNmCVx"

    slide_id = '64fa04772d82d04be3e59346'
    sub_seg_params = [
        {
            'name':'Nuclei',
            'threshold':200,
            'min_size':45,
            'color':[0,0,255],
            'marks_color':'rgb(0,0,255)'
        },            
        {
            'name':'PAS',
            'threshold':45,
            'min_size':20,
            'color':[255,0,0],
            'marks_color':'rgb(255,0,0)'
        },
        {
            'name':'Luminal Space',
            'threshold':0,
            'min_size':20,
            'color':[0,255,0],
            'marks_color':'rgb(0,255,0)'
        }
    ]

    feature_list = ['Distance Transform Features','Color Features','Texture Features','Morphological Features']
    output_path = '/Users/sumanthdevarasetty/Desktop/FTX/'

    FeatureExtractor(dsa_url, user_token,slide_id,sub_seg_params,feature_list,output_path)


if __name__=='__main__':
    main()

'''