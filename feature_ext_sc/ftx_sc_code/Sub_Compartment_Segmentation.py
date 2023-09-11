import numpy as np 
from skimage import color, exposure
from skimage import morphology, segmentation, feature, measure
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from PIL import Image
import skimage.measure._label as label
from skimage.color import rgb2hsv
import girder_client
import sys
sys.path.append("..")
from ftx_sc_code.annonaterator import Annotaterator
from ftx_sc_code.features import calculate_distance_transform_features, calculate_color_features, calculate_morphological_features, calculate_features
import csv
import argparse
import os

def sub_segment_image(image,mask,seg_params,view_method,transparency_val):
        
        # Sub-compartment segmentation
        sub_comp_image = np.zeros((np.shape(image)[0],np.shape(image)[1],3))
        remainder_mask = np.ones((np.shape(image)[0],np.shape(image)[1]))
        hsv_image = np.uint8(255*rgb2hsv(image)[:,:,1])

        # Applying adaptive histogram equalization
        #hsv_image = rank.equalize(hsv_image,footprint=disk(30))
        hsv_image = np.uint8(255*exposure.equalize_hist(hsv_image))

        for idx,param in enumerate(seg_params):

            remaining_pixels = np.multiply(hsv_image,remainder_mask)
            masked_remaining_pixels = np.multiply(remaining_pixels,mask)

            # Applying manual threshold
            masked_remaining_pixels[masked_remaining_pixels<param['threshold']] = 0
            masked_remaining_pixels[masked_remaining_pixels>0] = 1

            # Filtering by minimum size
            small_object_filtered = (1/255)*np.uint8(remove_small_objects(masked_remaining_pixels>0,param['min_size']))
            # Check for if the current sub-compartment is nuclei
            if param['name'].lower()=='nuclei':
                
                # Area threshold for holes is controllable for this
                sub_mask = remove_small_holes(small_object_filtered>0,area_threshold=10)
                sub_mask = sub_mask>0
                # Watershed implementation from: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
                distance = ndi.distance_transform_edt(sub_mask)
                coords = peak_local_max(distance,footprint=np.ones((3,3)),labels = sub_mask)
                watershed_mask = np.zeros(distance.shape,dtype=bool)
                watershed_mask[tuple(coords.T)] = True
                markers, _ = ndi.label(watershed_mask)
                sub_mask = watershed(-distance,markers,mask=sub_mask)
                sub_mask = sub_mask>0

            else:
                sub_mask = small_object_filtered

            sub_comp_image[sub_mask>0,:] = param['color']
            remainder_mask -= sub_mask>0

        # Assigning remaining pixels within the boundary mask to the last sub-compartment
        masked_remaining_pixels = np.multiply(remaining_pixels,mask)
        sub_comp_image[masked_remaining_pixels>0] = param['color']

        # have to add the final mask thing for the lowest segmentation hierarchy
        if view_method=='Side-by-side':
            # Side-by-side view of sub-compartment segmentation
            image = np.uint8(image)
            sub_comp_image_uint = np.uint8(sub_comp_image)       
            sub_comp_image_vis = np.concatenate((image,sub_comp_image_uint),axis=1)
            
            
            
        elif view_method=='Overlaid':
            # Overlaid view of sub-compartment segmentation
            # Processing combined annotations to set black background to transparent
            zero_mask = np.where(np.sum(sub_comp_image.copy(),axis=2)==0,0,255*transparency_val)
            sub_comp_mask_4d = np.concatenate((sub_comp_image,zero_mask[:,:,None]),axis=-1)
            rgba_mask = Image.fromarray(np.uint8(sub_comp_mask_4d),'RGBA')
            
            image = Image.fromarray(np.uint8(image)).convert('RGBA')
            image.paste(rgba_mask, mask = rgba_mask)
            sub_comp_image = np.array(image.copy())[:,:,0:3]

        current_sub_comp_image = sub_comp_image

        return sub_comp_image



parser = argparse.ArgumentParser()
parser.add_argument('--basedir')
parser.add_argument('--girderApiUrl')
parser.add_argument('--girderToken')
parser.add_argument('--input_image')
parser.add_argument('--outputdir')
parser.add_argument('--threshold_nuclei')
parser.add_argument('--threshold_PAS')
parser.add_argument('--threshold_LAS')
parser.add_argument('--transparency_val')
args = parser.parse_args()



gc = girder_client.GirderClient(apiUrl = args.girderApiUrl)

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

annotaterator = iter(Annotaterator(gc,file_id))

while True:
    try:
        new_image, new_mask, id = next(annotaterator)
    except StopIteration:
        break
    #print("type of new_image", type(new_image))
    #print("Shape of new_image", np.shape(new_image))

    expanded_mask = np.uint8(255*np.repeat(new_mask[:,:,None],3,axis=-1))
        
    threshold_nuclei = int(args.threshold_nuclei)
  
        
    threshold_PAS = int(args.threshold_PAS)
 
        
    threshold_LAS = int(args.threshold_LAS)
    
        
    seg_params = [{"name": 'nuclei' , "order": 0, "threshold": threshold_nuclei, 'color': (0, 0, 255), "min_size": 45}, {"name": 'PAS' , "order": 1, "threshold": threshold_PAS, 'color': (255, 0, 0), "min_size": 25}, {"name": 'LAS' , "order": 2 , "threshold": threshold_LAS, 'color': (0, 255, 0), "min_size": 25}] 

    view_method = "Overlaid"

    transparency_val = float(args.transparency_val)

    sub_compartment_seg = sub_segment_image(new_image, new_mask, seg_params, view_method, transparency_val)


    # Assigning custom labels to compartments. For easier calculations
    # 29 - blue = nuclei,
    # 76 - red = PAS
    # 149 - green = Luminal space
    compartment_mask = sub_compartment_seg

    luminance = 0.299 * compartment_mask[:, :, 2] + 0.587 * compartment_mask[:, :, 1] + 0.114 * compartment_mask[:, :, 0]


    # Convert to integer type
    luminance = luminance.astype(np.uint8)


    compartment_mask = luminance 
    image = new_image

    compartment_mask[compartment_mask == 29] = 3
    compartment_mask[compartment_mask == 76] = 2
    compartment_mask[compartment_mask == 149] = 1
    compartment_mask[compartment_mask > 3] = 0

    try:
        all_features = calculate_features(image, compartment_mask)
        # Write the color features to a CSV file
        csv_filename = args.basedir + "/tmp/features_output.csv"
        os.makedirs(args.basedir + "/tmp", exist_ok=True)
        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Sub Compartment id", id])
            for feature_name, value in all_features.items():
                writer.writerow([feature_name, ""])
                if isinstance(value, dict):
                    for sub_feature_name, sub_value in value.items():
                        writer.writerow(["", f"{sub_feature_name}", sub_value])
    except ZeroDivisionError:
        pass


# Upload the file
gc.setToken(args.girderToken)
upload_response = gc.uploadFileToFolder(girder_folder_id,csv_filename, reference=None, mimeType=None, filename=None, progressCallback=None)

print("done")


