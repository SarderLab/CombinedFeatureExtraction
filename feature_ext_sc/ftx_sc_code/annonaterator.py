"""

Example dataloader/iterator for loading annotated regions from DSA for feature extraction

"""

import os
import sys

import girder_client
import json
from skimage.draw import polygon
import numpy as np

from PIL import Image
from io import BytesIO
import requests
from time import sleep

from tqdm import tqdm

import matplotlib.pyplot as plt


class Annotaterator:
    def __init__(self, gc, file_id):
        
        
        self.gc = gc
        self.user_token = self.gc.get('token/session')['token']
        #self.collection = collection
        self.file_id = file_id

        #self.collection_dict = self.gc.get('resource/lookup',parameters={'path':self.collection,'type':'folder'})
        #self.collection_id = self.collection_dict['_id']
        #self.collection_type = self.collection_dict['_modelType']
        # Extra filter step for only images
        #self.collection_items = self.gc.get(f'resource/{self.collection_id}/items',parameters={'type':'folder'})
        #self.collection_items = [i for i in self.collection_items if 'largeImage' in i]

    def __iter__(self):

        #self.current_collection_idx = 0
        self.current_annotation_layer = 0
        self.current_annotation_item = 0

        #self.current_slide_id = self.collection_items[self.current_collection_idx]['_id']

        self.annotations = self.gc.get(f'annotation/item/{self.file_id}')

        return self
    
    def __next__(self):

        # Getting the next annotation item (check if all done)
        self.current_annotation_item+=1
        if self.current_annotation_item>=len(self.annotations[self.current_annotation_layer]['annotation']['elements']):
            self.current_annotation_layer+=1
            if self.current_annotation_layer>=len(self.annotations):
                raise StopIteration
                #self.current_collection_idx+=1
                #if self.current_collection_idx>=len(self.collection_items):
                    # Then all done with the collection
                    #raise StopIteration
                #else:
                    #self.current_annotation_layer = 0
                    #self.current_annotation_item = 0
            else:
                self.current_annotation_item = 0

        # Getting the current annotation item 
        current_item = self.annotations[self.current_annotation_layer]['annotation']['elements'][self.current_annotation_item]
        id = current_item['id']
        print("id",id)
        # There's a few other different shapes available for annotations. Most are polylines.
        if current_item['type']=='polyline':
            coordinates = np.squeeze(np.array(current_item['points']))

            # Defining the bounding box:
            min_x = np.min(coordinates[:,0])    
            min_y = np.min(coordinates[:,1])
            max_x = np.max(coordinates[:,0])
            max_y = np.max(coordinates[:,1])

            # Getting the image region and compartment mask:
            image, mask = self.grab_image_and_mask([min_x,min_y,max_x,max_y],coordinates)

            return image, mask, id
        
    def grab_image_and_mask(self,bbox,coordinates):

        # Grabbing image from bbox region
        image = np.uint8(np.array(Image.open(BytesIO(requests.get(self.gc.urlBase+f'/item/{self.file_id}/tiles/region?token={self.user_token}&left={bbox[0]}&top={bbox[1]}&right={bbox[2]}&bottom={bbox[3]}').content))))
        # Creating mask from coordinates 
        # scaling coordinates so that the boundary mask is relative to the bounding box
        scaled_coordinates = coordinates.tolist()
        scaled_coordinates = [[int(i[0]-bbox[0]),int(i[1]-bbox[1])] for i in scaled_coordinates]

        x_coords = [i[0] for i in scaled_coordinates]
        y_coords = [i[1] for i in scaled_coordinates]

        height = np.shape(image)[0]
        width = np.shape(image)[1]
        mask = np.zeros((height,width))
        cc,rr = polygon(y_coords,x_coords,(height,width))
        mask[cc,rr] = 1

        return image, mask





