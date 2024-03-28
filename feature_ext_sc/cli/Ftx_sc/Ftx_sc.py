import os
import sys
from ctk_cli import CLIArgumentParser

sys.path.append("..")
from ftx_sc_code.FeatureExtractor import FeatureExtractor

import girder_client

def main(args):  
    
    sys.stdout.flush()

    # Arguments: 
    # girderApiUrl = URL used for girder WebAPI calls
    # input_image = girder item id for current whole slide image
    # threshold_nuclei, minsize_nuclei, threshold_PAS, minsize_PAS, threshold_LS, minsize_LS = subcompartment segmentation parameters

    # Setting up girder client (initializing with user session token)
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    # Finding the id for the current WSI (input_image)
    file_id = args.input_image
    file_info = gc.get(f'/file/{file_id}')
    item_id = file_info['itemId']

    file_name = file_info['name']
    print(f'Running on: {file_name}')

    folder_id = file_info['folderId']
    folder_info = gc.get(f'/folder/{folder_id}')    
    print(f'{file_name} is in {folder_info["name"]}')

    # Converting sub-compartment segmentation parameters to correct format
    thresh_nuc = int(args.threshold_nuclei)
    minsize_nuc = int(args.minsize_nuclei)
    thresh_pas = int(args.threshold_PAS)
    minsize_pas = int(args.minsize_PAS)
    thresh_ls = int(args.threshold_LS)
    minsize_ls = int(args.minsize_LS)

    # Combining paramters into usable list
    sub_seg_params = [
        {
            'name':'Nuclei',
            'threshold': thresh_nuc,
            'min_size': minsize_nuc
        },
        {
            'name':'PAS',
            'threshold': thresh_pas,
            'min_size': minsize_pas
        },
        {
            'name':'Luminal Space',
            'threshold': thresh_ls,
            'min_size': minsize_ls
        }
    ]

    # Getting list of features to calculate
    feature_list = args.featureCats.split(',')
    if not type(feature_list)==list:
        feature_list = [feature_list]
    
    feature_list = [i.replace('"','') for i in feature_list]

    # Getting structures to skip
    skip_structures = args.ignoreAnns.split(',')
    if not type(skip_structures)==list:
        skip_structures = [skip_structures]

    # Whether or not to rename structures to nicer names
    rename = args.rename

    # Output path for excel files (if specified)
    if args.returnXlsx:
        if 'output_path' in vars(args):
            output_path = args.output_path
        else:
            output_path = '/tmp/'
    else:
        output_path = None

    # Added parameter "test_run". Selecting this runs feature extraction and sub-compartment determination for a single randomly selected structure.

    # Defining feature extractor object which should take care of the rest
    FeatureExtractor(
        gc = gc,
        slide_item_id = item_id,
        sub_seg_params=sub_seg_params,
        feature_list = feature_list,
        skip_structures = skip_structures,
        rename = rename,
        test_run = args.test_run,
        output_path = output_path
    )


if __name__ == "__main__":

    
    # If running this locally, just enter the output_path manually.
    # output_path is not included in the available args specified in Ftx_sc.xml
    """
    class args_object:
        def __init__(self):
            self.girderApiUrl = "http://ec2-3-230-122-132.compute-1.amazonaws.com:8080/api/v1/"
            self.girderToken = ""

            self.input_image = ""

            self.threshold_nuclei = 200
            self.threshold_PAS = 60
            self.threshold_LS = 0
            self.minsize_LS = 0
            self.minsize_nuclei = 50
            self.minsize_PAS = 20

            self.featureCats = "Distance Transform Features,Color Features,Texture Features,Morphological Features"
            self.ignoreAnns = ""
            self.returnXlsx = True
            self.output_path = ""
            self.rename = True
            self.test_run = True


    args = args_object()
    main(args)
    """
    # Comment this line out if running locally
    main(CLIArgumentParser().parse_args())
