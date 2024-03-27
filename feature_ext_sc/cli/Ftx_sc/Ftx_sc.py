import os
import sys
from ctk_cli import CLIArgumentParser

sys.path.append("..")
from ftx_sc_code.FeatureExtractor import FeatureExtractor

import girder_client

def main(args):  
    
    """
    cmd = "python3 ../ftx_sc_code/FeatureExtractor.py   --basedir '{}' --girderApiUrl '{}' --girderToken '{}' \
             --input_image '{}' --threshold_nuclei {} --minsize_nuclei {} --threshold_PAS {} --minsize_PAS {} --threshold_LAS {} --minsize_LAS {} \
                ".format(args.basedir, args.girderApiUrl, args.girderToken, args.input_image, args.threshold_nuclei, args.minsize_nuclei, args.threshold_PAS, args.minsize_PAS, args.threshold_LAS, args.minsize_LAS)
    """
    #print(cmd)
    sys.stdout.flush()
    #os.system(cmd)  

    # Arguments: 
    # basedir = parent folder of the image for feature extraction
    # girderApiUrl = URL used for girder WebAPI calls
    # input_image = girder item id for current whole slide image
    # threshold_nuclei, minsize_nuclei, threshold_PAS, minsize_PAS, threshold_LS, minsize_LS = subcompartment segmentation parameters

    # Setting up girder client (initializing with user session token)
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    # Accessing item id for the image
    basedir = args.basedir
    _ = os.system("printf 'Base Directory supplied: {}\n'".format(basedir))
    folder_id = basedir.split(os.sep)[-2]

    # Finding the id for the current WSI (input_image)
    file_name = args.input_image.split(os.sep)[-1]
    _ = os.system("printf 'Input Image supplied: {}\n'".format(file_name))

    all_files = list(gc.listItem(folder_id))
    all_file_names = [i['name'] for i in all_files]
    file_id = all_files[all_file_names.index(file_name)]['_id']
    _ = os.system("printf 'Found image: {} with id: {}'".format(file_name,file_id))

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
        output_path = [basedir+'/tmp', args.outputPath]
    else:
        output_path = [None]

    # Defining feature extractor object which should take care of the rest
    FeatureExtractor(
        gc = gc,
        slide_item_id = file_id,
        sub_seg_params=sub_seg_params,
        feature_list = feature_list,
        skip_structures = skip_structures,
        output_path = output_path
        rename = rename
    )


if __name__ == "__main__":

    """
    class args_object:
        def __init__(self):
            self.girderApiUrl = "http://ec2-3-230-122-132.compute-1.amazonaws.com:8080/api/v1/"
            self.girderToken = ""

            self.basedir = ""
            self.input_image = ""

            self.threshold_nuclei = ""
            self.threshold_PAS = ""
            self.threshold_LS = ""
            self.minsize_LS = ""
            self.minsize_nuclei = ""
            self.minsize_PAS = ""

            self.featureCats = "Distance Transform Features,Color Features,Texture Features,Morphological Features"
            self.ignoreAnns = ""
            self.returnXlsx = False
            self.outputPath = ""


    args = args_object()
    main(args)
    """
    main(CLIArgumentParser().parse_args())
