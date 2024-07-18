import os
import sys
from ctk_cli import CLIArgumentParser
from tiffslide import TiffSlide
sys.path.append("..")
from extractioncodes.FeatureExtractor import FeatureExtractor

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

    item_info = gc.get(f'/item/{item_id}')

    file_name = file_info['name']
    print(f'Running on: {file_name}')

    # mounted_path = '{}/{}'.format('/mnt/girder_worker', os.listdir('/mnt/girder_worker')[0])
    mounted_path = os.getenv('TMPDIR')
    file_path = '{}/{}'.format(mounted_path,file_name)
    gc.downloadFile(file_id, file_path)

    print(f'This is slide path: {file_path}')

    slide = TiffSlide(file_path)
    dim_x, dim_y = slide.dimensions

    print(f'Read the slide with dimensions: {dim_x, dim_y}')

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
            'name':'Eosinophilic',
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
    feature_list = ['Distance Transform Features','Color Features','Texture Features','Morphological Features']

    # Getting structures to skip
    skip_structures = args.ignoreAnns.split(',')
    if not type(skip_structures)==list:
        skip_structures = [skip_structures]
    skip_structures = [layer.strip() for layer in skip_structures]

    output_path = '/tmp/'

    FeatureExtractor(
        gc = gc,
        slide = slide,
        slide_item_id = item_id,
        sub_seg_params=sub_seg_params,
        feature_list = feature_list,
        skip_structures = skip_structures,
        test_run = args.type == 'Test_Run',
        output_path = output_path,
        replace_annotations = args.replace_annotations,
        returnXlsx = args.returnXlsx
    )

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
