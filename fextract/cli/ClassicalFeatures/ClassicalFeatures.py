import os
import sys
from glob import glob
import girder_client
from ctk_cli import CLIArgumentParser
from tiffslide import TiffSlide
sys.path.append("..")
from fextract.extractioncodes.run_feature_extraction import run_main
from fextract.extraction_utils.json_to_xml import get_xml_path

NAMES = ['cortical_interstitium','medullary_interstitium','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']

def main(args):
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    # Finding the id for the current WSI (input_image)
    file_id = args.input_file
    file_info = gc.get(f'/file/{file_id}')
    item_id = file_info['itemId']

    item_info = gc.get(f'/item/{item_id}')

    file_name = file_info['name']
    print(f'Running on: {file_name}')

    folder_id = item_info['folderId']
    folder_info = gc.get(f'/folder/{folder_id}')    
    print(f'{file_name} is in {folder_info["name"]}')
    if os.path.exists('/mnt/girder_worker'):
        print('Using /mnt/girder_worker as mounted path')
        mounted_path = '{}/{}'.format('/mnt/girder_worker', os.listdir('/mnt/girder_worker')[0])
    else:
        print('Using /tmp/ as mounted path') 
        mounted_path = os.getenv('TMPDIR')
    file_path = '{}/{}'.format(mounted_path,file_name)
    gc.downloadFile(file_id, file_path)

    print(f'This is slide path: {file_path}')

    tmp = mounted_path

    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(file_name))
    # get annotation
    annotations= gc.get('/annotation/item/{}'.format(item_id), parameters={'sort': 'updated'})
    annotations.reverse()
    annotations = list(annotations)
    
    annotations_filtered = [annot for annot in annotations if annot['annotation']['name'].strip() in NAMES]
    _ = os.system("printf '\tfound [{}] annotation layers...\n'".format(len(annotations_filtered)))
    del annotations
    # create root for xml file
    xml_path = get_xml_path(annotations_filtered, NAMES, tmp, file_name)  
    
    setattr(args,'xml_path',xml_path)
    setattr(args,'item_id',item_id)
    setattr(args,'file',file_path)
    setattr(args,'base_dir',tmp)

    run_main(args)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
