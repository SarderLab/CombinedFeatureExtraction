import os
import sys
from glob import glob
import girder_client
from ctk_cli import CLIArgumentParser
from tiffslide import TiffSlide
sys.path.append("..")
#from segmentationschool.utils.json_to_xml import get_xml_path

NAMES = ['cortical_interstitium','medullary_interstitium','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']

def main(args):
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)
    
    # folder = args.base_dir
    # wsi = args.input_files
    # file_name = wsi.split('/')[-1]
    # print(file_name)

    file_id = args.input_files
    file_info = gc.get(f'/file/{file_id}')
    item_id = file_info['itemId']
    item_info = gc.get(f'/item/{item_id}')

    file_name = file_info['name']
    print(file_id , file_name, item_id)
    
    cwd = os.getcwd()
    print(cwd)

    mounted_path = '{}/{}'.format('/mnt/girder_worker', os.listdir('/mnt/girder_worker')[0])

    gc.downloadItem(item_id, mounted_path, file_name)
    file_path = '{}/{}'.format(mounted_path,file_name)
    if not os.path.isfile(file_path):
        file_path = '{}/{}'.format(file_path,file_name)

    print(file_path,'2')
    # print ('\n')
    # tmp = os.path.dirname(file_name)
    # print(tmp)
    slide = TiffSlide(file_path)


    dim_x, dim_y=slide.dimensions
    print('made it 2',dim_x, dim_y)
    # folder_split  = folder.split('/')[:4]
    # folder_join = '/'.join(folder_split)
    # new_file = '{}/{}'.format(folder_join,file_name)

    # print(new_file,'my new file')
    # try:
    #     slide=TiffSlide(new_file)
    #     print('made it')
    # except:
    #     print('no luck 2')

    # base_dir_id = folder.split('/')[-2]
    # _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))

    

    # files = list(gc.listItem(base_dir_id))

    # item_dict = dict()
    # for file in files:
    #     d = {file['name']:file['_id']}
    #     item_dict.update(d)

    # cwd = os.getcwd()
    # print(cwd)
    
    tmp = mounted_path
    # wsi_xml = []
    #for file_path, file_name in file_names:
    # file_id = item_dict[file_name]

    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(file_name))
    # get annotation
    # annotations= gc.get('/annotation/item/{}'.format(file_id), parameters={'sort': 'updated'})
    # annotations.reverse()
    # annotations = list(annotations)
    
    # annotations_filtered = [annot for annot in annotations if annot['annotation']['name'].strip() in NAMES]
    # _ = os.system("printf '\tfound [{}] annotation layers...\n'".format(len(annotations_filtered)))
    # del annotations
    # # create root for xml file
        
    # xml_path = get_xml_path(annotations_filtered, NAMES, tmp, file_name)  

    # wsi_xml=[(file_path, xml_path, file_id)]

    
    # _ = os.system("printf '\ndone retriving data...\n\n'")
    
    # cmd = "python3 ../segmentationschool/segmentation_school.py --type {} --base_dir {} --wsi_xml {} --girderApiUrl {} --girderToken {}".format(args.type, args.base_dir, wsi_xml, args.girderApiUrl, args.girderToken)
    # print(cmd)
    # sys.stdout.flush()
    # os.system(cmd)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())