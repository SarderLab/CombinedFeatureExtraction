import sys
import os, girder_client
import numpy as np
from tqdm import tqdm
import pandas as pd
from tiffslide import TiffSlide
import lxml.etree as ET
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime

MODx=np.zeros((3,))
MODy=np.zeros((3,))
MODz=np.zeros((3,))
MODx[0]= 0.644211
MODy[0]= 0.716556
MODz[0]= 0.266844

MODx[1]= 0.175411
MODy[1]= 0.972178
MODz[1]= 0.154589

MODx[2]= 0.0
MODy[2]= 0.0
MODz[2]= 0.0
MOD=[MODx,MODy,MODz]

from fextract.extraction_utils.extract_ffpe_features import xml_to_mask
from .process_mc_features import process_glom_features, process_tubules_features, process_arteriol_features
from fextract.extraction_utils.xml_to_mask_minmax import write_minmax_to_xml

SATURATION_THRESHOLD =  0.3
H_THRESHOLD = 160
WHITESPACE_THRESHOLD = 0.88

def get_feature_matrix(features, columns):
    m, n = len(features), len(columns)
    feature_matrix = np.zeros((m, n)) 
    for i, feat in enumerate(features):
        for j in range(n):
            feature_matrix[i,j] = feat[j]
        
    return feature_matrix

def getPathomicFeatures(args):

    cores=multiprocessing.cpu_count()

    # assert args.target is not None, 'Directory of xmls must be specified, use --target /path/to/files.xml'
    # assert args.wsis is not None, 'Directory of WSIs must be specified, use --wsis /path/to/wsis'

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    file_name = args.file.split('/')[-1]
    slide_item_id = args.item_id
    output_dir = args.base_dir
    slide_name,slideExt=file_name.split('.')

    svsfile, xmlfile = args.file, args.xml_path

    print(xmlfile,'here')
    write_minmax_to_xml(xmlfile)
    slide = TiffSlide(svsfile)

    all_contours = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    xlsx_path = os.path.join(output_dir, os.path.basename(svsfile).split('.')[0] +'_pathomic'+'.xlsx')
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']
        if annotationID not in ['1','2','3','4','5','6']:
            pass
        else:
            for Region in Annotation.findall("./*/Region"): # iterate on all region
                verts=[]
                for Vert in Region.findall("./Vertices/Vertex"): # iterate on all vertex in region
                    verts.append([int(float(Vert.attrib['X'])),int(float(Vert.attrib['Y']))])
                all_contours[annotationID].append(np.array(verts))

    glom_features=Parallel(n_jobs=cores,prefer="threads")(delayed(process_glom_features)(points,
        MOD, slide, h_threshold=H_THRESHOLD, saturation_threshold=SATURATION_THRESHOLD) for points in tqdm(all_contours['3'],colour='yellow',unit='Glomerulus',leave=False))
    s_glom_features=Parallel(n_jobs=cores,prefer="threads")(delayed(process_glom_features)(points,
            MOD, slide, h_threshold=H_THRESHOLD, saturation_threshold=SATURATION_THRESHOLD) for points in tqdm(all_contours['4'],colour='yellow',unit='Scl. Glomerulus',leave=False))
    tub_features = Parallel(n_jobs=cores,prefer="threads")(delayed(process_tubules_features)(points,
            MOD, slide,whitespace_threshold=WHITESPACE_THRESHOLD) for points in tqdm(all_contours['5'],colour='blue',unit='Tubule',leave=False))
    art_features=Parallel(n_jobs=cores,prefer="threads")(delayed(process_arteriol_features)(points,
            ) for points in tqdm(all_contours['6'], colour='magenta',unit='Artery(-iole)',leave=False))
    glom_features=[i for i in glom_features if i is not None]
    s_glom_features=[i for i in s_glom_features if i is not None]
    tub_features=[i for i in tub_features if i is not None]
    art_features=[i for i in art_features if i is not None]

    all_columns = [['x1','x2','y1','y2','Area','Mesangial Area','Mesangial Fraction'],
                ['x1','x2','y1','y2','Area','Mesangial Area','Mesangial Fraction'],
                ['x1','x2','y1','y2','Average TBM Thickness','Average Cell Thickness','Luminal Fraction'],
                ['x1','x2','y1','y2','Arterial Area']]
    compart_names = ['gloms','s_gloms','tubs','arts']

    gloms = get_feature_matrix(glom_features, all_columns[0])
    s_gloms = get_feature_matrix(s_glom_features,all_columns[1])
    tubs = get_feature_matrix(tub_features, all_columns[2])
    arts = get_feature_matrix(art_features,all_columns[3])

    all_comparts = [gloms,s_gloms,tubs, arts]
    
    _ = os.system("printf '\tWriting Excel file: [{}]\n'".format(xlsx_path))
    with pd.ExcelWriter(xlsx_path) as writer:
        for idx,compart in enumerate(all_comparts):
            df = pd.DataFrame(compart,columns=all_columns[idx])
            df.to_excel(writer, index=False, sheet_name=compart_names[idx])
    
    gc.uploadFileToItem(slide_item_id, xlsx_path, reference=None, mimeType=None, filename=None, progressCallback=None)
    uploadFilesToUserFolder(gc, [xlsx_path], slide_item_id)
    print('Girder file uploaded!')

def uploadFilesToUserFolder(gc, outpt_filenames, slide_item_id):
    print('Uploading files to user folder')
    if(len(outpt_filenames)==0 or slide_item_id is None):
        print('No files to upload or slide item id not provided')
        return
    try:
        time_now = datetime.now()
        plugin_name = 'cominedfe_pathomic'
        time_stamp = time_now.strftime("%m_%d_%Y__%H_%M_%S")
        getItemName = gc.getItem(slide_item_id).get('name').split('.')[0]
        print('pathomic _ Item name: ', getItemName)
        user_id = gc.get('/user/me').get('_id')
        print('pathomic_user_id:', user_id)
        getListFolder = gc.listFolder(user_id, 'user', 'Private')
        print('pathomic_getListFolder:', getListFolder)
        getPrivateFolder = next(getListFolder)
        print('pathomic_getPrivateFolder:', getPrivateFolder)
        getSlideFolder = gc.loadOrCreateFolder(getItemName, getPrivateFolder.get('_id'), getPrivateFolder.get('_modelType'))
        print('pathomic_getSlideFolder:', getSlideFolder)
        getPluginFolder = gc.loadOrCreateFolder(plugin_name, getSlideFolder.get('_id'), getSlideFolder.get('_modelType'))
        print('pathomic_getPluginFolder:', getPluginFolder)
        getWorkFolder = gc.loadOrCreateFolder(time_stamp, getPluginFolder.get('_id'), getPluginFolder.get('_modelType'))
        print('pathomic_getWorkFolder:', getWorkFolder)
        for file in outpt_filenames:
            gc.uploadFileToFolder(getWorkFolder.get('_id'), file, reference=None, mimeType=None, filename=None, progressCallback=None)
    except Exception as e:
        print('Error in uploading files to user folder:', e)
    
    print('uploading files to user folder done!')

