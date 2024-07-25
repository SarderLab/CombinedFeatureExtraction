import os
from datetime import datetime
import girder_client
import shutil


def uploadFilesToOriginalFolder(gc, output_filenames, slide_item_id, plugin_name, girderApiUrl):
    print('Uploading files to user folder')
    # Get user id
    workPath = createWorkPath(gc, slide_item_id, plugin_name, girderApiUrl)
    user_id = getUserId(gc)
    # Check if there are files to upload
    if (len(output_filenames) == 0):
        print('No files to upload')
        return
    elif (workPath is None):
        print('No work path found')
        return
    # Upload files to imported folder path
    try:
        # Create the directory if it does not exist
        os.path.exists(workPath) or os.makedirs(workPath, mode=0o775)
        # change the permission of the directory
        os.chmod(workPath, 0o775)
        # Add files to the time stamp folder
        for file in output_filenames:
            try:
                shutil.copy2(file, workPath)
            except Exception as e:
                print(f'Error copying files to job items folder: {e}')
    except Exception as e:
        print(f'Error uploading files to user folder: {e}')
    print('uploading files to user folder done!')

# Create work path for the assetstore folder
def createWorkPath(gc, slideItemId, pluginName, girderApiUrl):
    print('Creating work path')
    try:
        importPathDirectory = getAssetstoreImportPath(slideItemId, girderApiUrl)
        if importPathDirectory is None:
            print('No import path directory found')
            return
        elif os.path.isfile(importPathDirectory): 
            print('Import path directory is a file')
            importPathDirectory = os.path.dirname(importPathDirectory)        
        time_now = datetime.now().astimezone()
        time_stamp = time_now.strftime("%m_%d_%Y__%H:%M:%S")
        getItemName = gc.getItem(slideItemId).get('name').split('.')[0]
        workPath = os.path.join(importPathDirectory, getItemName, pluginName, time_stamp)
        return workPath
    except Exception as e:
        print(f'Error getting work path: {e}')
        return None

# Get the assetstore import path
def getAssetstoreImportPath(slideItemId, girderApiUrl):
    print('Getting assetstore import path')
    try:
        api_key=os.getenv('GIRDER_API_KEY', 'lrn6ZVgCdWuHWOhXDpi1nlDC3aUflbEnoik3GEfW')
        # Start admin instance
        print(f'Getting api_key from env status is {api_key is not None}')
        gc_assetstore = girder_client.GirderClient(apiUrl=girderApiUrl)
        gc_assetstore.authenticate(apiKey=api_key)
        # Get the folder id for the slide item
        getItemInfo = gc_assetstore.get(f'/item/{slideItemId}')
        getFolderId = getItemInfo.get('folderId')
        getAssetstoreImports = gc_assetstore.get('/assetstore/all_imports', parameters={'limit': 0})
        # Get the import path for the folder id
        for eachImport in getAssetstoreImports:
            if (eachImport.get('params').get('destinationId') == getFolderId):
                assetStoreFileID = eachImport.get('_id')
                importPath = eachImport.get('params').get('importPath')
        # check if itemID is imported from the assetstore
        print(f'Assetstore id is {assetStoreFileID}')
        print(f'Import path is {importPath}')
        if assetStoreFileID:
            try:
                assetStoreFileRecord = gc_assetstore.get(f'/assetstore/import/{assetStoreFileID}')
                if assetStoreFileRecord:
                    print('Assetstore import item found')
                    return importPath
            except Exception as e:
                print(f'Item not imported: {e}')
                return None
        else:
            print('No assetstore id found')
            return None        
    except Exception as e:
        print(f'Error getting assetstore import path: {e}')
        return None

def getUserId(gc):
    print('Getting user id')
    user = gc.get('/user/me')
    if user is None:
        print('getting userId from token')
        return gc.get('/token/current').get('userId')
    else:
        return user.get('_id')