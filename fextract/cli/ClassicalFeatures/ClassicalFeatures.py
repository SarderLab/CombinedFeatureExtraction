import argparse
import os
import sys
from glob import glob
from fextract.storage_client import StorageClient
sys.path.append("..")
from fextract.extractioncodes.run_feature_extraction import run_main
from fextract.extraction_utils.json_to_xml import get_xml_path

NAMES = ['cortical_interstitium', 'medullary_interstitium', 'non_globally_sclerotic_glomeruli',
         'globally_sclerotic_glomeruli', 'tubules', 'arteries/arterioles']


def main():
    item_id = os.environ['ITEM_ID']
    storage_api_url = os.environ['STORAGE_API_URL']
    job_auth_token = os.environ['JOB_AUTH_TOKEN']

    client = StorageClient(storage_api_url, job_auth_token)
    args = argparse.Namespace(type='Extended_Clinical')

    mounted_path = os.getenv('TMPDIR', '/tmp')
    print(f'Downloading input for item {item_id} to {mounted_path}')
    file_path = client.download_input(item_id, mounted_path)
    file_name = os.path.basename(file_path)
    print(f'This is slide path: {file_path}')

    tmp = mounted_path

    # get annotation
    annotations = client.get_annotations(item_id)
    annotations.reverse()
    annotations = list(annotations)

    annotations_filtered = [annot for annot in annotations if annot['annotation']['name'].strip() in NAMES]
    print(f'\tfound [{len(annotations_filtered)}] annotation layers...')
    del annotations
    # create root for xml file
    xml_path = get_xml_path(annotations_filtered, NAMES, tmp, file_name)

    args.xml_path = xml_path
    args.item_id = item_id
    args.file = file_path
    args.base_dir = tmp
    args.storage_client = client

    run_main(args)


if __name__ == "__main__":
    main()
