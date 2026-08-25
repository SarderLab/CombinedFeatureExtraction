import os
import sys
from tiffslide import TiffSlide
sys.path.append("..")
from extractioncodes.FeatureExtractor import FeatureExtractor
from fextract.storage_client import StorageClient


def main():
    item_id = os.environ['ITEM_ID']
    storage_api_url = os.environ['STORAGE_API_URL']
    job_auth_token = os.environ['JOB_AUTH_TOKEN']
    job_type = os.environ.get('TYPE', 'Feature_Pipeline')

    client = StorageClient(storage_api_url, job_auth_token)

    mounted_path = os.getenv('TMPDIR', '/tmp')
    print(f'Downloading input for item {item_id} to {mounted_path}')
    file_path = client.download_input(item_id, mounted_path)
    print(f'This is slide path: {file_path}')

    slide = TiffSlide(file_path)
    dim_x, dim_y = slide.dimensions

    print(f'Read the slide with dimensions: {dim_x, dim_y}')

    # Converting sub-compartment segmentation parameters to correct format
    # env var names/casing match api/services/job_dispatch_common.py's build_job_env() exactly —
    # these come straight from the canonical_values dict, not an uppercase convention
    thresh_nuc = int(os.environ.get('threshold_nuclei', '200'))
    minsize_nuc = int(os.environ.get('minsize_nuclei', '20'))
    thresh_pas = int(os.environ.get('threshold_PAS', '50'))
    minsize_pas = int(os.environ.get('minsize_PAS', '20'))
    thresh_ls = int(os.environ.get('threshold_LS', '0'))
    minsize_ls = int(os.environ.get('minsize_LS', '0'))

    sub_seg_params = [
        {'name': 'Nuclei', 'threshold': thresh_nuc, 'min_size': minsize_nuc},
        {'name': 'Eosinophilic', 'threshold': thresh_pas, 'min_size': minsize_pas},
        {'name': 'Luminal Space', 'threshold': thresh_ls, 'min_size': minsize_ls},
    ]

    feature_list = ['Distance Transform Features', 'Color Features', 'Texture Features', 'Morphological Features']

    skip_structures = os.environ.get(
        'IGNORE_ANNS', 'tubules,cortical_interstitium,medullary_interstitium').split(',')
    skip_structures = [layer.strip() for layer in skip_structures]

    output_path = '/tmp/'

    FeatureExtractor(
        client=client,
        slide=slide,
        slide_item_id=item_id,
        sub_seg_params=sub_seg_params,
        feature_list=feature_list,
        skip_structures=skip_structures,
        test_run=job_type == 'Test_Run',
        output_path=output_path,
        replace_annotations=os.environ.get('REPLACE_ANNOTATIONS', 'false').lower() == 'true',
        returnXlsx=os.environ.get('RETURN_XLSX', 'true').lower() == 'true',
        glom_index=int(os.environ.get('GLOM_INDEX', '-1')),
        vessel_index=int(os.environ.get('VESSEL_INDEX', '-1')),
    )


if __name__ == "__main__":
    main()
