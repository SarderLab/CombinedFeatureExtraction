"""
Local (non-Girder) entry point for PathomicsFE feature extraction.

Accepts local file paths for the WSI and a folder of per-annotation-type
JSON files instead of Girder file IDs. Outputs are written to a local directory.

Usage example
-------------
python PathomicsFELocal.py \
    --input_image /data/slides/sample.svs \
    --annotations_dir /data/annotations/ \
    --output_dir /data/output \
    --threshold_nuclei 200 \
    --minsize_nuclei 20 \
    --threshold_PAS 50 \
    --minsize_PAS 20 \
    --threshold_LS 0 \
    --minsize_LS 0

Annotation folder
-----------------
The folder should contain one .json file per annotation type, as exported
from DSA/HistomicsUI (Annotations → Download → JSON per layer):

  annotations/
    tubules.json
    arteries_arterioles.json
    non_globally_sclerotic_glomeruli.json
    ...

Each file is a single annotation object:
  { "_id": "...", "annotation": { "name": "tubules", "elements": [...] }, ... }

A single .json file (list format) is also accepted for backward compatibility.

Outputs
-------
  <output_dir>/
    <annotation_name>_Features.xlsx   -- per-element feature spreadsheets
    metadata.json                     -- aggregated slide-level statistics
    sub_compartment_params.json       -- segmentation parameters used
    annotations_updated.json          -- original annotations enriched with
                                         per-element feature values
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from fextract.extractioncodes.LocalFeatureExtractor import LocalFeatureExtractor


def main(args):
    sys.stdout.flush()

    print(f'Running on: {args.input_image}')

    sub_seg_params = [
        {'name': 'Nuclei',       'threshold': int(args.threshold_nuclei), 'min_size': int(args.minsize_nuclei)},
        {'name': 'Eosinophilic', 'threshold': int(args.threshold_PAS),    'min_size': int(args.minsize_PAS)},
        {'name': 'Luminal Space','threshold': int(args.threshold_LS),     'min_size': int(args.minsize_LS)},
    ]

    feature_list = [
        'Distance Transform Features',
        'Color Features',
        'Texture Features',
        'Morphological Features',
    ]

    LocalFeatureExtractor(
        slide_path=args.input_image,
        annotations_input=args.annotations_dir,
        sub_seg_params=sub_seg_params,
        feature_list=feature_list,
        output_path=args.output_dir,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PathomicsFE — local mode (no Girder required)'
    )

    parser.add_argument(
        '--input_image', required=True,
        help='Path to the whole slide image (.svs, .tiff, .ndpi, etc.)'
    )
    parser.add_argument(
        '--annotations_dir', required=True,
        help=(
            'Path to a folder containing per-annotation-type .json files '
            '(as exported from DSA/HistomicsUI), or a single .json file.'
        )
    )
    parser.add_argument(
        '--output_dir', default='./output',
        help='Directory where results are written (default: ./output)'
    )
    parser.add_argument(
        '--threshold_nuclei', type=int, default=200,
        help='Pixel intensity threshold for nuclei sub-compartment (default: 200)'
    )
    parser.add_argument(
        '--minsize_nuclei', type=int, default=20,
        help='Minimum object size in pixels for nuclei (default: 20)'
    )
    parser.add_argument(
        '--threshold_PAS', type=int, default=50,
        help='Pixel intensity threshold for PAS sub-compartment (default: 50)'
    )
    parser.add_argument(
        '--minsize_PAS', type=int, default=20,
        help='Minimum object size in pixels for PAS (default: 20)'
    )
    parser.add_argument(
        '--threshold_LS', type=int, default=0,
        help='Pixel intensity threshold for luminal space (default: 0)'
    )
    parser.add_argument(
        '--minsize_LS', type=int, default=0,
        help='Minimum object size in pixels for luminal space (default: 0)'
    )

    main(parser.parse_args())
