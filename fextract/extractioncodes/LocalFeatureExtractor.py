"""
Local (non-Girder) version of FeatureExtractor.

Loads annotations from a folder of per-annotation-type JSON files
(DSA/HistomicsUI export format) and saves all outputs to local disk.

Accepts either:
  - A directory containing one or more .json files (one per annotation type)
  - A single .json file (for backward compatibility)

Each JSON file can be:
  a) A single annotation object  (as exported from DSA per annotation layer)
     { "_id": "...", "annotation": { "name": "tubules", "elements": [...] }, ... }

  b) A list of annotation objects (as returned by gc.get('annotation/item/{id}'))
     [ { "_id": "...", "annotation": { "name": "...", ... } }, ... ]

Both formats are handled automatically.
"""

import glob
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import exposure

from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

from skimage.draw import polygon
import pandas as pd
import json
from tqdm import tqdm
import os

from tiffslide import TiffSlide

NAMES = [
    'cortical_interstitium', 'medullary_interstitium',
    'non_globally_sclerotic_glomeruli', 'globally_sclerotic_glomeruli',
    'tubules', 'arteries/arterioles', 'gloms'
]


class LocalFeatureExtractor:
    def __init__(self,
                 slide_path: str,
                 annotations_input: str,
                 sub_seg_params: list,
                 feature_list: list,
                 output_path: str):
        """
        Parameters
        ----------
        slide_path : str
            Absolute path to the whole slide image file (.svs, .tiff, etc.)
        annotations_input : str
            Path to either:
            - A directory containing .json files (one per annotation type), OR
            - A single .json file (a list or a single annotation object).
        sub_seg_params : list
            Sub-compartment segmentation parameters, e.g.:
            [{'name': 'Nuclei', 'threshold': 200, 'min_size': 20}, ...]
        feature_list : list
            Feature categories to compute, e.g.:
            ['Distance Transform Features', 'Color Features',
             'Texture Features', 'Morphological Features']
        output_path : str
            Directory where Excel files, metadata.json, and
            annotations_updated.json will be written.
        """
        self.slide = TiffSlide(slide_path)
        self.sub_seg_params = sub_seg_params
        self.feature_list = feature_list
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

        self.replace_annotations = True
        self.returnXlsx = True

        # Build feature extraction dispatch table
        self.sub_comp_names = [i['name'] for i in self.sub_seg_params]
        self.feature_extract_list = {}
        for f in self.feature_list:
            print(f'Feature Type: {f}')
            if f == 'Distance Transform Features':
                self.feature_extract_list[f] = lambda comp: self.calculate_distance_transform_features(comp)
            elif f == 'Color Features':
                self.feature_extract_list[f] = lambda image, comp: self.calculate_color_features(image, comp)
            elif f == 'Texture Features':
                self.feature_extract_list[f] = lambda image, comp: self.calculate_texture_features(image, comp)
            elif f == 'Morphological Features':
                self.feature_extract_list[f] = lambda comp: self.calculate_morphological_features(comp)
            else:
                print(f'Invalid feature type: {f}')

        # Load annotations from a directory of JSON files or a single JSON file
        raw_annotations = self._load_annotations(annotations_input)
        for annot in raw_annotations:
            annot['annotation']['name'] = annot['annotation']['name'].strip()
        self.annotations = [a for a in raw_annotations if a['annotation']['name'] in NAMES]

        print(f'Loaded {len(self.annotations)} annotation layer(s) from {annotations_input}')

        output_filenames = []
        agg_feat_metadata = {}

        for a_idx, ann in tqdm(enumerate(self.annotations), total=len(self.annotations)):
            if 'annotation' in ann:
                if 'interstitium' not in ann['annotation']['name']:

                    compartment_feature_dict = {i: [] for i in self.feature_list}
                    compartment_feature_dict['Bounding Boxes'] = []
                    compartment_ids = []

                    for c_idx, comp in tqdm(
                            enumerate(ann['annotation']['elements']),
                            total=len(ann['annotation']['elements'])):

                        try:
                            image, mask, bbox = self.grab_image_and_mask(comp['points'])
                            sub_compartment_mask = self.sub_segment_image(image, mask)
                        except (UnidentifiedImageError, ValueError) as e:
                            print(e)
                            print(f'Error in {ann["annotation"]["name"]}, element {c_idx}')
                            print(comp['points'])
                            continue

                        if np.sum(np.sum(sub_compartment_mask, axis=-1)) > 0:
                            if 'user' not in comp:
                                comp['user'] = {}

                            compartment_ids.append(ann['annotation']['name'] + f'_{c_idx}')
                            compartment_feature_dict['Bounding Boxes'].append(
                                {'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]}
                            )

                            for feat in self.feature_extract_list:
                                try:
                                    cat_feat = self.feature_extract_list[feat](image, sub_compartment_mask)
                                except Exception:
                                    cat_feat = self.feature_extract_list[feat](sub_compartment_mask)

                                compartment_feature_dict[feat].append(cat_feat)

                                for c_f in cat_feat:
                                    comp['user'][c_f] = np.float64(cat_feat[c_f])

                                self.annotations[a_idx]['annotation']['elements'][c_idx] = comp
                        else:
                            continue

                    if len(compartment_ids) > 0:
                        agg_feat_metadata[f'{ann["annotation"]["name"]}_Morphometrics'] = {}

                        if self.returnXlsx:
                            output_file = os.path.join(
                                self.output_path,
                                f'{ann["annotation"]["name"].replace("/", "")}_Features.xlsx'
                            )
                            output_filenames.append(output_file)

                            with pd.ExcelWriter(output_file, mode='w', engine='openpyxl') as writer:
                                for feat_cat in compartment_feature_dict:
                                    feat_df = pd.DataFrame.from_records(compartment_feature_dict[feat_cat])
                                    feat_df['compartment_ids'] = compartment_ids
                                    feat_df.to_excel(writer, sheet_name=feat_cat)

                                    if not feat_df.empty and feat_cat != 'Bounding Boxes':
                                        agg_feat_df = self.aggregate_features(feat_df)
                                        for a_f in agg_feat_df:
                                            agg_feat_metadata[
                                                f'{ann["annotation"]["name"]}_Morphometrics'
                                            ][a_f] = agg_feat_df[a_f]
                        else:
                            for feat_cat in compartment_feature_dict:
                                feat_df = pd.DataFrame.from_records(compartment_feature_dict[feat_cat])
                                feat_df['compartment_ids'] = compartment_ids

                                if not feat_df.empty and feat_cat != 'Bounding Boxes':
                                    agg_feat_df = self.aggregate_features(feat_df)
                                    for a_f in agg_feat_df:
                                        agg_feat_metadata[
                                            f'{ann["annotation"]["name"]}_Morphometrics'
                                        ][a_f] = agg_feat_df[a_f]
                    else:
                        continue

        # Save aggregated feature metadata to local JSON (replaces gc.put)
        metadata_path = os.path.join(self.output_path, 'metadata.json')
        with open(metadata_path, 'w') as fh:
            json.dump(agg_feat_metadata, fh, indent=2, default=lambda x: float(x))
        print(f'Metadata written to {metadata_path}')

        # Save sub-compartment parameters alongside metadata
        seg_params_path = os.path.join(self.output_path, 'sub_compartment_params.json')
        with open(seg_params_path, 'w') as fh:
            json.dump({'Sub-Compartment Parameters': self.sub_seg_params}, fh, indent=2)
        print(f'Sub-compartment parameters written to {seg_params_path}')

        # Save updated annotations (with per-element user features) to local JSON
        # (replaces gc.post / gc.delete)
        if self.replace_annotations:
            self.save_annotations()

        print(f'Output Excel files ({len(output_filenames)}) written to {self.output_path}')

    # ------------------------------------------------------------------
    # Annotation loading
    # ------------------------------------------------------------------

    def _load_annotations(self, annotations_input: str) -> list:
        """Load annotations from a directory of JSON files or a single JSON file.

        Each file can be a single annotation object (dict) or a list of them.
        All results are merged into a flat list.
        """
        if os.path.isdir(annotations_input):
            json_files = sorted(glob.glob(os.path.join(annotations_input, '*.json')))
            if not json_files:
                raise FileNotFoundError(f'No .json files found in {annotations_input}')
            print(f'Found {len(json_files)} annotation file(s): {[os.path.basename(f) for f in json_files]}')
        else:
            json_files = [annotations_input]

        annotations = []
        for fpath in json_files:
            with open(fpath, 'r') as fh:
                data = json.load(fh)
            # Each file is either a single object or a list of objects
            if isinstance(data, list):
                annotations.extend(data)
            else:
                annotations.append(data)
        return annotations

    # ------------------------------------------------------------------
    # Image / mask helpers
    # ------------------------------------------------------------------

    def grab_image_and_mask(self, coordinates):
        coordinates = np.squeeze(np.array(coordinates))

        min_x = int(np.min(coordinates[:, 0]))
        min_y = int(np.min(coordinates[:, 1]))
        max_x = int(np.max(coordinates[:, 0]))
        max_y = int(np.max(coordinates[:, 1]))

        image = self.slide.read_region((min_x, min_y), 0, (max_x - min_x, max_y - min_y))
        image = np.array(image, dtype=np.uint8)
        image = image[:, :, :3]

        scaled_coordinates = [[int(c[0] - min_x), int(c[1] - min_y)] for c in coordinates.tolist()]
        x_coords = [c[0] for c in scaled_coordinates]
        y_coords = [c[1] for c in scaled_coordinates]

        height = int(max_y - min_y)
        width = int(max_x - min_x)
        mask = np.zeros((height, width))
        cc, rr = polygon(y_coords, x_coords, (height, width))
        mask[cc, rr] = 1

        return image, mask, [min_x, min_y, max_x, max_y]

    def sub_segment_image(self, image, mask):
        sub_comp_image = np.zeros((np.shape(image)[0], np.shape(image)[1], 3))
        remainder_mask = np.ones((np.shape(image)[0], np.shape(image)[1]))

        hsv_image = np.uint8(255 * rgb2hsv(image))
        hsv_image = hsv_image[:, :, 1]

        for idx, param in enumerate(self.sub_seg_params):
            if param['name'].lower() == 'nuclei':
                h_image = 255 - np.uint8(255 * rgb2hsv(image)[:, :, 2])
                h_image = np.uint8(255 * exposure.equalize_hist(h_image, mask=mask))

                remaining_pixels = np.multiply(h_image, remainder_mask)
                masked_remaining_pixels = np.multiply(remaining_pixels, mask)

                masked_remaining_pixels[masked_remaining_pixels <= param['threshold']] = 0
                masked_remaining_pixels[masked_remaining_pixels > 0] = 1

                sub_mask = remove_small_holes(masked_remaining_pixels > 0, area_threshold=10)
                sub_mask = sub_mask > 0

                distance = ndi.distance_transform_edt(sub_mask)
                labeled_mask, _ = ndi.label(sub_mask)
                coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=labeled_mask)
                watershed_mask = np.zeros(distance.shape, dtype=bool)
                watershed_mask[tuple(coords.T)] = True
                markers, _ = ndi.label(watershed_mask)
                sub_mask = watershed(-distance, markers, mask=sub_mask)
                sub_mask = sub_mask > 0

                sub_mask = remove_small_objects(sub_mask, param['min_size'])
            else:
                remaining_pixels = np.multiply(hsv_image, remainder_mask)
                masked_remaining_pixels = np.multiply(remaining_pixels, mask)

                masked_remaining_pixels[masked_remaining_pixels <= param['threshold']] = 0
                masked_remaining_pixels[masked_remaining_pixels > 0] = 1

                small_object_filtered = (1 / 255) * np.uint8(
                    remove_small_objects(masked_remaining_pixels > 0, param['min_size'])
                )
                sub_mask = small_object_filtered

            sub_comp_image[sub_mask > 0, idx] = 1
            remainder_mask -= sub_mask > 0

        remaining_pixels = np.multiply(mask, remainder_mask)
        sub_comp_image[remaining_pixels > 0, idx] = 1

        return sub_comp_image

    # ------------------------------------------------------------------
    # Feature computation (identical to FeatureExtractor)
    # ------------------------------------------------------------------

    def calculate_distance_transform_features(self, subcompartment_mask):
        feature_values = {}
        object_mask = np.sum(subcompartment_mask, axis=-1)
        object_area = np.sum(object_mask)

        for sc in range(len(self.sub_comp_names)):
            compartment_binary_mask = (subcompartment_mask[:, :, sc]).astype(np.uint8)
            sub_compartment_area = np.sum(compartment_binary_mask > 0)

            distance_transform = cv2.distanceTransform(compartment_binary_mask, cv2.DIST_L2, 5)
            distance_transform[distance_transform == 0] = np.nan

            sum_distance = np.nansum(distance_transform)
            mean_distance = np.nanmean(distance_transform)
            max_distance = np.nanmax(distance_transform)

            feature_values[f"Sum Distance Transform By Object Area {self.sub_comp_names[sc]}"] = (
                sum_distance / object_area if not np.isnan(sum_distance) and object_area > 0 else 0
            )
            feature_values[f"Sum Distance Transform By {self.sub_comp_names[sc]} Area"] = (
                sum_distance / sub_compartment_area if not np.isnan(sum_distance) and sub_compartment_area > 0 else 0
            )
            feature_values[f"Sum Distance Transform {self.sub_comp_names[sc]}"] = (
                sum_distance if not np.isnan(sum_distance) else 0
            )

            feature_values[f"Mean Distance Transform By Object Area {self.sub_comp_names[sc]}"] = (
                mean_distance / object_area if not np.isnan(mean_distance) and object_area > 0 else 0
            )
            feature_values[f"Mean Distance Transform By {self.sub_comp_names[sc]} Area"] = (
                mean_distance / sub_compartment_area if sub_compartment_area > 0 and not np.isnan(mean_distance) else 0
            )
            feature_values[f"Mean Distance Transform {self.sub_comp_names[sc]}"] = (
                mean_distance if not np.isnan(mean_distance) else 0
            )

            feature_values[f"Max Distance Transform By Object Area {self.sub_comp_names[sc]}"] = (
                max_distance / object_area if not np.isnan(max_distance) and object_area > 0 else 0
            )
            feature_values[f"Max Distance Transform By {self.sub_comp_names[sc]} Area"] = (
                max_distance / sub_compartment_area if not np.isnan(max_distance) and sub_compartment_area > 0 else 0
            )
            feature_values[f"Max Distance Transform {self.sub_comp_names[sc]}"] = (
                max_distance if not np.isnan(max_distance) else 0
            )

        return feature_values

    def calculate_color_features(self, image, subcompartment_mask):
        feature_values = {}
        for sc in range(len(self.sub_comp_names)):
            compartment_pixels = image[subcompartment_mask[:, :, sc] > 0]
            if len(compartment_pixels) > 0:
                mean_color = np.nanmean(compartment_pixels, axis=0)
                std_dev_color = np.nanstd(compartment_pixels, axis=0)
                for i, channel_value in enumerate(mean_color):
                    feature_values[f"Mean {['Red', 'Green', 'Blue'][i]} {self.sub_comp_names[sc]}"] = (
                        channel_value if not np.isnan(channel_value) else 0.0
                    )
                for i, channel_value in enumerate(std_dev_color):
                    feature_values[f"Standard Deviation {['Red', 'Green', 'Blue'][i]} {self.sub_comp_names[sc]}"] = (
                        channel_value if not np.isnan(channel_value) else 0.0
                    )
            else:
                for i in range(len(self.sub_comp_names)):
                    feature_values[f"Mean {['Red', 'Green', 'Blue'][i]} {self.sub_comp_names[sc]}"] = 0.0
                    feature_values[f"Standard Deviation {['Red', 'Green', 'Blue'][i]} {self.sub_comp_names[sc]}"] = 0.0
        return feature_values

    def calculate_texture_features(self, image, subcompartment_mask):
        feature_values = {}
        texture_feature_names = ['Contrast', 'Homogeneity', 'Correlation', 'Energy']
        for sc in range(len(self.sub_comp_names)):
            compartment_pixels = (subcompartment_mask[:, :, sc] > 0).astype(np.uint8)
            compartment_image = cv2.bitwise_and(image, image, mask=compartment_pixels)
            compartment_image_gray = rgb2gray(compartment_image)
            compartment_image_gray_uint = (compartment_image_gray * 255).astype(np.uint8)
            texture_matrix = graycomatrix(compartment_image_gray_uint, [1], [0],
                                          levels=256, symmetric=True, normed=True)
            for texture_name in texture_feature_names:
                val = graycoprops(texture_matrix, texture_name.lower())[0][0]
                feature_values[f"{texture_name} {self.sub_comp_names[sc]}"] = (
                    float(val) if not np.isnan(val) else 0.0
                )
        return feature_values

    def calculate_morphological_features(self, subcompartment_mask):
        feature_values = {}
        object_mask = np.sum(subcompartment_mask, axis=-1).astype(int)
        object_props = regionprops(object_mask)[0]
        object_area = np.sum(subcompartment_mask)

        for sc in range(len(self.sub_comp_names)):
            subcompartment_area = np.sum(subcompartment_mask[:, :, sc])
            feature_values[f"{self.sub_comp_names[sc]} Area By Object Area"] = (
                subcompartment_area / object_area if object_area > 0 else 0.0
            )
            feature_values[f"{self.sub_comp_names[sc]} Area"] = subcompartment_area

        nuclei_idx = self.sub_comp_names.index('Nuclei')
        nuclei_number = np.max(label(subcompartment_mask[:, :, nuclei_idx]))
        feature_values['Nuclei Number'] = nuclei_number

        nuclei_label = label(subcompartment_mask[:, :, nuclei_idx])
        nuclei_props = regionprops(nuclei_label)
        aspect_ratios = [
            p.axis_major_length / p.axis_minor_length if p.axis_minor_length > 0 else 0
            for p in nuclei_props
        ]
        feature_values['Mean Aspect Ratio Nuclei'] = (
            np.nanmean(aspect_ratios) if not np.isnan(np.nanmean(aspect_ratios)) else 0
        )
        feature_values['Standard Deviation Aspect Ratio Nuclei'] = (
            np.nanstd(aspect_ratios) if not np.isnan(np.nanstd(aspect_ratios)) else 0
        )
        feature_values['Mean Nuclear Area'] = (
            np.sum(subcompartment_mask[:, :, nuclei_idx]) / nuclei_number if nuclei_number > 0 else 0
        )

        feature_values['Total Object Area'] = object_area
        feature_values['Total Object Perimeter'] = object_props.perimeter
        feature_values['Total Object Aspect Ratio'] = (
            object_props.axis_major_length / object_props.axis_minor_length
            if object_props.axis_minor_length != 0 else 0
        )
        feature_values['Major Axis Length'] = object_props.axis_major_length
        feature_values['Minor Axis Length'] = object_props.axis_minor_length

        return feature_values

    def aggregate_features(self, feature_df):
        summ_stats = {
            'Sum': lambda a: np.nansum(a, axis=0),
            'Mean': lambda a: np.nanmean(a, axis=0),
            'Standard Deviation': lambda a: np.nanstd(a, axis=0),
            'Median': lambda a: np.nanmedian(a, axis=0),
            'Minimum': lambda a: np.nanmin(a, axis=0),
            'Maximum': lambda a: np.nanmax(a, axis=0),
        }
        agg_feat_dict = {}
        feature_df.drop(columns=['compartment_ids'], inplace=True)
        feature_names = feature_df.columns.tolist()
        feature_values = feature_df.values

        for s in summ_stats:
            stats_array = summ_stats[s](feature_values.copy())
            for feat, summ in zip(feature_names, stats_array.tolist()):
                if not np.isnan(summ) and not np.isinf(summ):
                    agg_feat_dict[f'{feat}_{s}'] = np.float64(summ)
                else:
                    agg_feat_dict[f'{feat}_{s}'] = np.float64(0.)
        return agg_feat_dict

    # ------------------------------------------------------------------
    # Local I/O (replaces Girder post/delete)
    # ------------------------------------------------------------------

    def save_annotations(self):
        """Save updated annotations (with per-element user features) to a JSON file."""
        out_path = os.path.join(self.output_path, 'annotations_updated.json')
        with open(out_path, 'w') as fh:
            json.dump(self.annotations, fh, indent=2, default=lambda x: float(x))
        print(f'Updated annotations written to {out_path}')
