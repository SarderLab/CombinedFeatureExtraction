import numpy as np
import cv2
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.color import rgb2gray


def calculate_distance_transform_features(compartment_mask):
    # Function to calculate distance transform features for each compartment
    feature_values = {}
    compartment_names = ['Luminal space compartment', 'PAS compartment', 'Nuclei compartment']

    for compartment in range(3):  # As there are 3 compartments
        compartment_binary_mask = (compartment_mask == (compartment + 1)).astype(np.uint8)
        distance_transform = cv2.distanceTransform(compartment_binary_mask, cv2.DIST_L2, 5)

        # Sum Distance Transform By Object Area
        sum_distance_by_object_area = np.sum(distance_transform)
        feature_values[f"Sum Distance Transform By Object Area {compartment_names[compartment]}"] = sum_distance_by_object_area

        # Sum Distance Transform By Compartment Area
        compartment_area = np.count_nonzero(compartment_mask == (compartment + 1))
        sum_distance_by_compartment_area = np.sum(distance_transform) / compartment_area
        feature_values[f"Sum Distance Transform By Compartment Area {compartment_names[compartment]}"] = sum_distance_by_compartment_area

        # Sum Distance Transform
        total_pixels = compartment_mask.shape[0] * compartment_mask.shape[1]
        sum_distance = np.sum(distance_transform) / total_pixels
        feature_values[f"Sum Distance Transform {compartment_names[compartment]}"] = sum_distance

        # Mean Distance Transform By Object Area
        mean_distance_by_object_area = np.mean(distance_transform)
        feature_values[f"Mean Distance Transform By Object Area {compartment_names[compartment]}"] = mean_distance_by_object_area

        # Mean Distance Transform By Compartment Area
        mean_distance_by_compartment_area = np.mean(distance_transform) / compartment_area
        feature_values[f"Mean Distance Transform By Compartment Area {compartment_names[compartment]}"] = mean_distance_by_compartment_area

        # Mean Distance Transform
        mean_distance = np.mean(distance_transform) / total_pixels
        feature_values[f"Mean Distance Transform {compartment_names[compartment]}"] = mean_distance

        # Max Distance Transform By Object Area
        max_distance_by_object_area = np.max(distance_transform)
        feature_values[f"Max Distance Transform By Object Area {compartment_names[compartment]}"] = max_distance_by_object_area

        # Max Distance Transform By Compartment Area
        max_distance_by_compartment_area = np.max(distance_transform) / compartment_area
        feature_values[f"Max Distance Transform By Compartment Area {compartment_names[compartment]}"] = max_distance_by_compartment_area

        # Max Distance Transform
        max_distance = np.max(distance_transform) / total_pixels
        feature_values[f"Max Distance Transform {compartment_names[compartment]}"] = max_distance

    return feature_values


def calculate_color_features(image, compartment_mask):
    # Function to calculate color features for each compartment
    feature_values = {}
    compartment_names = ['Luminal space compartment', 'PAS compartment', 'Nuclei compartment']

    for compartment in range(3):  # As there are 3 compartments
        compartment_pixels = image[compartment_mask == (compartment + 1)]

        if len(compartment_pixels) > 0:
            # Mean Color
            mean_color = np.mean(compartment_pixels, axis=0)
            for i, channel_value in enumerate(mean_color):
                feature_values[f"Mean {['Red', 'Green', 'Blue'][i]} {compartment_names[compartment]}"] = channel_value

            # Standard Deviation Color
            std_dev_color = np.std(compartment_pixels, axis=0)
            for i, channel_value in enumerate(std_dev_color):
                feature_values[f"Standard Deviation {['Red', 'Green', 'Blue'][i]} {compartment_names[compartment]}"] = channel_value
        else:
            # If compartment has no pixels, set values to zero
            for i in range(3):
                feature_values[f"Mean {['Red', 'Green', 'Blue'][i]} {compartment_names[compartment]}"] = 0.0
                feature_values[f"Standard Deviation {['Red', 'Green', 'Blue'][i]} {compartment_names[compartment]}"] = 0.0

    return feature_values


def calculate_texture_features(image, compartment_mask):
    # Function to calculate texture features for each compartment
    feature_values = {}
    texture_feature_names = ['Contrast', 'Homogeneity', 'Correlation', 'Energy']
    compartment_names = ['Luminal space compartment', 'PAS compartment', 'Nuclei compartment']

    for compartment in range(3):  # As there are 3 compartments
        compartment_pixels = (compartment_mask == compartment+1).astype(np.uint8)
        compartment_image = cv2.bitwise_and(image, image, mask=compartment_pixels)
        compartment_image_gray = rgb2gray(compartment_image)
        compartment_image_gray_uint = (compartment_image_gray * 255).astype(np.uint8)
        texture_matrix = graycomatrix(compartment_image_gray_uint, [1], [0], levels=256, symmetric=True, normed=True)

        for i, texture_name in enumerate(texture_feature_names):
            texture_feature_value = graycoprops(texture_matrix, texture_name.lower())
            feature_values[f"{texture_name} {compartment_names[compartment]}"] = texture_feature_value[0][0]

    return feature_values



def calculate_morphological_features(compartment_mask):
    # Function to calculate morphological features for each compartment
    feature_values = {}
    compartment_names = ['Luminal space compartment', 'PAS compartment', 'Nuclei compartment']

    # Calculate region properties for the entire image (all compartments combined)
    regions_total = measure.regionprops(compartment_mask)
    total_object_area = 0
    total_object_perimeter = 0
    aspect_ratios_total = []
    major_axis_lengths = []
    minor_axis_lengths = []
    nuclei_area = 0

    for region in regions_total:
        total_object_area += region.area
        total_object_perimeter += region.perimeter
        aspect_ratio = region.major_axis_length / region.minor_axis_length
        if aspect_ratio > 0:
            aspect_ratios_total.append(aspect_ratio)
        major_axis_lengths.append(region.major_axis_length)
        minor_axis_lengths.append(region.minor_axis_length)

        if region.label == 3:  # Nuclei compartment label is 3
            nuclei_area += region.area

    # Compartment Area By Object Area and Compartment Area
    object_area = np.count_nonzero(compartment_mask)
    for compartment in range(3):  # As there are 3 compartments
        compartment_area = np.count_nonzero(compartment_mask == (compartment + 1))
        normalized_compartment_area_by_object_area = compartment_area / object_area
        normalized_compartment_area = compartment_area / (compartment_mask.shape[0] * compartment_mask.shape[1])
        feature_values[f"Compartment Area By Object Area {compartment_names[compartment]}"] = normalized_compartment_area_by_object_area
        feature_values[f"Compartment Area {compartment_names[compartment]}"] = normalized_compartment_area

    # Calculate Nuclei Number
    nuclei_number = np.count_nonzero(compartment_mask == 3)  # Count nuclei objects
    feature_values[f"Nuclei Number {compartment_names[2]}"] = nuclei_number

    # Calculate Mean Aspect Ratio and Standard Deviation Aspect Ratio for the nuclei compartment
    mean_aspect_ratio = np.mean(aspect_ratios_total)
    std_dev_aspect_ratio = np.std(aspect_ratios_total)
    feature_values[f"Mean Aspect Ratio {compartment_names[2]}"] = mean_aspect_ratio
    feature_values[f"Standard Deviation Aspect Ratio {compartment_names[2]}"] = std_dev_aspect_ratio

    # Calculate Mean Nuclear Area for the nuclei compartment
    feature_values[f"Mean Nuclei Area {compartment_names[2]}"] = nuclei_area

    # Total Object Area
    feature_values["Total Object Area Total compartment"] = total_object_area

    # Total Object Perimeter
    feature_values["Total Object Perimeter Total compartment"] = total_object_perimeter

    # Total Object Aspect Ratio
    mean_aspect_ratio_total = np.mean(aspect_ratios_total)
    feature_values["Total Object Aspect Ratio Total compartment"] = mean_aspect_ratio_total

    # Major Axis Length
    mean_major_axis_length = np.mean(major_axis_lengths)
    feature_values["Major Axis Length Total compartment"] = mean_major_axis_length

    # Minor Axis Length
    mean_minor_axis_length = np.mean(minor_axis_lengths)
    feature_values["Minor Axis Length Total compartment"] = mean_minor_axis_length

    return feature_values


def calculate_features(image, compartment_mask):
    # Function to calculate all features (color and texture) for each compartment
    distance_transform_features_dict = calculate_distance_transform_features(compartment_mask)
    color_features_dict = calculate_color_features(image, compartment_mask)
    texture_features_dict = calculate_texture_features(image, compartment_mask)
    morphological_features_dict = calculate_morphological_features(compartment_mask)
    all_features = {
        "Distance Transform Features": distance_transform_features_dict,
        "Color Features": color_features_dict,
        "Texture Features": texture_features_dict,
        "Morphological Features": morphological_features_dict
    }

    return all_features


