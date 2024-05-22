import os
import argparse
import sys
import time

sys.path.append('..')

class InputType(Enum):
    pathomic = 'pathomic'
    extended_clinical = 'extended_clinical'

def main(args):

    from extractioncodes.Codes.extract_extended_clinical_features import getExtendedClinicalFeatures


    if args.type == InputType.pathomic.value:
        getPathomicFeatures(args=args)
    elif args.type == InputType.extended_clinical.value:
        getExtendedClinicalFeatures(args=args)
    else:
        print('please specify an option in: \n\t--option [get_extended_clinical_features, get_pathomic_features]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--girderApiUrl', dest='girderApiUrl', default=' ' ,type=str,
        help='girderApiUrl')
    parser.add_argument('--girderToken', dest='girderToken', default=' ' ,type=str,
        help='girderToken')
    parser.add_argument('--option', dest='option', default=' ' ,type=str,
        help='option for [get_extended_clinical_features, get_pathomic_features]')
    parser.add_argument('--wsi_xml', dest='wsi_xml', default=' ' ,type=str,
        help='path to input wsi and xml file')


    args = parser.parse_args()
    main(args=args)