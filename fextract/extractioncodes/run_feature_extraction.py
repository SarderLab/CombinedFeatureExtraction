import os
import argparse
import sys
import time
from enum import Enum

sys.path.append('..')

from .extract_reference_features import getExtendedClinicalFeatures
from .run_pathomic_fe import getPathomicFeatures

class InputType(Enum):
    pathomic = 'Pathomic'
    extended_clinical = 'Extended_Clinical'

def main(args):

    if args.type == InputType.pathomic.value:
        getPathomicFeatures(args=args)
    elif args.type == InputType.extended_clinical.value:
        getExtendedClinicalFeatures(args=args)
    else:
        print('please specify an option in: \n\t--option [get_extended_clinical_features, get_pathomic_features]')


def run_main(args):
    main(args)
