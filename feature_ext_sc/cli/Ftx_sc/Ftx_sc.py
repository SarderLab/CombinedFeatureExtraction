import os
import sys
from ctk_cli import CLIArgumentParser


sys.path.append("..")


def main(args):  
    
    cmd = "python3 ../ftx_sc_code/FeatureExtractor.py   --basedir {} --girderApiUrl {} --girderToken {} \
             --input_image {} --threshold_nuclei {} --minsize_nuclei {} --threshold_PAS {} --minsize_PAS {} --threshold_LAS {} --minsize_LAS {} \
                ".format(args.basedir, args.girderApiUrl, args.girderToken, args.input_image, args.threshold_nuclei, args.minsize_nuclei, args.threshold_PAS, args.minsize_PAS, args.threshold_LAS, args.minsize_LAS)
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)  


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())

