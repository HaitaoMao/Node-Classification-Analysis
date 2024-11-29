import numpy as np
from utils.args import *
from utils.data_utils import *
import json
import pickle
from output_analyze.analyze_output import main_output_analysis
from output_analyze.analyze_fairness import main_fair_analysis
from output_analyze.analyze_fairness2 import main_fair_analysis2
from output_analyze.analyze_fairness3 import main_fair_analysis3
from output_analyze.analyze_fairness4 import main_fair_analysis4
from output_analyze.analyze_fairness5 import main_fair_analysis5
from output_analyze.analyze_fairness6 import main_fair_analysis6
from output_analyze.analyze_fairness7 import main_fair_analysis7

from output_analyze.analyze_ood import main_analysis_ood
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0)
    
    args = parser.parse_args()
    if args.mode == 0:
        main_output_analysis()
    if args.mode == 1:
        main_fair_analysis()
    elif args.mode == 2:
        main_fair_analysis2()
    elif args.mode == 3:
        main_fair_analysis3()
    elif args.mode == 4:
        main_analysis_ood()
    elif args.mode == 5:
        main_fair_analysis4()
    elif args.mode == 6:
        main_fair_analysis5()
    elif args.mode == 7:
        main_fair_analysis6()
    elif args.mode == 8:
        main_fair_analysis7() 
    else:
        main_output_analysis()

        
def main2():


if __name__ == "__main__":
    main()
        

