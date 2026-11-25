"""
Fidelity analysis worked into the utility analysis frame work
Calculates the following univariate and bivariate fiedlity metrics across runs and plots the means and standard deviations
Univariate: KS and TVD
Bivariate: Corrilation similarity and Contingency simialrity 
"""
from pathlib import Path
from random import shuffle
import numpy as np
import pandas as pd
from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
from sdmetrics.single_column import KSComplement, TVComplement

from src import load_real, load_split_synthetic

NAME = "evaluate_fidelity"

def main(cfg, dpath):
    return 0


if __name__ == "__main__":
    main()