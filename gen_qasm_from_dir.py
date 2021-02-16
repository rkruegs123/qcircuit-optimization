import os
import logging
import random
import matplotlib.pyplot as plt
import pdb
import argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

import sys
sys.path.append('../pyzx')
import pyzx as zx


parser = argparse.ArgumentParser(description='Arguments for generating a directory of circuit benchmarks in QASM format from an existing directory of circuit files (any format)')
parser.add_argument('--outdir', type=str, help='The directory to store QASM files in', required=True)
parser.add_argument('--indir', type=str, help='The directory containing circuit files in any format to convert to QAMS and store in OUTDIR', required=True)
args = parser.parse_args()

idir = args.indir
odir = args.outdir

# check that input directory exists
if not os.path.exists(idir):
    sys.exit("ERROR: input directory doesn't exist")

# check that output directory doesn't already exist
if os.path.exists(odir):
    sys.exit("ERROR: output directory already exists")

# create new output directory
os.makedirs(odir)

infiles = [f for f in listdir(idir) if isfile(join(idir, f))]

for cfile in tqdm(infiles):
    if cfile.count('.') > 1:
        print(f"WARNING: file {cfile} may have more than one extension")
        pdb.set_trace()

    cfile_no_ext = os.path.splitext(cfile)[0]
    cfile_qasm = cfile_no_ext + ".qasm"
    c = zx.Circuit.load(join(idir, cfile))
    with open(join(odir, cfile_qasm), 'w') as f:
        f.write(c.to_qasm())
