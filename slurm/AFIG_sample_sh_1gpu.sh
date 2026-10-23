#!/bin/bash
#SBATCH --job-name=AFIG
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8   # one process on one GPU; extra CPUs only speed up XLA compilation of each new length
#SBATCH --mem=40G
##SBATCH --partition=gpu     # uncomment and adjust if your cluster needs a partition / account
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log


### PATHS (absolute)
OUT_DIR="${HOME}/AFIG/output"     # <name>_af2pred.pdb, the score file, check.point. PDBs that already have a <name>_af2pred.pdb here (or in OTHER_OUTPUT_DIRS) are skipped, the rest is run and appended to the score file
SCRIPT_DIR="${HOME}/AFIG/script"  # copy of the job log (updated every 30 s); must not be OUT_DIR or inside it

### CONFIGURATION
INPUT_DIR="${HOME}/AFIG/input"    # folder of .pdb files (ligands, waters and other HETATM records in them are ignored; files are not modified)
BINDER_CHAIN=""      # chain ID of the binder; empty = first chain of the first PDB. It may sit anywhere in a file
TARGET_CHAINS=""     # chain ID(s) the binder is scored against, e.g. "B" or "B,C"; empty = all other chains. Other chains are dropped
SCORE_FILE_NAME="out.sc"  # score file in OUT_DIR; a name ending in .csv or .tsv gives a table. Columns follow the chain IDs, e.g. binder A, targets B,C: ipSAE_AB ipSAE_BA ipSAE_AB_max ipSAE_AC ipSAE_CA ipSAE_AC_max ... (see README)
BATCH_SIZE=10,all    # batches: a pilot of ~10 structures (checked), then the rest in one process; or e.g. 50 (batches of ~50) or all
OTHER_OUTPUT_DIRS=() # absolute paths of earlier output folders whose finished PDBs are skipped, e.g. ( "${HOME}/AFIG/output_run1" )
AFIG_ARGS=(
    -calc_ipsae true             # compute ipSAE for each binder-target pair, both directions and max (script default: false)
    -ipsae_cutoff 10             # PAE cutoff in A used by ipSAE (default 10)
    -recycle 3                   # AF2 recycles (default 3)
    -max_amide_dist 3.0          # peptide bonds longer than this (A) are treated as chain breaks (default 3.0)
    -strict_binder_check true    # abort a structure whose binder is not one contiguous chain, e.g. TER or a repeated chain ID inside it (default true)
    -ignore_ligands true         # ignore ligands, waters, other HETATM and CONECT records; MSE is kept (default true)
    -checkpoint_failures false   # true: failed structures are also checkpointed and not retried on resubmit (default false)
    # -no_initial_guess          # do not use the input structure as initial guess
    # -force_monomer             # predict only the first chain (the binder) as a monomer
    # -maintain_res_numbering    # error instead of renumbering when residue numbers repeat across chains
    # -debug                     # stop the batch on an error and print it, instead of skipping that structure
)   # not allowed here (the wrapper sets them): -pdbdir -outpdbdir -scorefilename -checkpoint_name -binder_chain -target_chains -allow_missing_targets -silent -outsilent -runlist

# Wrapper: job I/O, environment, input check, batching, runs predict_ipsae_multitarget.py
# Python environment (JAX, PyRosetta, dm-haiku; see include/af2_binder_design2.yml); `python` must be on PATH. Example:
# source ~/miniconda3/etc/profile.d/conda.sh; conda activate af2_binder_design2
# This repository, with the AF2 weights in af2_initial_guess/model_weights/params:
# export DL_BINDER_DESIGN_ROOT="${HOME}/dl_binder_design"
AFIG_WRAPPER="${HOME}/dl_binder_design/slurm/AFIG_wrapper.sh"   # edit: where AFIG_wrapper.sh is
source "$AFIG_WRAPPER"
