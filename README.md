# dl_binder_design

> **This is a fork** of [WEHI-ResearchComputing/dl_binder_design](https://github.com/WEHI-ResearchComputing/dl_binder_design)
> at commit [`ab173bc`](https://github.com/WEHI-ResearchComputing/dl_binder_design/commit/ab173bcd918d47d70145af492451c90dfc10c27d)
> (itself a fork of [nrbennet/dl_binder_design](https://github.com/nrbennet/dl_binder_design)). The original README follows below this section.
> Changes are listed here and in the git history (`git log ab173bc..HEAD`); to see exactly what changed in a file, run e.g. `git diff ab173bc -- af2_initial_guess/predict.py`, and `git show ab173bc:<path>` gives the upstream file. MIT licence unchanged.

This repo contains the scripts described in the paper [Improving de novo Protein Binder Design with Deep Learning](https://www.nature.com/articles/s41467-023-38328-5).

![forgithub](https://github.com/nrbennet/dl_binder_design/assets/56419265/4c5d6a05-d2fb-4c7b-b1e0-0c743b2114eb)

# Table of Contents

* [Changes in this fork](#changes)
  * [Score file columns](#scorefile)
* [Third Party Source Code](#sourcecode)
* [Setup](#setup0)
  * [Conda Environment](#setup1)
    * [Install ProteinMPNN-FastRelax Environment](#setup1.1)
    * [Install AlphaFold2 Environment](#setup1.2)
    * [Troubleshooting AF2 GPU Compatibility](#setup1.3)
  * [Clone ProteinMPNN](#setup2)
  * [Explanation of Silent Tools](#setup3)
  * [Download AlphaFold2 Model Weights](#setup4)
* [Inference](#inf0)
  * [ProteinMPNN-FastRelax Binder Design](#inf1)
    * [Running ProteinMPNN with Fixed Residues](#inf2)
  * [AlphaFold2 Complex Prediction](#inf3)
* [Troubleshooting](#trb0)
* [Credits](#credits)

# Changes in this fork <a name="changes"></a>

| File | Change |
|-|-|
| `af2_initial_guess/predict_ipsae_multitarget.py` (new) | Same prediction as `predict.py`, plus ipSAE scoring in both directions, any number of target chains, and selection of the binder and target chains by chain ID (`-binder_chain`, `-target_chains`). Ligands, waters and other HETATM/CONECT records in PDB inputs are ignored (`-ignore_ligands`). Chain IDs and residue numbers of the input are kept on the output PDB. Reruns with `-pdbdir` skip structures that are already finished. Other options: `-sort_by_length`, `-checkpoint_failures`, `-strict_binder_check`. Details and the score file columns are below. |
| `af2_initial_guess/predict.py` (modified) | The first target residue is now part of the template (upstream leaves it out: `i > binderlen` is now `i >= binderlen`), so scores differ slightly from upstream. Complexes with more than two chains are accepted: the first chain is the binder, the rest is the target (the output PDB merges the target chains; `predict_ipsae_multitarget.py` keeps them). A failed structure prints the error type, message and traceback. |
| `include/af2_binder_design.yml` (modified) | Pinned to Python 3.10, JAX / jaxlib 0.4.20 (cuda12.cudnn89), dm-haiku 0.0.11, dm-tree 0.1.9, TensorFlow 2.15.1, ml-dtypes 0.3.2. |
| `include/af2_binder_design2.yml` (new) | As above, plus CUDA 12.4 user-space libraries (nvcc, runtime, nvrtc, cupti, cuDNN 8.9), so that XLA uses a `ptxas` that matches NVIDIA drivers supporting CUDA 12.4. These are the pins of the environment we run; creating an environment from this file alone has not been tested. PyRosetta needs its own licence and conda channel. |
| `helper_scripts/rmvlig.py` (new) | Copies a folder of PDBs (recursively, keeping sub-folders) while dropping `HETATM` and `CONECT` lines, for when you want cleaned copies on disk. dl_binder_design has no ligand handling (the only HETATM record it converts is MSE). `predict_ipsae_multitarget.py` ignores these records itself, so cleaning is not needed. This script also removes MSE residues. |
| `slurm/` (new) | Example Slurm job script and wrapper that we use on our cluster: input check, length-sorted batches, resume, chain selection, score file naming. See `slurm/README.md`. |

## Using `predict_ipsae_multitarget.py`

Use the AF2 environment (`include/af2_binder_design2.yml`, see Setup below for PyRosetta and the weights).
```bash
# binder = chain A, scored against chains B and C; any other chain in the files is dropped
python af2_initial_guess/predict_ipsae_multitarget.py -pdbdir inputs/ -outpdbdir outputs/ \
       -binder_chain A -target_chains B,C -calc_ipsae true -ipsae_cutoff 10 -scorefilename scores.csv
# without the chain options: binder = first chain of the first file, targets = all other chains found in the inputs
```
- Inputs are PDB files (ATOM records of the first model). The binder may sit anywhere in a file; it is moved to the front before prediction. With `-silent` instead of `-pdbdir`, the binder must already be the first chain, both chain options are required and no other chains may be present.
- `-strict_binder_check` (default true) rejects a structure whose binder is split into several blocks, i.e. a `TER` record or another chain between its atoms.
- If residue numbers repeat across chains (binder 1-80, target 1-150), the structure is renumbered internally for AF2; the output PDB gets your numbers back. Chain IDs are never renamed.
- True/false options take the words `true` / `false` (`-calc_ipsae` is `false` unless you set it).

## Score file columns <a name="scorefile"></a>

One row per structure is appended to `-scorefilename`. The default `out.sc` has the Rosetta-style `SCORE:` format; a name ending in `.csv` or `.tsv` writes a comma / tab separated table instead (first column `description`), which is easier to read in a spreadsheet or pandas. A file that already exists must have the same columns, otherwise the script stops before predicting anything (use another file name).

Column names follow the chain IDs of your input: with binder chain `B` and target chain `X`, "BX" means *residues of B scored against X* (the first letter is the chain whose residues are scored, as in Dunbrack's `ipsae.py`). The columns are decided once per run, from `-binder_chain` / `-target_chains` or, by default, from the files in `-pdbdir`. Any number of targets works. Columns in file order, for binder A and targets B, C, D:

| Column(s) | Meaning |
|-|-|
| `binder_chain`, `target_chains` | chain IDs used for this structure (e.g. `A`, `B,C,D`) |
| `ipSAE_AB`, `ipSAE_BA`, `ipSAE_AB_max`, then `ipSAE_AC`, `ipSAE_CA`, `ipSAE_AC_max`, `ipSAE_AD`, ... | per target: ipSAE of the binder residues against the target (`AB`), of the target residues against the binder (`BA`), and the larger of the two (`_max`, the `max` value of `ipsae.py`) |
| `iPAE_AB`, `iPAE_AC`, `iPAE_AD`, ... | per target: mean inter-chain PAE of the binder-target pair, both directions averaged (for one target this is upstream's `pae_interaction`) |
| `n_iface_AB`, `n_iface_BA`, `n_iface_AC`, `n_iface_CA`, ... | per target, both directions: number of residues of the first-named chain that have at least one partner residue with PAE below `-ipsae_cutoff` (0 = no interface) |
| `plddt_total`, `plddt_binder`, `plddt_B`, `plddt_C`, ..., `plddt_target` | mean pLDDT of everything, of the binder, of each target, of all targets together |
| `pae_binder`, `pae_B`, `pae_C`, ..., `pae_target`, `pae_cross_targets` | mean PAE within the binder, within each target, within all targets together (including the target-target blocks), and between different targets (targets are not pre-associated, so high values are expected; `nan` with one target) |
| `binder_aligned_rmsd`, `target_aligned_rmsd` | RMSD of the prediction to the input after aligning on the binder / on the target |
| `target_drift_rmsd` | change of the arrangement of the first two targets between input and prediction (`nan` with one target) |
| `n_targets` | number of target chains in this structure |
| `time` | seconds spent on this structure |
| `description` | tag of the structure (`<input name>_af2pred`); first column in `.csv`/`.tsv`, last in `.sc` |

- The ipSAE and `n_iface` columns are `nan` unless `-calc_ipsae true`.
- A target that a structure lacks gives `nan` in all its columns. This applies when the targets are derived from the inputs (no `-target_chains`) or with `-allow_missing_targets`; with an explicit `-target_chains`, every input must contain all of them, otherwise that structure fails.
- With one target only the `AB` columns are filled; `AC`, `AD`, ... exist only if more targets are in the column layout. Binder `H` with targets `L` and `M` gives `ipSAE_HL`, `ipSAE_LH`, `ipSAE_HL_max`, `ipSAE_HM`, `ipSAE_MH`, `ipSAE_HM_max`.
- Derived targets depend on the files in `-pdbdir`: two runs on different subsets can give different columns, and the second is refused. Pass `-binder_chain` / `-target_chains` explicitly, or use the Slurm wrapper, which fixes the chains for the whole run.
- Run one process per score file; there is no locking.

## Third Party Source Code <a name="sourcecode"></a>
This repository provides a copy of Brian Coventry's silent tools, these tools are also provided [here](https://github.com/bcov77/silent_tools). This repository provides a wrapper to [Justas' ProteinMPNN code](https://github.com/dauparas/ProteinMPNN) and some of the code in the wrapper class is adapted directly from ProteinMPNN code. Finally, this repository provides a version of the AlphaFold2 source code with the "initial guess" modifications described in [this paper](https://www.nature.com/articles/s41467-023-38328-5). The AF2 source code is provided with the original DeepMind license at the top of each file.

# Setup <a name="setup0"></a>

## Conda Environment <a name="setup1"></a>

I have split the single conda env that this repo used to use (dl_binder_design.yml; still provided in <base_dir>/include if anyone is interested) into two smaller and easier to install environments. The old environment required PyTorch, JAX, TensorFlow, and PyRosetta packages to all be compatible with one another, which is difficult. The new environments should be easier to install and I have also added import tests so that it is easier and faster to check that the installation has been successful.

The steps to installing the environments are as follows:

- Ensure that you have the Anaconda or Miniconda package manager
- Clone this repo

## Install ProteinMPNN-FastRelax Environment <a name="setup1.1"></a>
- Navigate to <base_dir>/include
- Run `conda env create -f proteinmpnn_fastrelax.yml`
- Test the environment by activating your environment and running `python importtests/proteinmpnn_importtest.py`, if you encounter an error in this script then something has gone wrong with your installation. If the script prints out that the tests pass then you have installed this environment correctly. This script will also test whether it can access a GPU, it is not recommended to run ProteinMPNN with a GPU as it is only marginally faster than running on CPU and FastRelax cannot take advantage of the GPU anyway.

## Install AlphaFold2 Environment <a name="setup1.2"></a>
- Navigate to <base_dir>/include
- Run `conda env create -f af2_binder_design.yml`
- Test the environment by activating your environment and running `python importtests/af2_importtest.py`. Run this script from a node which has acccess to the GPU you wish to use for AF2 inference, this script will print out a message about whether it was able to find and use the GPU on your node. If this script hits an error before printing anything then the installation has not been done correctly.

### Troubleshooting AF2 GPU Compatibility <a name="setup1.3"></a>

Getting a conda environment that recognizes and can run AF2 on your GPU is one of the more difficult parts of this process. Because of the many different GPUs out there, it is not possible for us to provide one .yml file that will work with all GPUs. We provide a CUDA 11.1 compatible env (dl_binder_design.yml) and a CUDA 12 compatible env (af2_binder_design.yml). For other versions of the CUDA driver, you may need to change which CUDA version is installed in the conda env, this can be done by changing the CUDA version in this line of af2_binder_design.yml: `- jax[cuda12_pip]`.

NOTE: This conda environment can only accomodate NVIDIA GPUs at this point in time.

## Clone ProteinMPNN <a name="setup2"></a>
This repo requires the code from ProteinMPNN to work. It expects this code to be in the mpnn_fr directory so we can just clone it to be there
- Naviate to <base_dir>/mpnn_fr
- Run `git clone https://github.com/dauparas/ProteinMPNN.git`

## Silent Tools <a name="setup3"></a>
The scripts contained in this repository work with a type of file called silent files. These are essentially a bunch of compressed .pdb files that are all stored in one file. Working with silent files is conventient and saves a lot of disk space when dealing with many thousands of structures.

Brian Coventry wrote a bunch of really nice commandline tools (called silent_tools) to manipulate silent files. These tools are included in this repository but may also be downloaded separately from [this](https://github.com/bcov77/silent_tools) GitHub repo.

The two commands that allow you to go from pdb to silent file and back are the following:

pdbs to silent: `<base_dir>/silentfrompdbs *.pdb > my_designs.silent` 

silent to pdbs: `<base_dir>/silentextract all_structs.silent`

NOTE: Some silent tools require PyRosetta and will fail if run in a Python environment without access to PyRosetta.

## Download AlphaFold2 Model Weights <a name="setup4"></a>
The scripts in this repository expect AF2 weights to be in <base_dir>/model_weights/params and will fail to run if the weights are not there. If you already have AF2 params_model_1_ptm.npz weights downloaded then you may simply copy them to <base_dir>/model_weights/params or create a symlink. If you do not have these weights downloaded you will have to download them from DeepMind's repository, this can be done as follows:

```
cd <base_dir>/af2_initial_guess
mkdir -p model_weights/params && cd model_weights/params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar --extract --verbose --file=alphafold_params_2022-12-06.tar 
```

# Inference <a name="inf0"></a>

## Summary <a name="inf0.5"></a>

The binder design pipeline requires protein binder backbones as an input. The recommended way to generate these backbones is to use [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), which will give you a directory of .pdb files. These files can either be turned into silent files which are more memory efficient and easier to work with than a directory of pdb files (and your system administrator will thank you for using them), or you can use these directories of .pdb files as-is with this pipeline, using either the `-pdbdir` flag alone or the `-pdbdir` flag in combination with the `-runlist` flag.

## Example Commands <a name="inf0.5"></a>

Example commands demonstrating how to run each of these scripts with different types of input can be found here:

```
<base_dir>/examples
```

## ProteinMPNN-FastRelax Binder Design <a name="inf1"></a>

Here is an example of how to run ProteinMPNN-FastRelax with a silent file of designs. This will use the default of 1 FastRelax cycle of 1 ProteinMPNN sequence per round (NOTE: running with -relax_cycles > 0 and -seqs_per_struct > 1 is disabled as it leads to an explosion in the amount of FastRelax trajectories being run and is probably bad idea):

```
<base_dir>/mpnn_fr/dl_interface_design.py -silent my_designs.silent
```

This will create a file titled `out.silent` containing your designs. This file can be fed directly to AF2 interface prediction.

With the refactor, this script is now able to read and write both PDB files and silent files. The script I have also added more informative argument messages which can be accessed by running:

```
<base_dir>/mpnn_fr/dl_interface_design.py -h
```

NOTE: This script expects your binder design to be the first chain it receives. This script is robust to non-unique indices, unlike the AF2 interface script.
NOTE 2: The outputs of this script do not have relaxed sidechains (since sidechains are not input to AF2 and it's not worth the computation to relax them) so the structures will look strange if you visualize them in PyMol, this is perfectly normal, the structures will look better after run though AF2.

### Running ProteinMPNN with Fixed Residues <a name="inf2"></a>

If you used RFdiffusion to generate your binder designs and would like to fix a region, you can use the following command to add 'FIXED' labels to your pdbs which will be recognized by the ProteinMPNN scripts. Thanks to Preetham Venkatesh for writing this!

```
python <base_dir>/helper_scripts/addFIXEDlabels.py --pdbdir /dir/of/pdbs --trbdir /dir/of/trbs --verbose
```

These pdb files can be collected into a silent file (or just used as PDB files) and run through the ProteinMPNN script which will detect the FIXED labels and keep those sequence positions fixed.


## AlphaFold2 Complex Prediction <a name="inf3"></a>

Running the interface prediction script is simple:

```
<base_dir>/af2_initial_guess/predict.py -silent my_designs.silent
```

This will create a file titled `out.silent` containing the AF2 predictions of your designs. It will also output a file titled `out.sc` with the scores of the designs, `pae_interaction` is the score that showed the most predictivity in the experiments performed in the paper.

With the refactor, this script is now able to read and write PDB and silent files as well as perform both monomer and complex predictions. The arguments for these can be listed by running:

```
<base_dir>/af2_initial_guess/predict.py -h
```

NOTE: This script expects your binder design to be the first chain it receives. The binder will be predicted from single sequence and with an intial guess. The target chains will be fixed to the input structure. The script also expects your residue indices to be unique, ie. your binder and target cannot both start with residue 1.

# Troubleshooting <a name="trb0"></a>

One of the most common errors that people have been having is one that looks like this:

```
Struct with tag SAMETAG failed in 0 seconds with error: <class 'EXCEPTION'>
```

Where SAMETAG and EXCEPTION can be many different things. What is happening here is that the main loops of both of the scripts provided here are wrapped in a try-catch block; the script tries to run each design and if an error occurs, the script notes which design had an error and continues to the next design. This error catching is convenient when running production-scale design campaigns but is a nuisance for debugging since the messages are not very informative.

If you hit this error, I recommend running the same command that yielded the error but while adding the `-debug` flag to the command. This flag will make the script run without the try-catch block and errors will print with the standard verbose, easier-to-debug messages.

# Credits <a name="credits"></a>
* dl_binder_design: Bennett NR, et al. *Improving de novo protein binder design with deep learning.* Nat Commun 14, 2625 (2023). https://doi.org/10.1038/s41467-023-38328-5
* ProteinMPNN: Dauparas J, et al. Science 378, 49 (2022). https://doi.org/10.1126/science.add2187
* AlphaFold2: Jumper J, et al. Nature 596, 583 (2021). https://doi.org/10.1038/s41586-021-03819-2
* ipSAE: Dunbrack RL. *Rēs ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and how to fix it.* bioRxiv 2025. https://doi.org/10.1101/2025.02.10.637595 ; reference implementation https://github.com/DunbrackLab/IPSAE (MIT). The ipSAE code here is an independent numpy re-implementation of the by-residue d0 variant.
* WEHI-ResearchComputing for the maintained fork this is based on (conda-lock files, FastRelax multi-sequence script, install fixes).

