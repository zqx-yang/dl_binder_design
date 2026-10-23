# Slurm scripts

`AFIG_sample_sh_1gpu.sh` (the job script you edit) and `AFIG_wrapper.sh` (sourced by it) run
`af2_initial_guess/predict_ipsae_multitarget.py` on a folder of PDBs on a Slurm cluster. We run them on our cluster; they are
shared here as an example to adapt.

## What the wrapper does
- Checks the inputs before any GPU time: every PDB in `INPUT_DIR` (a flat folder of `*.pdb`, first model only) must contain the binder chain, and the target chains if you list them; JAX must see a GPU.
- Skips PDBs that already have a `<name>_af2pred.pdb` in `OUT_DIR` (or in `OTHER_OUTPUT_DIRS`), so a resubmitted job continues where it stopped.
- Sorts the remaining PDBs by length and runs them in batches, one python process per batch, so AlphaFold2 compiles each length once. `BATCH_SIZE` is a list: `50` = batches of about 50; `10,all` = a first batch of about 10 (the pilot) and then everything else in one process; `all` = one process. The pilot must produce outputs for at least `PILOT_MIN_OK_PERCENT` (default 50) % of its structures, otherwise the run stops before the rest starts.
- Writes `<name>_af2pred.pdb`, the score file (`SCORE_FILE_NAME`; `.csv` / `.tsv` give a table, columns are described in the main README) and `check.point` (finished structures) to `OUT_DIR`. Structures that fail are listed with the reason in `check.point.failed` and the job log, are retried when you resubmit, and make the job return non-zero.

## Setup
1. Create the Python environment from `include/af2_binder_design2.yml` (JAX with CUDA 12, PyRosetta, dm-haiku, TensorFlow). PyRosetta needs its own licence / conda channel. We have not tested creating the environment from this file alone; it lists the packages and pins of the environment we use. Put the AF2 weights in `af2_initial_guess/model_weights/params` (see the main README).
2. In the job script set `OUT_DIR`, `SCRIPT_DIR`, `INPUT_DIR` (absolute paths), `AFIG_WRAPPER` (path of `AFIG_wrapper.sh`), the `#SBATCH` lines for your cluster, and the environment lines before `source`: activate the environment and set `DL_BINDER_DESIGN_ROOT` to this repository (otherwise the wrapper runs `module load dl_binder_design`).
3. Submit from a folder where the Slurm log may be written: `sbatch AFIG_sample_sh_1gpu.sh` (the log is `AFIG_<jobid>.log` there; a copy is kept in `SCRIPT_DIR`).

`common_minimal.sh` provides the job helpers the wrapper expects: results go straight to `OUT_DIR`. Point `COMMON_SH` at your own file to use scratch space or syncing instead (the contract is at the top of that file). Temporary files go to `$WORK_TMP_ROOT` (default `/tmp/$USER/jobs`); use a node-local folder. Needs bash 4.3 or newer.
