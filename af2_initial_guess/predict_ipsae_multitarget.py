#!/usr/bin/env python
"""
AlphaFold2 initial guess (dl_binder_design) with ipSAE scoring and multi-target support.

Predicts every complex in -pdbdir (or -silent) with the binder as a single sequence and the target chain(s) fixed to the
input structure, then writes the confidence scores and the predicted PDBs.

Chains
  -binder_chain / -target_chains choose the chains by ID. Default: the binder is the first chain of the first input file
  and the targets are all other chains found in the inputs. The binder may sit anywhere in a file; it is moved to the
  front before prediction. Chains that are neither binder nor target are dropped (not modelled, not scored). Chain IDs
  and residue numbers of the input are kept on the output. (PDB inputs, first model only. With -silent the binder must
  already be the first chain, both options are required, and no other chains may be present.)
  -strict_binder_check (default on) aborts a structure whose binder is not one contiguous chain, e.g. a TER record or a
  repeated chain ID inside it.
Ligands
  -ignore_ligands (default on): for PDB inputs, HETATM records (except MSE) and CONECT records are ignored, so inputs
  with ligands, waters or ions can be used as they are. Input files are never modified.
Scores (-calc_ipsae true, -ipsae_cutoff PAE cutoff in A, default 10)
  For binder chain B and each target chain X:
    ipSAE_BX      ipSAE of the binder residues against X
    ipSAE_XB      ipSAE of the X residues against the binder
    ipSAE_BX_max  larger of the two (the "max" value of Dunbrack's ipsae.py)
  plus iPAE_BX, n_iface_BX, n_iface_XB, plddt_* and pae_* columns. The README lists all columns. The column layout is
  fixed per run; a target missing from a structure gives nan.
  ipSAE is the by-residue variant with d0 from the number of valid partners (ipsae_d0res of ipsae.py).
Output
  -scorefilename: a name ending in .csv or .tsv writes a table (first column "description"), any other name the
  Rosetta-style SCORE: format (default out.sc). Rows are appended; a file with another column layout is refused.
Other
  -sort_by_length          sort the run list by residue count so AF2 recompiles less
  -checkpoint_failures     also checkpoint failed structures (they are then not retried)
"""

import os
import sys
import glob
import uuid
import argparse
from timeit import default_timer as timer

import numpy as np

import jax
import jax.numpy as jnp

from jax.lib import xla_bridge

from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

import af2_util

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent, 'include'))
from silent_tools import silent_tools

from pyrosetta import *
from rosetta import *
init('-in:file:silent_struct_type binary -mute all')


def range1(size):
    return range(1, size + 1)


CA_IDX = 1  # index of CA in AlphaFold's atom37 ordering


#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()

# I/O Arguments
parser.add_argument("-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model')
parser.add_argument("-silent", type=str, default="", help='The name of a silent file to run through the model')
parser.add_argument("-outpdbdir", type=str, default="outputs", help='The directory to which the output PDB files will be written. Only used when -pdbdir is active')
parser.add_argument("-outsilent", type=str, default="out.silent", help='The name of the silent file to which output structs will be written. Only used when -silent is active')
parser.add_argument("-runlist", type=str, default='', help="The path of a list of pdb tags to run. Only used when -pdbdir is active (default: ''; Run all PDBs)")
parser.add_argument("-checkpoint_name", type=str, default='check.point', help="The name of a file where tags which have finished will be written (default: check.point)")
parser.add_argument("-scorefilename", type=str, default='out.sc', help="The name of a file where scores will be written (default: out.sc). A name ending in .csv or .tsv writes a comma/tab separated table (first column 'description'), any other name the Rosetta-style SCORE: format")
parser.add_argument("-maintain_res_numbering", action="store_true", default=False, help='When active, the model will not renumber the residues when bad inputs are encountered (default: False)')

parser.add_argument("-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')

# AF2-Specific Arguments
parser.add_argument("-max_amide_dist", type=float, default=3.0, help='The maximum distance between an amide bond\'s carbon and nitrogen (default: 3.0)')
parser.add_argument("-recycle", type=int, default=3, help='The number of AF2 recycles to perform (default: 3)')
parser.add_argument("-no_initial_guess", action="store_true", default=False, help='When active, the model will not use an initial guess (default: False)')
parser.add_argument("-force_monomer", action="store_true", default=False, help='When active, the model will predict the structure of a monomer (default: False)')

# ipSAE / multi-target Arguments
parser.add_argument(
    "-calc_ipsae",
    type=lambda x: str(x).lower() == "true",
    default=False,
    help="Calculate ipSAE interface score (default: False)"
)
parser.add_argument(
    "-ipsae_cutoff",
    type=float,
    default=10.0,
    help="PAE cutoff (A) used for ipSAE calculation (default: 10.0)"
)
parser.add_argument(
    "-ignore_ligands",
    type=lambda x: str(x).lower() == "true",
    default=True,
    help="PDB inputs only. Ignore everything that is not a protein ATOM record (ligands, waters, ions, other HETATM, "
         "CONECT; selenomethionine MSE is kept) so inputs with ligands need no cleaning step. dl_binder_design has "
         "no ligand support, so the default is True. The input files are never modified (default: True)"
)
parser.add_argument(
    "-binder_chain",
    type=str,
    default='',
    help="Chain ID of the binder. Default: the first chain of the first input file. For PDB inputs the binder is moved "
         "to the front before prediction, whatever its position in the file. With -silent the binder must already be "
         "the first chain and this option is required."
)
parser.add_argument(
    "-allow_missing_targets",
    action="store_true",
    default=False,
    help="Treat the -target_chains list as a set of columns that a structure may lack (missing targets stay NaN) instead "
         "of requiring every input to contain all of them. Used by the AFIG wrapper when it derived the list itself."
)
parser.add_argument(
    "-target_chains",
    type=str,
    default='',
    help="Chain ID(s) of the target(s) to score the binder against, e.g. B or B,C. Default: every other chain "
         "found in the inputs. Chains that are neither binder nor target are DROPPED from the input before "
         "prediction (they are neither modelled nor scored); not possible with -silent, where the list must name all "
         "chains. With an explicit list every input must contain all of these chains (unless -allow_missing_targets)."
)
parser.add_argument(
    "-sort_by_length",
    action="store_true",
    default=False,
    help="Sort the run list by residue count. AF2 recompiles for each new total "
         "length, so grouping equal lengths cuts wall time substantially on a "
         "heterogeneous benchmark (default: False)"
)
parser.add_argument(
    "-checkpoint_failures",
    type=lambda x: str(x).lower() == "true",
    default=False,
    help="When true, a structure that raised is still written to the checkpoint "
         "file and will be skipped on a rerun. Default false, so fixing a bad "
         "input and resubmitting actually retries it (default: False)"
)
parser.add_argument(
    "-strict_binder_check",
    type=lambda x: str(x).lower() == "true",
    default=True,
    help="Abort a structure if the binder is not a single contiguous chain, e.g. a TER or a second chain with the same ID inside it (default: True)"
)

args = parser.parse_args()


#################################
# Scoring helpers (pure numpy)
#################################

def _calc_d0(n0):
    """d0 of ipSAE as in Dunbrack's ipsae.py (calc_d0_array): L floored at 26, d0 at least 1."""
    L = max(float(n0), 26.0)
    return max(1.0, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def ipsae_asym(pae, rows, cols, cutoff):
    """
    Asymmetric, by-residue ipSAE for rows -> cols.

    For each residue i in `rows`, take the residues j in `cols` with
    pae[i, j] < cutoff, derive d0 from the number of such partners, average the
    pTM-like transform over them, then report the maximum over i.

    Returns (score, n_interface_rows) where n_interface_rows is the number of
    `rows` residues with at least one valid partner.
    """
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    if rows.size == 0 or cols.size == 0:
        return float('nan'), 0

    sub = pae[np.ix_(rows, cols)]
    valid = np.isfinite(sub) & (sub < cutoff)
    n_valid = valid.sum(axis=1)

    byres = np.zeros(rows.size, dtype=float)
    for i in range(rows.size):
        if n_valid[i] == 0:
            continue
        vals = sub[i][valid[i]]
        d0 = _calc_d0(n_valid[i])
        byres[i] = float(np.mean(1.0 / (1.0 + (vals / d0) ** 2)))

    n_iface = int((n_valid > 0).sum())
    if n_iface == 0:
        return 0.0, 0
    return float(byres.max()), n_iface


def ipsae_both(pae, binder_res, target_res, cutoff):
    """
    Both asymmetric ipSAE directions plus their max.

    AB = binder -> target : residues i of the BINDER, partners j in the target with pae[i, j] < cutoff
    BA = target -> binder : residues i of the TARGET, partners j in the binder
    (Dunbrack's ipsae.py prints the same two numbers as the 'asym' rows A B / B A when the binder is chain A.)
    max(AB, BA) is the 'max' row of ipsae.py.

    Returns (ab, ba, max, n_iface_binder_side, n_iface_target_side).
    """
    ab, n_bt = ipsae_asym(pae, binder_res, target_res, cutoff)
    ba, n_tb = ipsae_asym(pae, target_res, binder_res, cutoff)
    vals = [v for v in (ab, ba) if np.isfinite(v)]
    return ab, ba, (float(max(vals)) if vals else float('nan')), n_bt, n_tb


def _kabsch(P, Q):
    """Rotation R and translation t minimising |R@P.T + t - Q.T|. P, Q are (N,3)."""
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def _rmsd(A, B):
    if A.shape[0] == 0:
        return float('nan')
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def superposed_rmsd(mobile, ref):
    """RMSD of `mobile` onto `ref` after optimal superposition (both (N,3))."""
    if mobile.shape[0] < 3:
        return float('nan')
    R, t = _kabsch(mobile, ref)
    return _rmsd((R @ mobile.T).T + t, ref)


def target_drift_rmsd(init_ca, pred_ca, idx_t1, idx_t2):
    """
    How far target 2 moved relative to target 1.

    Superpose predicted target 1 onto input target 1, apply that transform to
    predicted target 2, and measure RMSD against input target 2.  This is the
    check on whether "target positions fixed" actually held: AF2-IG templates
    and seeds the targets but does not pin their relative arrangement.
    """
    if len(idx_t1) < 3 or len(idx_t2) < 3:
        return float('nan')
    R, t = _kabsch(pred_ca[idx_t1], init_ca[idx_t1])
    moved_t2 = (R @ pred_ca[idx_t2].T).T + t
    return _rmsd(moved_t2, init_ca[idx_t2])


#################################
# Feature holder
#################################

class FeatureHolder():
    '''
    This is a struct which holds the features for a single structure being run
    through the model
    '''

    def __init__(self, pose, monomer, binderlen, tag):
        self.pose = pose
        self.tag = tag
        self.outtag = self.tag + '_af2pred'

        self.seq = pose.sequence()
        self.binderlen = binderlen
        self.monomer = monomer

        # Chain bookkeeping (filled in process_struct)
        self.chain_ids = None       # (N,) array of chain letters, input order
        self.res_numbers = None     # (N,) array of input residue numbers
        self.binder_chain = None
        self.target_chains = []     # sorted, slot order
        self.chain_index = {}       # chain letter -> 0-based residue indices

        # Pre model features
        self.initial_all_atom_positions = None
        self.initial_all_atom_masks = None

        # Post model features
        self.outpose = None
        self.plddt_array = None
        self.score_dict = None
        self.string_dict = None
        self.predicted_coords = None


#################################
# Runner
#################################

class AF2_runner():
    '''
    This class handles generating features, running the model, and parsing outputs
    '''

    def __init__(self, args, struct_manager):

        self.max_amide_dist = args.max_amide_dist

        # ipSAE settings
        self.calc_ipsae = args.calc_ipsae
        self.ipsae_cutoff = args.ipsae_cutoff

        # For timing
        self.t0 = None

        self.struct_manager = struct_manager

        # Other models may be run but their weights will also need to be downloaded
        self.model_name = "model_1_ptm"

        model_config = config.model_config(self.model_name)
        model_config.data.eval.num_ensemble = 1

        model_config.data.common.num_recycle = args.recycle
        model_config.model.num_recycle = args.recycle

        model_config.model.embeddings_and_evoformer.initial_guess = False if args.no_initial_guess else True

        model_config.data.common.max_extra_msa = 5
        model_config.data.eval.max_msa_clusters = 5

        params_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_weights')

        model_params = data.get_model_haiku_params(model_name=self.model_name, data_dir=params_dir)

        self.model_runner = model.RunModel(model_config, model_params)

    def featurize(self, feat_holder) -> None:

        all_atom_positions, all_atom_masks = af2_util.af2_get_atom_positions(
            feat_holder.pose, self.struct_manager.tmp_fn)

        feat_holder.initial_all_atom_positions = all_atom_positions
        feat_holder.initial_all_atom_masks = all_atom_masks

        initial_guess = af2_util.parse_initial_guess(feat_holder.initial_all_atom_positions)

        # Determine which residues to template
        if feat_holder.monomer:
            # For monomers predict all residues
            feat_holder.residue_mask = [False for i in range(len(feat_holder.seq))]
        else:
            # For interfaces fix the target(s) and predict the binder
            feat_holder.residue_mask = [int(i) >= feat_holder.binderlen for i in range(len(feat_holder.seq))]

        template_dict = af2_util.generate_template_features(
            feat_holder.seq,
            feat_holder.initial_all_atom_positions,
            feat_holder.initial_all_atom_masks,
            feat_holder.residue_mask
        )

        # Gather features
        feature_dict = {
            **pipeline.make_sequence_features(sequence=feat_holder.seq,
                                              description="none",
                                              num_res=len(feat_holder.seq)),
            **pipeline.make_msa_features(msas=[[feat_holder.seq]],
                                         deletion_matrices=[[[0] * len(feat_holder.seq)]]),
            **template_dict
        }

        if feat_holder.monomer:
            breaks = []
        else:
            # Geometric: catches every chain boundary AND internal loop gaps,
            # so >2-chain inputs need no special handling here.
            breaks = af2_util.check_residue_distances(
                feat_holder.initial_all_atom_positions,
                feat_holder.initial_all_atom_masks,
                self.max_amide_dist
            )

        n_expected = max(0, len(feat_holder.target_chains))  # chain boundaries = n_chains - 1
        print(f'  chain boundaries expected >= {n_expected}, residue_index breaks found: {len(breaks)}')
        if len(breaks) < n_expected:
            print('  WARNING: fewer residue_index breaks than chain boundaries. '
                  'Two chains may be fused into one polymer. Check -max_amide_dist '
                  'and the input geometry.')

        feature_dict['residue_index'] = af2_util.insert_truncations(feature_dict['residue_index'], breaks)

        feature_dict = self.model_runner.process_features(feature_dict, random_seed=0)

        return feature_dict, initial_guess

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def generate_scoredict(self, feat_holder, confidences, rmsds) -> None:

        binderlen = feat_holder.binderlen
        plddt_array = np.asarray(confidences['plddt'])
        pae = np.asarray(confidences['predicted_aligned_error'])

        plddt = float(np.mean(plddt_array))

        # Score dict in the column order of the score file (see StructManager.score_columns); all nan until filled.
        nan = float('nan')
        B = self.struct_manager.binder_id
        score_dict = {k: nan for k in self.struct_manager.score_columns()[2:]}
        score_dict["plddt_total"] = plddt
        score_dict["binder_aligned_rmsd"] = rmsds['binder_aligned_rmsd']
        score_dict["target_aligned_rmsd"] = rmsds['target_aligned_rmsd']
        score_dict["n_targets"] = 0

        string_dict = {
            "binder_chain": str(feat_holder.binder_chain) if feat_holder.binder_chain is not None else "NA",
            "target_chains": ",".join(feat_holder.target_chains) if feat_holder.target_chains else "NA",
        }

        # ---- monomer short-circuit ----------------------------------- #
        if feat_holder.monomer:
            score_dict["plddt_binder"] = plddt
            score_dict["pae_binder"] = float(np.mean(pae)) if pae.size else float('nan')
            score_dict["time"] = timer() - self.t0
            feat_holder.score_dict = score_dict
            feat_holder.string_dict = string_dict
            print(score_dict)
            print(f"Tag: {feat_holder.outtag} reported success in {score_dict['time']} seconds")
            return

        # ---- complex ------------------------------------------------- #
        binder_res = feat_holder.chain_index[feat_holder.binder_chain]
        all_target_res = np.concatenate(
            [feat_holder.chain_index[c] for c in feat_holder.target_chains]
        ) if feat_holder.target_chains else np.array([], dtype=int)

        score_dict["plddt_binder"] = float(np.mean(plddt_array[:binderlen]))
        score_dict["plddt_target"] = float(np.mean(plddt_array[binderlen:])) if binderlen < len(plddt_array) else float('nan')
        score_dict["pae_binder"] = float(np.mean(pae[:binderlen, :binderlen]))
        score_dict["pae_target"] = float(np.mean(pae[binderlen:, binderlen:])) if binderlen < pae.shape[0] else float('nan')
        score_dict["n_targets"] = len(feat_holder.target_chains)

        # cross-target PAE block (targets should NOT be pre-associated;
        # high values here are expected and this doubles as a drift proxy)
        if len(feat_holder.target_chains) >= 2:
            cross_vals = []
            tcs = feat_holder.target_chains
            for a in range(len(tcs)):
                for b in range(len(tcs)):
                    if a == b:
                        continue
                    ra = feat_holder.chain_index[tcs[a]]
                    rb = feat_holder.chain_index[tcs[b]]
                    cross_vals.append(pae[np.ix_(ra, rb)].mean())
            score_dict["pae_cross_targets"] = float(np.mean(cross_vals))

        # per-target columns (target chains actually present in this structure)
        for X in feat_holder.target_chains:
            tres = feat_holder.chain_index[X]
            if tres.size == 0:
                continue
            bt = pae[np.ix_(binder_res, tres)]
            tb = pae[np.ix_(tres, binder_res)]
            score_dict[f"iPAE_{B}{X}"] = float((bt.mean() + tb.mean()) / 2.0)
            score_dict[f"plddt_{X}"] = float(np.mean(plddt_array[tres]))
            score_dict[f"pae_{X}"] = float(pae[np.ix_(tres, tres)].mean())
            if self.calc_ipsae:
                ab, ba, mx, n_bt, n_tb = ipsae_both(pae, binder_res, tres, self.ipsae_cutoff)
                score_dict[f"ipSAE_{B}{X}"] = ab
                score_dict[f"ipSAE_{X}{B}"] = ba
                score_dict[f"ipSAE_{B}{X}_max"] = mx
                score_dict[f"n_iface_{B}{X}"] = int(n_bt)
                score_dict[f"n_iface_{X}{B}"] = int(n_tb)

        # target-target drift (only meaningful with >= 2 targets)
        if (len(feat_holder.target_chains) >= 2
                and feat_holder.predicted_coords is not None
                and feat_holder.initial_all_atom_positions is not None):
            init_ca = np.asarray(feat_holder.initial_all_atom_positions)[:, CA_IDX, :]
            pred_ca = np.asarray(feat_holder.predicted_coords)
            idx1 = feat_holder.chain_index[feat_holder.target_chains[0]]
            idx2 = feat_holder.chain_index[feat_holder.target_chains[1]]
            score_dict["target_drift_rmsd"] = target_drift_rmsd(init_ca, pred_ca, idx1, idx2)

        score_dict["time"] = timer() - self.t0

        feat_holder.score_dict = score_dict
        feat_holder.string_dict = string_dict

        print(score_dict)
        print(f"Tag: {feat_holder.outtag} reported success in {score_dict['time']} seconds")

    # ------------------------------------------------------------------ #

    def process_output(self, feat_holder, feature_dict, prediction_result) -> None:
        '''
        Take the AF2 output, parse the confidence scores from this and register the
        scores in the score file. Also write out the structure.
        '''

        structure_module = prediction_result['structure_module']

        # Predicted CA coordinates (atom index 1 = CA in atom37 ordering)
        feat_holder.predicted_coords = np.asarray(structure_module['final_atom_positions'])[:, CA_IDX, :]

        this_protein = protein.Protein(
            aatype=feature_dict['aatype'][0],
            atom_positions=structure_module['final_atom_positions'][...],
            atom_mask=structure_module['final_atom_mask'][...],
            residue_index=feature_dict['residue_index'][0] + 1,
            b_factors=np.zeros_like(structure_module['final_atom_mask'][...]))

        confidences = {}
        confidences['distogram'] = prediction_result['distogram']
        confidences['plddt'] = confidence.compute_plddt(
            prediction_result['predicted_lddt']['logits'][...])
        if 'predicted_aligned_error' in prediction_result:
            confidences.update(confidence.compute_predicted_aligned_error(
                prediction_result['predicted_aligned_error']['logits'][...],
                prediction_result['predicted_aligned_error']['breaks'][...]))

        feat_holder.plddt_array = confidences['plddt']

        # Calculate the RMSDs
        target_mask = np.zeros(len(feat_holder.seq), dtype=bool)
        target_mask[feat_holder.binderlen:] = True

        rmsds = af2_util.calculate_rmsds(
            feat_holder.initial_all_atom_positions,
            this_protein.atom_positions,
            target_mask
        )

        # Write the structure as a pdb file so Rosetta can read it
        unrelaxed_pdb_lines = protein.to_pdb(this_protein)

        with open(self.struct_manager.tmp_fn, 'w') as f:
            f.write(unrelaxed_pdb_lines)

        feat_holder.outpose = pyrosetta.pose_from_file(self.struct_manager.tmp_fn)

        os.remove(self.struct_manager.tmp_fn)

        self.generate_scoredict(feat_holder, confidences, rmsds)
        self.struct_manager.dump_pose(feat_holder)
        # the score row is written last, so a structure that failed while writing its PDB leaves no row behind
        self.struct_manager.record_scores(feat_holder.outtag, feat_holder.score_dict, feat_holder.string_dict)

    def process_struct(self, tag) -> None:

        self.t0 = timer()

        pose, monomer, binderlen, usetag = self.struct_manager.load_pose(tag)

        feat_holder = FeatureHolder(pose, monomer, binderlen, usetag)

        # Chain bookkeeping straight off the input pose
        pdbinfo = feat_holder.pose.pdb_info()
        feat_holder.chain_ids = np.array(
            [pdbinfo.chain(i) for i in range1(feat_holder.pose.size())])
        feat_holder.res_numbers = np.array(
            [pdbinfo.number(i) for i in range1(feat_holder.pose.size())])
        if self.struct_manager.orig_numbering is not None:
            feat_holder.res_numbers = self.struct_manager.orig_numbering

        feat_holder.binder_chain = feat_holder.chain_ids[0]
        sm = self.struct_manager
        if not monomer:
            if feat_holder.binder_chain != sm.binder_id:
                raise Exception(f"Input {usetag}: first chain is '{feat_holder.binder_chain}' but the binder chain is "
                                f"'{sm.binder_id}' (silent inputs must already have the binder first)")
            extra = sorted(set(feat_holder.chain_ids) - {sm.binder_id} - set(sm.target_ids))
            if extra:
                raise Exception(f"Input {usetag}: chains {','.join(extra)} are neither binder nor in -target_chains")
        feat_holder.target_chains = [t for t in sm.target_ids if t in set(feat_holder.chain_ids)]
        feat_holder.chain_index = {
            c: np.where(feat_holder.chain_ids == c)[0]
            for c in set(feat_holder.chain_ids)
        }

        # binder check: binderlen (Rosetta's first chain, cut at every TER / chain-ID change) must equal the number of
        # residues carrying the binder's chain ID, and those residues must form one block at the start of the pose
        if not monomer:
            first_block = feat_holder.chain_index[feat_holder.binder_chain]
            contiguous = (first_block.size > 0
                          and first_block[0] == 0
                          and first_block[-1] == first_block.size - 1)
            msg = None
            if first_block.size != binderlen:
                msg = (f"{first_block.size} residues carry chain ID '{feat_holder.binder_chain}' but Rosetta's first chain has "
                       f"{binderlen} (a TER or chain break inside the binder, or the chain ID is used twice)")
            elif not contiguous:
                msg = (f"chain '{feat_holder.binder_chain}' residues are not contiguous "
                       f"at the start of the pose")
            if msg is not None:
                full = (f"Input {usetag}: binder chain check failed -- {msg}. "
                        f"Check the input PDB (TER records, duplicate chain IDs).")
                if args.strict_binder_check:
                    raise Exception(full)
                print('WARNING: ' + full)

        print(f'Processing struct with tag: {feat_holder.tag} '
              f'(binder={feat_holder.binder_chain}, targets={feat_holder.target_chains}, '
              f'N={feat_holder.pose.size()})')

        feature_dict, initial_guess = self.featurize(feat_holder)

        start = timer()
        print(f'Running {self.model_name}')

        prediction_result = self.model_runner.apply(
            self.model_runner.params,
            jax.random.PRNGKey(0),
            feature_dict,
            initial_guess
        )

        print(f'Tag: {feat_holder.tag} finished AF2 prediction in {timer() - start} seconds')

        self.process_output(feat_holder, feature_dict, prediction_result)


#################################
# Struct manager
#################################

class StructManager():
    '''
    This class handles all of the input and output for the AF2 model. It deals with
    silent files vs. pdbs, checkpointing, and writing of outputs
    '''

    def __init__(self, args):
        self.args = args

        self.maintain_res_numbering = args.maintain_res_numbering
        self.score_fn = args.scorefilename
        self.tmp_fn = f'tmp_{uuid.uuid4()}.pdb'
        self.force_monomer = args.force_monomer

        self.orig_numbering = None      # input residue numbers, set by load_pose when it had to renumber
        self.silent = False
        if not args.silent == '':
            self.silent = True

            self.struct_iterator = silent_tools.get_silent_index(args.silent)['tags']

            self.sfd_in = rosetta.core.io.silent.SilentFileData(
                rosetta.core.io.silent.SilentFileOptions())
            self.sfd_in.read_file(args.silent)

            self.sfd_out = core.io.silent.SilentFileData(
                args.outsilent, False, False, "binary", core.io.silent.SilentFileOptions())

            self.outsilent = args.outsilent

        self.pdb = False
        if not args.pdbdir == '':
            self.pdb = True

            self.pdbdir = args.pdbdir
            self.outpdbdir = args.outpdbdir

            self.struct_iterator = sorted(glob.glob(os.path.join(args.pdbdir, '*.pdb')))

            if args.runlist != '':
                with open(args.runlist, 'r') as f:
                    self.runlist = set([line.strip() for line in f])
                self.struct_iterator = [
                    s for s in self.struct_iterator
                    if self.tag_for(s) in self.runlist
                ]
                print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

            if args.sort_by_length:
                self.struct_iterator = self._sort_by_length(self.struct_iterator)

        assert (self.silent ^ self.pdb), (
            f'Both silent and pdb are set to {args.silent} and {args.pdbdir} respectively. '
            f'Only one of these may be active at a time')

        # Setup checkpointing
        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

        self._resolve_chains()
        self._check_score_file()

    # ---- chain selection: which chain is the binder, which chains are scored as targets -------------- #
    @staticmethod
    def pdb_chain_order(path):
        """Chain IDs of the ATOM records of a PDB file, in order of first appearance."""
        order = []
        with open(path) as fh:
            for line in fh:
                if line.startswith('ENDMDL'):       # first model only
                    break
                if line.startswith('ATOM') and len(line) > 21 and line[21] not in order:
                    order.append(line[21])
        return order

    def _binder_blocks(self, path):
        """Number of separate blocks of ATOM records that carry the binder chain ID (1 = one contiguous chain)."""
        blocks, in_block = 0, False
        with open(path) as fh:
            for line in fh:
                if line.startswith('ENDMDL'):
                    break
                if line.startswith('TER'):
                    in_block = False
                elif line.startswith('ATOM') and len(line) > 21:
                    if line[21] == self.binder_id:
                        if not in_block:
                            blocks += 1
                        in_block = True
                    else:
                        in_block = False
        return blocks

    def _resolve_chains(self):
        a = self.args
        explicit = [c for c in a.target_chains.replace(',', ' ').replace(' ', '')]
        if len(set(explicit)) != len(explicit):
            raise Exception(f"-target_chains lists a chain twice: {a.target_chains}")
        self.explicit_targets = bool(explicit) and not a.allow_missing_targets
        if self.pdb:
            orders = [self.pdb_chain_order(st) for st in self.struct_iterator]
            if not orders or not orders[0]:
                raise Exception("No ATOM records in the first input PDB / no input PDBs to read chain IDs from.")
            binder = a.binder_chain or orders[0][0]
            if explicit:
                targets = explicit
            else:
                targets = []
                for o in orders:
                    for c in o:
                        if c != binder and c not in targets:
                            targets.append(c)
        else:
            if not a.binder_chain or not explicit:
                raise Exception("With -silent input, give -binder_chain and -target_chains (the chain IDs cannot be "
                                "pre-scanned from a silent file).")
            binder, targets = a.binder_chain, explicit
        for c in [binder] + list(targets):
            if len(c) != 1 or not c.isalnum():
                raise Exception(f"Chain IDs must be single letters or digits, got '{c}' (a blank chain ID in the input?)")
        if binder in targets:
            raise Exception(f"binder chain '{binder}' is also listed as a target chain")
        self.binder_id, self.target_ids = binder, targets
        print(f"Chain selection: binder = {binder}, targets = {','.join(targets) if targets else '(none: monomer inputs)'}"
              f" ({'from -target_chains' if explicit else 'all other chains found in the inputs'})")

    def score_columns(self):
        """Columns of the score file, in order (string columns first)."""
        B, T = self.binder_id, self.target_ids
        cols = ["binder_chain", "target_chains"]
        for X in T:
            cols += [f"ipSAE_{B}{X}", f"ipSAE_{X}{B}", f"ipSAE_{B}{X}_max"]
        cols += [f"iPAE_{B}{X}" for X in T]
        for X in T:
            cols += [f"n_iface_{B}{X}", f"n_iface_{X}{B}"]
        cols += ["plddt_total", "plddt_binder"] + [f"plddt_{X}" for X in T] + ["plddt_target"]
        cols += ["pae_binder"] + [f"pae_{X}" for X in T] + ["pae_target", "pae_cross_targets"]
        cols += ["binder_aligned_rmsd", "target_aligned_rmsd", "target_drift_rmsd", "n_targets", "time"]
        return cols

    def _score_file_format(self):
        ext = os.path.splitext(self.score_fn)[1].lower()
        return ',' if ext == '.csv' else '\t' if ext == '.tsv' else None

    def _check_score_file(self):
        """Fail before any GPU time if the score file exists with a different column layout (appending would corrupt it)."""
        import csv
        fn = self.score_fn
        if not os.path.isfile(fn) or os.path.getsize(fn) == 0:
            return
        delim = self._score_file_format()
        with open(fn, newline='') as fh:
            first = fh.readline()
        header = next(csv.reader([first], delimiter=delim)) if delim else first.split()
        cols = self.score_columns()
        expected = (['description'] + cols) if delim else (['SCORE:'] + cols + ['description'])
        if header != expected:
            raise Exception(f"Score file {fn} has a different column layout than this run (other binder/target chain IDs, "
                            f"or another program version). Use another -scorefilename.")

    def _prepare_pdb(self, path):
        """
        Returns (path_to_load, is_temporary). The pose handed to AF2 must be: binder chain first, then the target
        chains; anything else in the file is dropped. If the file already is exactly that, it is loaded as is,
        otherwise a filtered/reordered temporary copy (ATOM records only) is written and loaded instead.
        """
        chains = self.pdb_chain_order(path)
        tag = self.tag_for(path)
        if self.binder_id not in chains:
            raise Exception(f"Input {tag}: binder chain '{self.binder_id}' not found (chains in file: "
                            f"{','.join(chains)}). Set -binder_chain, or split the inputs by binder chain.")
        if self.explicit_targets:
            missing = [t for t in self.target_ids if t not in chains]
            if missing:
                raise Exception(f"Input {tag}: target chain(s) {','.join(missing)} not found (chains in file: "
                                f"{','.join(chains)}).")
        blocks = self._binder_blocks(path)
        if blocks > 1:
            msg = (f"Input {tag}: binder chain '{self.binder_id}' is split into {blocks} blocks "
                   f"(a TER record or another chain between its atoms)")
            if self.args.strict_binder_check:
                raise Exception(msg + ". Check the input PDB, or set -strict_binder_check false.")
            print('WARNING: ' + msg)
        wanted = [self.binder_id] + [t for t in self.target_ids if t in chains]
        n_nonprotein = 0
        multi_model = False
        with open(path) as fh:
            seen_end = False
            for line in fh:
                if line.startswith('ENDMDL'):
                    seen_end = True
                elif seen_end and line.startswith(('ATOM', 'MODEL')):
                    multi_model = True
                    break
                elif not seen_end and self.args.ignore_ligands and (
                        (line.startswith('HETATM') and line[17:20] != 'MSE') or line.startswith('CONECT')):
                    n_nonprotein += 1
        if chains == wanted and n_nonprotein == 0 and not multi_model:
            return path, False
        dropped = [c for c in chains if c not in wanted]
        notes = []
        if chains != wanted:
            notes.append(f"using chains {','.join(wanted)} (binder first)" +
                         (f"; dropped chains {','.join(dropped)}" if dropped else '') +
                         f"; file order was {','.join(chains)}")
        if n_nonprotein:
            notes.append(f"ignoring {n_nonprotein} ligand/water/other HETATM or CONECT record(s)")
        if multi_model:
            notes.append("file has several models: using the first")
        print(f"Input {tag}: " + "; ".join(notes))
        keep = {c: [] for c in wanted}
        with open(path) as fi:
            for line in fi:
                if line.startswith('ENDMDL'):
                    break
                if len(line) > 21 and line[21] in keep and (line.startswith('ATOM') or
                                                          (line.startswith('HETATM') and line[17:20] == 'MSE')):
                    keep[line[21]].append(line)
        tmp = f'tmp_chains_{uuid.uuid4()}.pdb'
        with open(tmp, 'w') as fo:
            for c in wanted:
                fo.writelines(keep[c])
                fo.write('TER\n')
            fo.write('END\n')
        return tmp, True

    # one definition of "tag", used by both iterate and checkpoint
    def tag_for(self, struct):
        if self.pdb:
            return '.'.join(os.path.basename(struct).split('.')[:-1])
        return struct

    @staticmethod
    def _sort_by_length(paths):
        """Cheap CA count straight off the file; avoids an AF2 recompile per length."""
        def n_ca(p):
            try:
                with open(p) as fh:
                    return sum(1 for line in fh
                               if line.startswith('ATOM') and line[12:16].strip() == 'CA')
            except Exception:
                return 10 ** 9
        decorated = sorted(((n_ca(p), p) for p in paths))
        print('Sorted run list by residue count: '
              f'{decorated[0][0] if decorated else 0} .. '
              f'{decorated[-1][0] if decorated else 0} residues')
        return [p for _, p in decorated]

    def input_check(self, pose, tag) -> bool:
        '''
        Check that all residue indices are unique (AF2 yields garbage otherwise)
        '''
        seen_indices = set()
        pdbinfo = pose.pdb_info()
        for resi in range1(pose.size()):
            residx = pdbinfo.number(resi)
            if residx in seen_indices:
                print(f"\nNon-unique residue indices detected for tag: {tag}. "
                      "This will cause AF2 to yield garbage outputs.")
                return False
            seen_indices.add(residx)
        return True

    def record_checkpoint(self, tag):
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\n')

    def iterate(self):
        for struct in self.struct_iterator:
            tag = self.tag_for(struct)
            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue
            yield struct

    @staticmethod
    def _fmt(v):
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            return str(int(v))
        v = float(v)
        return f"{v:.3f}" if np.isfinite(v) else "nan"

    def record_scores(self, tag, score_dict, string_dict):
        """
        Append one row to the score file. The columns are written in the order of the dicts (string columns first, then the
        scores in the order they were created: ipSAE_AB, ipSAE_BA, ipSAE_AB_max, ... see the scaffold).
        File type by extension: .csv (comma) / .tsv (tab): first column 'description', RFC-4180 quoting;
        anything else (default out.sc): Rosetta-style 'SCORE: <columns> description' lines.
        """
        import csv
        fn = self.score_fn
        delim = self._score_file_format()
        cols = list(string_dict.keys()) + list(score_dict.keys())
        vals = [str(string_dict[k]) for k in string_dict] + [self._fmt(score_dict[k]) for k in score_dict]
        new_file = (not os.path.isfile(fn)) or os.path.getsize(fn) == 0
        with open(fn, 'a', newline='') as f:
            if delim:
                w = csv.writer(f, delimiter=delim, lineterminator='\n')
                if new_file:
                    w.writerow(['description'] + cols)
                w.writerow([tag] + vals)
            else:
                if new_file:
                    f.write("SCORE:     %s description\n" % " ".join(cols))
                f.write("SCORE:     %s        %s\n" % (" ".join("%8s" % v for v in vals), tag))

    # restore all chains and the input numbering on the output
    @staticmethod
    def _restore_chains(pose, chain_ids, res_numbers):
        """
        Mark every chain boundary and write the input chain letters / residue
        numbers back onto the output pose, so a 3+ chain complex does not come
        back as 2 chains and DockQ can map residues without a lookup table.
        """
        n = pose.size()
        if chain_ids is None or len(chain_ids) != n:
            print('  WARNING: chain bookkeeping length mismatch; leaving output chains as-is')
            return pose

        boundaries = [i for i in range(1, n) if chain_ids[i] != chain_ids[i - 1]]
        try:
            conf = pose.conformation()
            for b in boundaries:
                # b is the 0-based index of the first residue of a new chain,
                # so the last residue of the previous chain is b in 1-based terms
                conf.insert_chain_ending(b)
        except Exception as e:
            print(f'  WARNING: insert_chain_ending failed ({e}); '
                  f'PDB chain letters will still be set from pdb_info')

        info = pose.pdb_info()
        for resi in range1(n):
            info.chain(resi, str(chain_ids[resi - 1]))
            info.number(resi, int(res_numbers[resi - 1]))
        info.obsolete(False)
        pose.pdb_info(info)
        return pose

    def dump_pose(self, feat_holder):
        if feat_holder.monomer:
            pose = feat_holder.outpose
        else:
            # mark every chain boundary, then restore the chain IDs and residue numbers of the input
            try:
                pose = af2_util.insert_Rosetta_chainbreaks(feat_holder.outpose, feat_holder.binderlen)
            except Exception as e:
                print(f'  WARNING: insert_Rosetta_chainbreaks failed ({e}); continuing')
                pose = feat_holder.outpose
            pose = self._restore_chains(pose, feat_holder.chain_ids, feat_holder.res_numbers)

        # Add the plddt scores as b-factors to the pose
        info = pose.pdb_info()
        for resi in range1(pose.size()):
            info.add_reslabel(resi, f'{feat_holder.plddt_array[resi-1]}')
            for atom_i in range1(pose.residue(resi).natoms()):
                info.bfactor(resi, atom_i, feat_holder.plddt_array[resi - 1])
        pose.pdb_info(info)

        if self.pdb:
            if not os.path.exists(self.outpdbdir):
                os.makedirs(self.outpdbdir)
            pdbfile = os.path.join(self.outpdbdir, feat_holder.outtag + '.pdb')
            pose.dump_pdb(pdbfile)

        if self.silent:
            struct = self.sfd_out.create_SilentStructOP()
            struct.fill_struct(pose, feat_holder.outtag)
            for scorename, value in feat_holder.score_dict.items():
                struct.add_energy(scorename, value, 1)
            self.sfd_out.add_structure(struct)
            self.sfd_out.write_silent_struct(struct, self.outsilent)

    def load_pose(self, tag):
        if not self.pdb and not self.silent:
            raise Exception('Neither pdb nor silent is set to True. Cannot load pose')

        if self.pdb:
            load_path, is_tmp = self._prepare_pdb(tag)
            try:
                pose = pose_from_pdb(load_path)
            finally:
                if is_tmp and os.path.exists(load_path):
                    os.remove(load_path)
            usetag = self.tag_for(tag)

        if self.silent:
            pose = Pose()
            usetag = tag
            self.sfd_in.get_structure(tag).fill_pose(pose)

        self.orig_numbering = None
        passes_check = self.input_check(pose, tag)

        if not passes_check:
            if not self.maintain_res_numbering:
                # AF2 needs unique residue numbers: renumber 1..N for the model, but keep the chain IDs, and remember the
                # input numbers so that the output PDB gets them back
                print(f"Renumbering {tag} (residue numbers repeat)")
                old = pose.pdb_info()
                chains = [old.chain(i) for i in range1(pose.size())]
                self.orig_numbering = np.array([old.number(i) for i in range1(pose.size())])
                info = core.pose.PDBInfo(pose)
                for i, c in enumerate(chains, start=1):
                    info.chain(i, c)
                pose.pdb_info(info)
            else:
                raise Exception(f"Pose {tag} failed input checking.")

        splits = pose.split_by_chain()

        if len(splits) < 1:
            raise Exception(f"Pose {tag} is empty. This is not supported by this script.")

        elif len(splits) == 1:
            monomer = True
            binderlen = -1

        elif len(splits) > 1:
            if self.force_monomer:
                print("/" * 60)
                print(f"Pose {tag} has multiple chains. But force_monomer is set to True. Treating as monomer.")
                print(f"I am going to assume that the first chain is the binder and that is the chain I will predict")
                print("/" * 60)
                monomer = True
                pose = splits[1]
                binderlen = -1
            else:
                monomer = False
                binderlen = splits[1].size()

        return pose, monomer, binderlen, usetag


####################
####### Main #######
####################

def detect_gpu():
    """Report the JAX devices; device_kind is the card name, never 'gpu', so ask for the platform instead."""
    print("=== JAX Device Detection ===")
    print(f"All JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    gpu_devices = []
    for platform in ('gpu', 'cuda', 'rocm'):
        try:
            gpu_devices = jax.devices(platform)
            if gpu_devices:
                break
        except RuntimeError:
            continue

    if gpu_devices:
        print('/' * 60)
        print('Found GPU and will use it to run AF2')
        print(f'GPU devices: {gpu_devices}')
        print('/' * 60)
        jax.config.update('jax_default_device', gpu_devices[0])
        print(f'Set JAX default device to: {gpu_devices[0]}\n')
    else:
        print('/' * 60)
        print('WARNING! No GPU detected running AF2 on CPU')
        print(f'JAX backend: {jax.default_backend()}')
        print('/' * 60)
        print('\n')
    return gpu_devices


if __name__ == '__main__':

    detect_gpu()

    struct_manager = StructManager(args)
    af2_runner = AF2_runner(args, struct_manager)

    for struct in struct_manager.iterate():

        tag = struct_manager.tag_for(struct)

        succeeded = False

        if args.debug:
            af2_runner.process_struct(struct)
            succeeded = True

        else:
            t0 = timer()
            try:
                af2_runner.process_struct(struct)
                succeeded = True

            except KeyboardInterrupt:
                sys.exit("Script killed by Control+C, exiting")

            except Exception as e:
                seconds = int(timer() - t0)
                print(f"Struct with tag {tag} failed in {seconds} seconds")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                import traceback
                traceback.print_exc()
                print("\n" + "=" * 80 + "\n")
                with open(struct_manager.chkfn + '.failed', 'a') as fh:
                    fh.write(f'{tag}\t{type(e).__name__}\t{str(e)}\n')

        # record the same string iterate() will compare against
        # Only on success unless -checkpoint_failures, so that repairing a bad
        # input and resubmitting retries it instead of skipping it forever.
        if succeeded or args.checkpoint_failures:
            struct_manager.record_checkpoint(tag)