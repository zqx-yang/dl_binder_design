#!/bin/bash
# AFIG_wrapper.sh
#
# Batch runner for AlphaFold2 initial guess (dl_binder_design, predict_ipsae_multitarget.py).
# SOURCE it as the last line of the Slurm job script (see AFIG_sample_sh_1gpu.sh):
#   source /path/to/AFIG_wrapper.sh
#
# Set by the job script
#   OUT_DIR, SCRIPT_DIR  absolute paths: results folder; folder for a copy of the job log
#   INPUT_DIR            folder of input PDBs
#   AFIG_ARGS            bash array of predict_ipsae_multitarget.py options (listed in the job script)
# Optional variables [default]
#   BINDER_CHAIN         chain ID of the binder [first chain of the first PDB]
#   TARGET_CHAINS        chain ID(s) to score the binder against, e.g. B,C [all other chains]; chains that are
#                        neither binder nor target are dropped
#   SCORE_FILE_NAME      score file inside OUT_DIR; .csv / .tsv give a table [out.sc]
#   BATCH_SIZE           batch schedule, integers and "all": 50 = batches of ~50; 10,all = a pilot of ~10 structures,
#                        checked, then everything else in one process [50]
#   OTHER_OUTPUT_DIRS    bash array of earlier output folders (read-only); PDBs predicted there are skipped [()]
#   MAX_BATCH_SIZE       cap on structures per batch, 0 = none [2x each integer in BATCH_SIZE]
#   PILOT_MIN_OK_PERCENT a pilot batch must produce outputs for at least this % of its structures, or the run stops [50]
#   SORT_BY_LENGTH       sort the pending PDBs by residue count (AF2 compiles once per distinct length and process) [true]
#   GROUP_BY_LENGTH      end a batch only where the length changes, so no length is compiled twice [true]
#   AFIG_ALLOW_CPU       run even if JAX sees no GPU [false]
#   COMMON_SH            job I/O helpers [common_minimal.sh next to this file]
#   WORK_TMP_ROOT        folder for temporary files [/tmp/$USER/jobs]
#
# What it does
#   1. checks that every PDB has the binder (and, if TARGET_CHAINS is set, the targets) and that JAX sees a GPU;
#   2. skips PDBs that already have <name>_af2pred.pdb in OUT_DIR or OTHER_OUTPUT_DIRS (resume);
#   3. sorts the rest by length, splits them into batches and runs predict_ipsae_multitarget.py once per batch;
#   4. writes <name>_af2pred.pdb, the score file and check.point (finished structures) to OUT_DIR.
# Each batch logs a BATCH_TIMING line (grep it for wall times).
# Reserved options, set by the wrapper or not allowed here, so they must not be in AFIG_ARGS: -pdbdir -outpdbdir
# -scorefilename -checkpoint_name -binder_chain -target_chains -allow_missing_targets -silent -outsilent -runlist.
# Structures that fail are logged, listed in <OUT_DIR>/check.point.failed and not checkpointed; the wrapper returns
# non-zero if any structure has no output.

(return 0 2>/dev/null) || {
  echo "ERROR: AFIG_wrapper.sh must be sourced, not executed:" >&2
  echo "  source ${BASH_SOURCE[0]}" >&2
  exit 1
}

# ---- required ------------------------------------------------------------ #
: "${OUT_DIR:?OUT_DIR must be set in the job script (absolute path)}"
: "${SCRIPT_DIR:?SCRIPT_DIR must be set in the job script (absolute path)}"
: "${INPUT_DIR:?INPUT_DIR must be set in the job script}"
declare -p AFIG_ARGS >/dev/null 2>&1 || { echo "ERROR: AFIG_ARGS array is not defined in the job script." >&2; return 1; }
[[ -d "$INPUT_DIR" ]] || { echo "ERROR: INPUT_DIR not found: $INPUT_DIR" >&2; return 1; }

# ---- defaults ------------------------------------------------------------ #
BINDER_CHAIN="${BINDER_CHAIN:-}"
TARGET_CHAINS="${TARGET_CHAINS:-}"
SCORE_FILE_NAME="${SCORE_FILE_NAME:-out.sc}"
BATCH_SIZE="${BATCH_SIZE:-50}"
# Without MAX_BATCH_SIZE every integer entry of the schedule is capped at 2x its size.
MAX_USER_SET=false
[[ -n "${MAX_BATCH_SIZE:-}" ]] && MAX_USER_SET=true
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
PILOT_MIN_OK_PERCENT="${PILOT_MIN_OK_PERCENT:-50}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-true}"
SORT_BY_LENGTH="${SORT_BY_LENGTH:-true}"

if ! declare -p OTHER_OUTPUT_DIRS >/dev/null 2>&1; then declare -a OTHER_OUTPUT_DIRS=(); fi

# --- fixed flags must not be in AFIG_ARGS ---
for a in "${AFIG_ARGS[@]}"; do
  case "$a" in
    -pdbdir|-outpdbdir|-scorefilename|-checkpoint_name|-binder_chain|-target_chains|-allow_missing_targets|-silent|-outsilent|-runlist)
      echo "ERROR: AFIG_ARGS must not contain $a -- the wrapper sets it (use BINDER_CHAIN, TARGET_CHAINS, SCORE_FILE_NAME)." >&2; return 1 ;;
  esac
done

# --- OTHER_OUTPUT_DIRS: absolute, existing directories ---
for d in ${OTHER_OUTPUT_DIRS[@]+"${OTHER_OUTPUT_DIRS[@]}"}; do
    [[ "$d" == /* ]] || { echo "ERROR: OTHER_OUTPUT_DIRS entry must be an absolute path, got '$d'" >&2; return 1; }
    [[ -d "$d" ]]    || { echo "ERROR: OTHER_OUTPUT_DIRS entry is not a directory: $d" >&2; return 1; }
done

if [[ -z "$SCORE_FILE_NAME" || "$SCORE_FILE_NAME" == */* || "$SCORE_FILE_NAME" == .* || "$SCORE_FILE_NAME" == check.point || "$SCORE_FILE_NAME" == *.pdb ]]; then
    echo "ERROR: SCORE_FILE_NAME must be a plain file name (no '/', not starting with '.', not check.point, not *.pdb), got '$SCORE_FILE_NAME'" >&2
    return 1
fi
TARGET_CHAINS="${TARGET_CHAINS//[[:space:],]/}"          # "B,C" / "B C" / "BC" -> BC
if [[ -n "$BINDER_CHAIN" && ! "$BINDER_CHAIN" =~ ^[A-Za-z0-9]$ ]]; then
    echo "ERROR: BINDER_CHAIN must be one chain ID character (A-Z a-z 0-9), got '$BINDER_CHAIN'" >&2; return 1
fi
if [[ -n "$TARGET_CHAINS" ]]; then
    [[ "$TARGET_CHAINS" =~ ^[A-Za-z0-9]+$ ]] || { echo "ERROR: TARGET_CHAINS must be chain ID characters (A-Z a-z 0-9), e.g. B,C; got '$TARGET_CHAINS'" >&2; return 1; }
    for (( _i=0; _i<${#TARGET_CHAINS}; _i++ )); do
        _c="${TARGET_CHAINS:_i:1}"
        [[ "${TARGET_CHAINS:0:_i}" == *"$_c"* ]] && { echo "ERROR: TARGET_CHAINS lists chain $_c twice." >&2; return 1; }
        [[ -n "$BINDER_CHAIN" && "$_c" == "$BINDER_CHAIN" ]] && { echo "ERROR: BINDER_CHAIN $BINDER_CHAIN is also in TARGET_CHAINS." >&2; return 1; }
    done
fi

# BATCH_SIZE is a schedule: comma-separated entries, each a positive integer or
# "all". Entry i sizes the i-th python process; the LAST entry repeats until the
# pending list is used up; "all" takes whatever remains, so nothing may follow it.
declare -a sched=()
IFS=',' read -r -a sched_raw <<< "$BATCH_SIZE"
for tok in ${sched_raw[@]+"${sched_raw[@]}"}; do
    tok="${tok//[[:space:]]/}"
    if [[ "$tok" == "all" ]]; then
        sched+=( 0 )                          # 0 encodes "all"
    elif [[ "$tok" =~ ^[0-9]+$ ]] && (( tok >= 1 )); then
        sched+=( "$tok" )
    else
        echo "ERROR: BATCH_SIZE entries must be positive integers or 'all', got '$BATCH_SIZE'" >&2
        return 1
    fi
done
if (( ${#sched[@]} == 0 )); then
    echo "ERROR: BATCH_SIZE is empty" >&2
    return 1
fi
for (( i=0; i<${#sched[@]}-1; i++ )); do
    if (( sched[i] == 0 )); then
        echo "ERROR: 'all' must be the last BATCH_SIZE entry, got '$BATCH_SIZE'" >&2
        return 1
    fi
done
if [[ "$MAX_USER_SET" == true ]]; then
    if ! [[ "$MAX_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
        echo "ERROR: MAX_BATCH_SIZE must be a non-negative integer, got '$MAX_BATCH_SIZE'" >&2
        return 1
    fi
    if (( MAX_BATCH_SIZE > 0 )); then
        for sz in "${sched[@]}"; do
            if (( sz > 0 && MAX_BATCH_SIZE < sz )); then
                echo "ERROR: MAX_BATCH_SIZE ($MAX_BATCH_SIZE) is below a BATCH_SIZE entry ($sz)" >&2
                return 1
            fi
        done
    fi
fi
if ! [[ "$PILOT_MIN_OK_PERCENT" =~ ^[0-9]+$ ]] || (( PILOT_MIN_OK_PERCENT > 100 )); then
    echo "ERROR: PILOT_MIN_OK_PERCENT must be an integer 0-100, got '$PILOT_MIN_OK_PERCENT'" >&2
    return 1
fi

shopt -s nullglob
all_pdbs=("$INPUT_DIR"/*.pdb)
shopt -u nullglob
(( ${#all_pdbs[@]} > 0 )) || { echo "ERROR: no .pdb files in INPUT_DIR: $INPUT_DIR" >&2; return 1; }

### COMMON FUNCTIONS AND PATHS
source "${COMMON_SH:-$(dirname "${BASH_SOURCE[0]}")/common_minimal.sh}"
resolve_job_paths "$OUT_DIR" "$SCRIPT_DIR" || return 1   # sets OUT_REL / SCRIPT_REL
job_start
setup_scratch_out
export SCRATCH_OUT
seed_scratch_from_existing "$OUT_REL"        # existing results in OUT_DIR are visible to the run (resume)
start_log_mirror "$SCRIPT_REL" 30            # job log copied to SCRIPT_DIR every 30 s
start_incremental_sync "$OUT_REL"            # scratch -> OUT_DIR
# Temporary files and the working directory must be OUTSIDE $SCRATCH_OUT: the python script writes and deletes a
# tmp_*.pdb per structure, which would race the result sync.
WORK_TMP="${WORK_TMP_ROOT:-/tmp/${USER}/jobs}/${SLURM_JOB_ID:-$$}_tmp"
mkdir -p "$WORK_TMP" || return 1
finalize() {
  job_end
  stop_log_mirror
  stop_incremental_sync_and_finalize
  rm -rf "${WORK_TMP:-}"
}
trap finalize EXIT

SCOREFILE="$SCRATCH_OUT/$SCORE_FILE_NAME"
CHECKPOINT="$SCRATCH_OUT/check.point"

# ---- environment --------------------------------------------------------- #
# Python environment: skipped if DL_BINDER_DESIGN_ROOT (this repository) is already set and the environment active.
if [[ -z "${DL_BINDER_DESIGN_ROOT:-}" ]]; then
    module load dl_binder_design
fi
if [[ -z "${DL_BINDER_DESIGN_ROOT:-}" ]]; then
    echo "ERROR: DL_BINDER_DESIGN_ROOT is not set. Set it to the dl_binder_design checkout and activate its python environment." >&2
    return 1
fi

# XLA wants CUDA_HOME; derive it from nvcc if unset (not fatal when the CUDA libs are already on the loader path).
CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -z "$CUDA_HOME" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [[ -n "$CUDA_HOME" ]]; then
    export CUDA_HOME
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
    echo "CUDA_HOME              : $CUDA_HOME"
else
    echo "CUDA_HOME              : not set and nvcc not found -- leaving XLA_FLAGS alone"
fi

AFIG_DIR="$DL_BINDER_DESIGN_ROOT/af2_initial_guess"
PREDICT_SCRIPT="$AFIG_DIR/predict_ipsae_multitarget.py"

if [[ ! -f "$PREDICT_SCRIPT" ]]; then
    echo "ERROR: prediction script not found: $PREDICT_SCRIPT" >&2
    return 1
fi

# ---- check that the script has the options this wrapper passes ------------------ #
PREDICT_HELP="$(python "$PREDICT_SCRIPT" --help 2>&1)"
missing_flags=()
for flag in -binder_chain -target_chains -allow_missing_targets -ignore_ligands -strict_binder_check; do
    grep -q -- "$flag" <<<"$PREDICT_HELP" || missing_flags+=("$flag")
done
if (( ${#missing_flags[@]} )); then
    echo "ERROR: $PREDICT_SCRIPT does not accept: ${missing_flags[*]}" >&2
    echo "  Use predict_ipsae_multitarget.py from the same version of this repository as the wrapper." >&2
    return 1
fi

# ---- chain selection, fixed once for the whole run ------------------------------ #
# The score file has one header named after the chain IDs, but batches are separate python processes, so binder and
# targets are resolved here from ALL PDBs in INPUT_DIR and passed on as explicit flags. A PDB without the binder (or,
# with TARGET_CHAINS set, without a target) is reported now, before any GPU time.
CHAIN_CFG="$(BINDER_CHAIN="$BINDER_CHAIN" TARGET_CHAINS="$TARGET_CHAINS" INPUT_DIR="$INPUT_DIR" python - <<'PYCHAIN'
import os, glob, sys
b_arg, t_arg, d = os.environ["BINDER_CHAIN"], os.environ["TARGET_CHAINS"], os.environ["INPUT_DIR"]
files = sorted(glob.glob(os.path.join(d, "*.pdb")))
if not files:
    print("ERROR: no .pdb files in INPUT_DIR", file=sys.stderr); sys.exit(2)
def order(p):
    o = []
    with open(p) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):          # first model only
                break
            if line.startswith("ATOM") and len(line) > 21 and line[21] not in o:
                o.append(line[21])
    return o
orders = {f: order(f) for f in files}
badid = [os.path.basename(f) for f, o in orders.items() if any(not c.isalnum() for c in o)]
if badid:
    print("ERROR: chain IDs must be letters or digits; blank or other chain IDs in: " + ", ".join(badid[:10]), file=sys.stderr); sys.exit(2)
first = orders[files[0]]
if not b_arg and not first:
    print(f"ERROR: no ATOM records in {files[0]}", file=sys.stderr); sys.exit(2)
binder = b_arg or first[0]
if t_arg:
    targets = list(t_arg)
else:
    targets = []
    for f in files:
        for c in orders[f]:
            if c != binder and c not in targets:
                targets.append(c)
bad = []
for f, o in orders.items():
    if binder not in o:
        bad.append(f"{os.path.basename(f)}: binder chain {binder} missing (has {','.join(o)})")
    elif t_arg:
        miss = [t for t in targets if t not in o]
        if miss:
            bad.append(f"{os.path.basename(f)}: target chain(s) {','.join(miss)} missing (has {','.join(o)})")
if bad:
    print(f"ERROR: {len(bad)} input PDB(s) do not match binder={binder} targets={','.join(targets) or '-'}:", file=sys.stderr)
    for x in bad[:15]: print("   " + x, file=sys.stderr)
    if len(bad) > 15: print(f"   ... and {len(bad)-15} more", file=sys.stderr)
    print("   Set BINDER_CHAIN / TARGET_CHAINS, or split the inputs into folders by chain layout.", file=sys.stderr)
    sys.exit(3)
print(binder + "|" + "".join(targets))
PYCHAIN
)" || { echo "ERROR: chain check of INPUT_DIR failed (see above)." >&2; return 1; }
BINDER_RESOLVED="${CHAIN_CFG%%|*}"
TARGETS_RESOLVED="${CHAIN_CFG#*|}"
if [[ -n "$TARGET_CHAINS" && "$TARGET_CHAINS" == *"$BINDER_RESOLVED"* ]]; then
    echo "ERROR: the binder chain ($BINDER_RESOLVED, the first chain of the first PDB unless BINDER_CHAIN is set) is also in TARGET_CHAINS." >&2
    return 1
fi

# ---- preflight ----------------------------------------------------------- #
echo "==================== AFIG_wrapper ===================="
echo "DL_BINDER_DESIGN_ROOT : $DL_BINDER_DESIGN_ROOT"
echo "python                : $(which python)  ($(python -V 2>&1))"
echo "predict script        : $(basename "$PREDICT_SCRIPT")"
echo "binder chain          : $BINDER_RESOLVED   ($([[ -n "$BINDER_CHAIN" ]] && echo BINDER_CHAIN || echo "first chain of first input"))"
echo "target chain(s)       : ${TARGETS_RESOLVED:-none}   ($([[ -n "$TARGET_CHAINS" ]] && echo TARGET_CHAINS || echo "all other chains in INPUT_DIR"))"
echo "input  dir            : $INPUT_DIR"
echo "output dir            : $SCRATCH_OUT"
echo "score file            : $SCOREFILE"
echo "checkpoint            : $CHECKPOINT"
echo "batch schedule        : $BATCH_SIZE   max: $([[ "$MAX_USER_SET" == true ]] && echo "$MAX_BATCH_SIZE" || echo "2x each entry")   group by length: $GROUP_BY_LENGTH"
if (( ${#sched[@]} > 1 )); then
    echo "pilot gate            : continue past a leading batch only if >= ${PILOT_MIN_OK_PERCENT}% of it produced outputs (0 = off)"
fi
echo "AFIG_ARGS             : ${AFIG_ARGS[*]}"
echo "sort by length        : $SORT_BY_LENGTH"
echo "work tmp              : $WORK_TMP"
(( ${#OTHER_OUTPUT_DIRS[@]} )) && printf 'extra output dir      : %s\n' "${OTHER_OUTPUT_DIRS[@]}"

python - <<'PYGPU'
import sys
try:
    import jax
    gpus = []
    for p in ("gpu", "cuda", "rocm"):
        try:
            gpus = jax.devices(p)
            if gpus:
                break
        except RuntimeError:
            continue
    print(f"jax backend           : {jax.default_backend()}")
    print(f"jax GPU devices       : {gpus if gpus else 'NONE -- AF2 will run on CPU'}")
    if not gpus:
        sys.exit(3)
except Exception as e:
    print(f"jax check failed      : {e}")
    sys.exit(3)
PYGPU
gpu_rc=$?
if [[ $gpu_rc -ne 0 ]]; then
    echo "AFIG_wrapper: no GPU visible to JAX. Aborting rather than running on CPU."
    echo "              Set AFIG_ALLOW_CPU=true to override."
    [[ "${AFIG_ALLOW_CPU:-false}" != "true" ]] && return 1
fi
echo "======================================================"

# Every path handed to the script below is absolute; the working directory is WORK_TMP.
export TMPDIR="$WORK_TMP"
cd "$WORK_TMP" || return 1

# ---- collect pending inputs ---------------------------------------------- #
pending_pdbs=()
for pdb in "${all_pdbs[@]}"; do
    base=$(basename "$pdb" .pdb)
    outfile="${base}_af2pred.pdb"
    found=false
    [[ -f "$SCRATCH_OUT/$outfile" ]] && found=true
    if [[ "$found" == false ]]; then
        for dir in ${OTHER_OUTPUT_DIRS[@]+"${OTHER_OUTPUT_DIRS[@]}"}; do
            if [[ -f "$dir/$outfile" ]]; then found=true; break; fi
        done
    fi
    [[ "$found" == false ]] && pending_pdbs+=("$pdb")
done

total=${#pending_pdbs[@]}
echo "Inputs found: ${#all_pdbs[@]}   already done: $(( ${#all_pdbs[@]} - total ))   to process: $total"
if (( total == 0 )); then
    echo "Nothing to do."
    echo "Output folder: ${SYNC_DEST:-$SCRATCH_OUT}"
    return 0
fi

# ---- sort pending PDBs by length -------------------------------------------------- #
# pending_lens[i] is the residue count of pending_pdbs[i]; GROUP_BY_LENGTH uses it to decide where a batch may end.
pending_lens=()
if [[ "$SORT_BY_LENGTH" == "true" ]]; then
    echo "Sorting pending inputs by residue count ..."
    mapfile -t sorted_lines < <(
        for p in "${pending_pdbs[@]}"; do
            n=$(awk '$1=="ATOM" && substr($0,13,4)==" CA "{n++} END{print n+0}' "$p")
            printf '%08d\t%s\n' "$n" "$p"
        done | sort -t $'\t' -k1,1n -k2,2
    )
    pending_pdbs=()
    for line in "${sorted_lines[@]}"; do
        n="${line%%$'\t'*}"
        pending_lens+=( $(( 10#$n )) )
        pending_pdbs+=( "${line#*$'\t'}" )
    done
    unset sorted_lines
    echo "  residue count range: ${pending_lens[0]} .. ${pending_lens[-1]}"
fi

# ---- batch plan ------------------------------------------------------------------ #
# batch_starts: first index of each batch plus a final sentinel = total. Batch i has sched[i] structures (the last
# entry repeats, 0 = "all"). With GROUP_BY_LENGTH a batch takes at least that many, then extends to the next change in
# residue count, up to its cap; otherwise exactly that many. batch_pilot[i] is true for batches followed by a further
# schedule entry (the "10" in "10,all"); those are checked before the next one starts.
use_groups=false
if [[ "$GROUP_BY_LENGTH" == "true" ]]; then
    if [[ "$SORT_BY_LENGTH" == "true" ]]; then
        use_groups=true
    else
        echo "AFIG_wrapper: GROUP_BY_LENGTH needs SORT_BY_LENGTH=true; using exact-size batches."
    fi
fi

batch_starts=(0)
batch_pilot=()
pos=0
ci=0
while (( pos < total )); do
    si=$(( ci < ${#sched[@]} - 1 ? ci : ${#sched[@]} - 1 ))
    size=${sched[si]}
    if (( size == 0 )); then
        end=$total
    else
        end=$(( pos + size ))
        (( end > total )) && end=$total
        if [[ "$use_groups" == true ]]; then
            if [[ "$MAX_USER_SET" == true ]]; then cap=$MAX_BATCH_SIZE; else cap=$(( size * 2 )); fi
            while (( end < total && pending_lens[end] == pending_lens[end-1] )) \
                  && (( cap == 0 || end - pos < cap )); do
                end=$(( end + 1 ))
            done
        fi
    fi
    batch_starts+=( "$end" )
    if (( ci < ${#sched[@]} - 1 )); then batch_pilot+=( true ); else batch_pilot+=( false ); fi
    pos=$end
    ci=$(( ci + 1 ))
done
n_batches=$(( ${#batch_starts[@]} - 1 ))
n_pilot=0
for f in "${batch_pilot[@]}"; do [[ "$f" == true ]] && n_pilot=$(( n_pilot + 1 )); done

batch_distinct=()
if (( ${#pending_lens[@]} )); then
    distinct_total=$(printf '%s\n' "${pending_lens[@]}" | sort -u | wc -l)
    compile_events=0
    min_b=$total
    max_b=0
    for (( b=0; b<n_batches; b++ )); do
        s=${batch_starts[b]}
        c=$(( batch_starts[b+1] - s ))
        d=$(printf '%s\n' "${pending_lens[@]:s:c}" | sort -u | wc -l)
        batch_distinct+=( "$d" )
        compile_events=$(( compile_events + d ))
        (( c < min_b )) && min_b=$c
        (( c > max_b )) && max_b=$c
    done
    echo "Batch plan: $n_batches batches of $min_b..$max_b structures;" \
         "$distinct_total distinct lengths, $compile_events compile events" \
         "(ideal = $distinct_total, extra = $(( compile_events - distinct_total )))"
else
    echo "Batch plan: $n_batches batches from schedule '$BATCH_SIZE'"
fi
if (( n_pilot > 0 && n_batches > n_pilot )); then
    echo "Pilot: the first $n_pilot batch(es) run first and are checked (>= ${PILOT_MIN_OK_PERCENT}% outputs, exit 0)" \
         "before the remaining $(( total - batch_starts[n_pilot] )) structures start."
fi

# ---- run ------------------------------------------------------------------ #
run_predict() {
    local pdbdir="$1"
    local -a cmd=(
        python "$PREDICT_SCRIPT"
        -pdbdir "$pdbdir"
        -outpdbdir "$SCRATCH_OUT"
        -scorefilename "$SCOREFILE"
        -checkpoint_name "$CHECKPOINT"
    )
    cmd+=( -binder_chain "$BINDER_RESOLVED" )
    if [[ -n "$TARGET_CHAINS" ]]; then
        cmd+=( -target_chains "$TARGETS_RESOLVED" )                           # explicit: every input must have them all
    elif [[ -n "$TARGETS_RESOLVED" ]]; then
        cmd+=( -target_chains "$TARGETS_RESOLVED" -allow_missing_targets )    # derived union: a structure may lack some
    fi
    cmd+=( "${AFIG_ARGS[@]}" )
    "${cmd[@]}"
}

failed_batches=0
pilot_failed=false

for (( b=0; b<n_batches; b++ )); do
    s=${batch_starts[b]}
    e=${batch_starts[b+1]}
    cnt=$(( e - s ))
    echo "--- batch $(( b + 1 )) / $n_batches  (inputs $(( s + 1 ))-$e, $cnt structures) ---"
    t_batch0=$(date +%s)
    TMPDIR_B=$(mktemp -d "$WORK_TMP/tmp.XXXXXX") || { echo "ERROR: mktemp failed" >&2; return 1; }
    for pdb in "${pending_pdbs[@]:s:cnt}"; do
        cp "$pdb" "$TMPDIR_B/"
    done

    rc=0
    run_predict "$TMPDIR_B" || rc=$?
    (( rc != 0 )) && { echo "batch $(( b + 1 )) exited with status $rc"; failed_batches=$(( failed_batches + 1 )); }

    rm -rf "$TMPDIR_B"

    t_batch=$(( $(date +%s) - t_batch0 ))
    if (( ${#pending_lens[@]} )); then
        echo "BATCH_TIMING batch=$(( b + 1 )) n=$cnt distinct_len=${batch_distinct[b]}" \
             "len_min=${pending_lens[s]} len_max=${pending_lens[e-1]} wall_s=$t_batch rc=$rc"
    else
        echo "BATCH_TIMING batch=$(( b + 1 )) n=$cnt distinct_len=NA len_min=NA len_max=NA wall_s=$t_batch rc=$rc"
    fi

    # Pilot gate: outputs are counted by the same <input>_af2pred.pdb name the resume logic uses.
    if (( b < n_batches - 1 )) && [[ "${batch_pilot[b]}" == true ]] && (( PILOT_MIN_OK_PERCENT > 0 )); then
        n_ok=0
        for pdb in "${pending_pdbs[@]:s:cnt}"; do
            [[ -f "$SCRATCH_OUT/$(basename "$pdb" .pdb)_af2pred.pdb" ]] && n_ok=$(( n_ok + 1 ))
        done
        pct=$(( 100 * n_ok / cnt ))
        if (( rc != 0 || pct < PILOT_MIN_OK_PERCENT )); then
            echo "PILOT_CHECK FAILED: batch $(( b + 1 )) produced $n_ok/$cnt outputs ($pct%, need >= ${PILOT_MIN_OK_PERCENT}%), exit status $rc."
            echo "                    Not starting the remaining $(( total - e )) structures."
            echo "                    Inspect the log above and the pilot outputs in ${SYNC_DEST:-$SCRATCH_OUT}, fix the cause and resubmit;"
            echo "                    finished structures are skipped on resubmit. PILOT_MIN_OK_PERCENT=0 disables this check."
            pilot_failed=true
            break
        fi
        echo "PILOT_CHECK passed: batch $(( b + 1 )) produced $n_ok/$cnt outputs ($pct%), exit status $rc. Continuing."
    fi
done

if [[ "$pilot_failed" == true ]]; then
    echo "======================================================"
    echo "AFIG_wrapper aborted after a failed pilot batch"
    echo "Output folder: ${SYNC_DEST:-$SCRATCH_OUT}"
    return 1
fi

n_missing=0
for pdb in "${pending_pdbs[@]}"; do
    [[ -f "$SCRATCH_OUT/$(basename "$pdb" .pdb)_af2pred.pdb" ]] || n_missing=$(( n_missing + 1 ))
done
echo "======================================================"
echo "AFIG_wrapper finished: $n_batches batches, $failed_batches with a non-zero exit"
echo "Structures: $(( total - n_missing )) of $total predicted, $n_missing without output"
(( n_missing > 0 )) && echo "Failed structures and reasons: ${CHECKPOINT}.failed (details in this log)"
echo "Scores: $SCOREFILE"
if [[ -f "$SCOREFILE" ]]; then
    echo "Rows in score file: $(( $(wc -l < "$SCOREFILE") - 1 ))"
fi
echo "Output folder: ${SYNC_DEST:-$SCRATCH_OUT}"
(( failed_batches == 0 && n_missing == 0 )) || return 1
return 0
