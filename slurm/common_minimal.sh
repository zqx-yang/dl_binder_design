#!/bin/bash
# Minimal job I/O helpers for AFIG_wrapper.sh. Sourced by the wrapper (or set COMMON_SH to your own file with the same functions).
# Results are written straight into OUT_DIR; the Slurm log is copied to SCRIPT_DIR every few seconds.
#
# Contract (what the wrapper calls and expects):
#   resolve_job_paths OUT_DIR SCRIPT_DIR   check/create the folders
#   setup_scratch_out                      sets SCRATCH_OUT, the folder the run writes to (here: OUT_DIR); SYNC_DEST is the final location
#   job_start / job_end                    announce start and end
#   seed_scratch_from_existing, start_incremental_sync, stop_incremental_sync_and_finalize   hooks for copying results between a
#                                          scratch folder and OUT_DIR; nothing to do when SCRATCH_OUT is OUT_DIR
#   start_log_mirror DIR SECONDS / stop_log_mirror   keep a copy of the Slurm log in DIR

resolve_job_paths() {
    mkdir -p "$1" "$2" || return 1
    case "$(cd "$2" && pwd -P)/" in
        "$(cd "$1" && pwd -P)/"*) echo "ERROR: SCRIPT_DIR must not be OUT_DIR or inside it" >&2; return 1 ;;
    esac
    OUT_REL="$1"; SCRIPT_REL="$2"
}
setup_scratch_out() { SCRATCH_OUT="$OUT_DIR"; SYNC_DEST="$OUT_DIR"; }
job_start() { echo "Job started: $(date)"; JOB_T0=$(date +%s); }
job_end()   { echo "Job ended: $(date) ($(( $(date +%s) - JOB_T0 )) s)"; }
seed_scratch_from_existing()         { :; }
start_incremental_sync()             { :; }
stop_incremental_sync_and_finalize() { :; }

start_log_mirror() {                 # $1 = folder for the copy, $2 = seconds between copies
    MIRROR_SRC=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p')
    [[ -n "$MIRROR_SRC" ]] || return 0
    MIRROR_DST="$1/$(basename "$MIRROR_SRC")"
    ( while sleep "$2"; do cp -f "$MIRROR_SRC" "$MIRROR_DST" 2>/dev/null; done ) &
    MIRROR_PID=$!
}
stop_log_mirror() {
    [[ -n "${MIRROR_PID:-}" ]] && kill "$MIRROR_PID" 2>/dev/null
    [[ -n "${MIRROR_SRC:-}" ]] && cp -f "$MIRROR_SRC" "$MIRROR_DST" 2>/dev/null
    return 0
}
