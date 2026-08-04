# =============================================================================
# _lib.sh — shared helpers for the run/ entry points.  Not meant to be run
# directly; every run/*.sh sources this file as its first action.
# =============================================================================

set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BUNDLE_ROOT

source "$BUNDLE_ROOT/config/env.sh"
source "$BUNDLE_ROOT/config/budgets.sh"
source "$BUNDLE_ROOT/config/phenomena.sh"

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=60          # keep \r progress lines out of SLURM logs
export PYTHONPATH="$BUNDLE_ROOT/src:${PYTHONPATH:-}"

# ── Dry run ──────────────────────────────────────────────────────────────────
# With DRY_RUN=1 every command is printed instead of executed, and every
# directory creation is skipped.  Use it to check that a given
# (corpus, budget, phenomenon) combination resolves to the paths you expect.

is_dry() { [[ "${DRY_RUN:-0}" == "1" ]]; }

run_cmd() {
    if is_dry; then
        printf '  [dry-run] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

make_dir() { is_dry || mkdir -p "$@"; }

# Report an input path and whether it currently exists (dry runs only).
check_input() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then
        printf '  %-22s %s  [ok]\n' "$label" "$path"
    else
        printf '  %-22s %s  [MISSING]\n' "$label" "$path"
    fi
}

show_help() {
    # Print the script's leading comment block — everything between the two
    # "# ====" rules at the top of the file — so --help cannot drift out of
    # sync with the documentation the way a hardcoded line range would.
    # (two plain -e expressions rather than one alternation: BSD sed, which is
    # what macOS ships, does not support GNU's \| in a basic regex)
    awk '/^# ={10,}$/ { n++; next } n == 1' "$1" | sed -e 's/^# //' -e 's/^#$//'
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Note on the ${arr[@]+"${arr[@]}"} idiom used in the run/ scripts: under
# `set -u`, bash 3.2 (still the default on macOS) treats "${arr[@]}" on an
# EMPTY array as an unbound variable and aborts.  The +-form expands to nothing
# when the array is empty and to the quoted elements otherwise.

banner() {
    echo "=============================================================="
    echo "$*"
    echo "Node       : $(hostname)"
    echo "SLURM Job  : ${SLURM_JOB_ID:-<none, running locally>}"
    echo "Start time : $(date)"
    is_dry && echo "MODE       : DRY RUN — no commands executed, no files written"
    echo "=============================================================="
}

# ── Naming ───────────────────────────────────────────────────────────────────
# One place defines what a model directory and an experiment are called, so
# train_*.sh and attribute_*.sh cannot drift apart.

model_name() {
    # model_name <corpus> <budget> [phenomenon]
    # Omitting the phenomenon names the full model f_θ; supplying one names the
    # No-X model f_θ⁻ trained with that phenomenon's D_X removed.
    local corpus="$1" budget="$2" phenomenon="${3:-}"
    if [[ -z "$phenomenon" ]]; then
        echo "${corpus}_${budget}_full"
    else
        echo "${corpus}_${budget}_no_${phenomenon}"
    fi
}

model_path() { echo "$WORK_DIR/models/$(model_name "$@")"; }

experiment_name() {
    # experiment_name <corpus> <budget> <phenomenon> <max_base>
    echo "attr_${1}_${2}_${3}_base${4}"
}

# ── Scratch ──────────────────────────────────────────────────────────────────
# Gradient index builds do many small random writes into large mmap files.  On
# NFS that costs ~30s/batch instead of ~0.3s, so attribution runs against
# node-local disk and rsyncs its results back on exit.

setup_scratch() {
    if is_dry; then
        SCRATCH_DIR="$SCRATCH_ROOT/<job-id>"
        echo "  scratch dir            $SCRATCH_DIR (would be created)"
        return 0
    fi
    SCRATCH_DIR="$SCRATCH_ROOT/${SLURM_JOB_ID:-$$}"
    if ! mkdir -p "$SCRATCH_DIR" 2>/dev/null; then
        log "WARNING: $SCRATCH_ROOT not writable on $(hostname); falling back to \$TMPDIR"
        SCRATCH_DIR="$(mktemp -d)"
    fi
    log "Scratch dir: $SCRATCH_DIR ($(df -h "$SCRATCH_DIR" | awk 'NR==2{print $4}') free)"
}

# Sync scratch back to a durable location and remove it.  Installed as an EXIT
# trap by the attribution scripts so a job that dies still keeps its results.
sync_and_cleanup() {
    local status=$?
    is_dry && exit "$status"
    log "EXIT trap fired with status $status"
    if [[ $status -eq 0 ]]; then
        log "Syncing artifacts → $ATTRIBUTION_DIR"
        mkdir -p "$ATTRIBUTION_DIR"
        rsync -a --timeout=120 "$SCRATCH_DIR/" "$ATTRIBUTION_DIR/" || \
            log "WARNING: final rsync failed — artifacts left in $SCRATCH_DIR"
    else
        log "Job failed — leaving $SCRATCH_DIR in place for inspection"
        exit "$status"
    fi
    rm -rf "$SCRATCH_DIR" || true
    exit "$status"
}

# ── Corpus assembly ──────────────────────────────────────────────────────────

concat_corpus() {
    # concat_corpus <out_file> <corpus_root> <chunks> <relative_file>
    # Concatenates $corpus_root/chunk_XX/<relative_file> across chunks.
    # Idempotent: an existing non-empty output is left alone.
    local out="$1" root="$2" chunks="$3" rel="$4"

    if [[ -s "$out" ]]; then
        log "Corpus already assembled ($(wc -l < "$out") lines) → $out"
        return 0
    fi
    log "Assembling corpus → $out"
    make_dir "$(dirname "$out")"
    local c src missing=0
    for c in $chunks; do
        src="$root/chunk_${c}/${rel}"
        if [[ -f "$src" ]]; then
            is_dry || cat "$src" >> "$out"
        else
            echo "  WARNING: $src not found — skipping chunk_${c}"
            missing=$((missing + 1))
        fi
    done
    is_dry && { echo "  [dry-run] would concatenate $(echo "$chunks" | wc -w) chunks of $rel"; return 0; }
    [[ $missing -gt 0 ]] && log "WARNING: $missing chunk(s) missing"
    log "Corpus assembled: $(wc -l < "$out") lines"
}

# ── Training ─────────────────────────────────────────────────────────────────

train_model() {
    # train_model <model_name> <plain_text_file>
    #
    # Tokenize → shuffle → hold out 2,000 sequences → train for one epoch.
    # Each stage is idempotent, so an interrupted job can simply be resubmitted.
    #
    # Sequence-level shuffling happens before the train/eval split and is done
    # independently per model, so a full model and its No-X counterpart do not
    # share an example ordering — the two runs should differ in their data, not
    # in a coincidentally shared curriculum.
    local name="$1" plain_text="$2"
    local out_dir="$WORK_DIR/models/$name"
    local tokenized="$WORK_DIR/tokenized_data/${name}.txt"
    local shuffled="$WORK_DIR/shuffled_tokenized_data/${name}.txt"
    local train_file="$WORK_DIR/tokenized_data_split/${name}.txt"
    local eval_file="$WORK_DIR/tokenized_data_split/${name}_eval2k.txt"

    make_dir "$out_dir" "$(dirname "$tokenized")" "$(dirname "$shuffled")" "$(dirname "$train_file")"

    # --- Tokenize (max 512 tokens per sequence, both architectures) ---
    if [[ -f "$tokenized" ]]; then
        log "Tokenized data exists, skipping."
    else
        log "Tokenizing → $tokenized"
        run_cmd "$GOLDFISH_PYTHON" "$GOLDFISH_DIR/tokenize_dataset.py" \
            --tokenizer="$TOKENIZER" \
            --input_file="$plain_text" \
            --output_file="$tokenized" \
            --max_segments=-1 --max_seq_len=512 --max_examples=-1
    fi

    # --- Shuffle at the sequence level ---
    if [[ -f "$shuffled" ]]; then
        log "Shuffled data exists, skipping."
    else
        log "Shuffling → $shuffled"
        run_cmd shuf "$tokenized" -o "$shuffled"
    fi

    # --- Split off a 2,000-sequence held-out set ---
    if [[ -f "$train_file" ]]; then
        log "Train/eval split exists, skipping."
    elif is_dry; then
        echo "  [dry-run] would split $shuffled into $train_file + $eval_file (2,000 held out)"
    else
        local total eval_lines=2000 train_lines
        total=$(wc -l < "$shuffled")
        train_lines=$((total - eval_lines))
        head -n "$train_lines" "$shuffled" > "$train_file"
        tail -n "$eval_lines"  "$shuffled" > "$eval_file"
        log "Train: $train_lines sequences, Eval: $eval_lines sequences."
    fi

    # --- Train ---
    if [[ -f "$out_dir/config.json" ]]; then
        log "Model already trained at $out_dir, skipping."
        return 0
    fi

    local n_train max_steps warmup_steps eval_steps
    if is_dry; then
        n_train="<N>"; max_steps="<ceil(N/$BUDGET_BATCH_SIZE)>"
        warmup_steps="<10% of max_steps>"; eval_steps="<N/(2*batch)>"
    else
        n_train=$(wc -l < "$train_file")
        max_steps=$(python3 -c "import math; print(math.ceil($n_train / $BUDGET_BATCH_SIZE))")
        warmup_steps=$(python3 -c "print(int($max_steps * 0.10))")
        eval_steps=$(python3 -c "print(int($n_train / $BUDGET_BATCH_SIZE / 2))")
        log "N_TRAIN=$n_train  MAX_STEPS=$max_steps  WARMUP=$warmup_steps  SEED=$BUDGET_SEED"
    fi

    log "Training $name ($BUDGET_ARCH, one epoch)"
    run_cmd "$GOLDFISH_TORCHRUN" --nproc_per_node=1 \
        "$GOLDFISH_DIR/lm_code/run_transformer_language_modeling.py" \
        --tokenizer_name="$TOKENIZER" \
        --config_name="$BUDGET_MODEL_CONFIG" \
        --do_train --train_iterable --eval_iterable \
        --train_data_file="$train_file" \
        --eval_data_file="$eval_file" \
        --per_device_train_batch_size="$BUDGET_BATCH_PER_DEVICE" \
        --gradient_accumulation_steps="$BUDGET_GRAD_ACCUM" \
        --per_device_eval_batch_size=64 \
        --dataloader_num_workers=4 \
        --evaluation_strategy=steps --save_strategy=steps \
        --eval_steps="$eval_steps" --save_steps="$eval_steps" \
        --max_steps="$max_steps" \
        --warmup_steps="$warmup_steps" \
        --learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
        --bf16 \
        --seed="$BUDGET_SEED" \
        --override_n_examples="$n_train" \
        --output_dir="$out_dir"

    # The tokenizer travels with the checkpoint so downstream steps
    # (probe_models.py, attribution) can load it with the model.
    run_cmd cp "$TOKENIZER"/. "$out_dir"/ -r
    log "Training done → $out_dir"
}

# ── Argument parsing ─────────────────────────────────────────────────────────

require_arg() {
    local name="$1" value="$2"
    if [[ -z "$value" ]]; then
        echo "ERROR: --${name} is required. See --help." >&2
        exit 2
    fi
}
