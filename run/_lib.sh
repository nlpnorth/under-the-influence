# =============================================================================
# _lib.sh — shared helpers for the run/ entry points.  Not meant to be run
# directly; every run/*.sh sources this file as its first action.
# =============================================================================

set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BUNDLE_ROOT

source "$BUNDLE_ROOT/config/env.sh"
source "$BUNDLE_ROOT/config/budgets.sh"
source "$BUNDLE_ROOT/config/architectures.sh"
source "$BUNDLE_ROOT/config/phenomena.sh"

# Model architecture, set by every run/ script's --arch flag (default gpt2).
# It prefixes every model name, so architectures never overwrite each other.
: "${ARCH:=gpt2}"

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

# Like check_input, but for $ARCH_TOKENIZER: a hub-resolved tokenizer (e.g.
# smollm2) is an identifier, not a local path, so an existence check on it
# would always misreport [MISSING].
check_tokenizer() {
    if [[ "$ARCH_TOKENIZER_IS_LOCAL" == 1 ]]; then
        check_input "tokenizer" "$ARCH_TOKENIZER"
    else
        printf '  %-22s %s  [hub]\n' "tokenizer" "$ARCH_TOKENIZER"
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
    # No-X model f_θ⁻ trained with that phenomenon's D_X removed.  The $ARCH
    # prefix keeps two architectures trained on the same data apart.
    local corpus="$1" budget="$2" phenomenon="${3:-}"
    if [[ -z "$phenomenon" ]]; then
        echo "${ARCH}_${corpus}_${budget}_full"
    else
        echo "${ARCH}_${corpus}_${budget}_no_${phenomenon}"
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
        # Deliberately not `wc -l`: these files run to tens of GB, and counting
        # their lines over network storage takes minutes for a log message.
        log "Corpus already assembled ($(du -Lh "$out" | cut -f1)) → $out"
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
    log "Corpus assembled: $(du -Lh "$out" | cut -f1)"
}

# ── Training ─────────────────────────────────────────────────────────────────

train_model() {
    # train_model <model_name> <plain_text_file> [held_out_plain_text_file]
    #
    # Tokenize → shuffle → pick an evaluation set → train for one epoch.
    # Each stage is idempotent, so an interrupted job can simply be resubmitted.
    #
    # Sequence-level shuffling is done independently per model, so a full model
    # and its No-X counterpart do not share an example ordering — the two runs
    # should differ in their data, not in a coincidentally shared curriculum.
    #
    # The evaluation set comes from the held-out split the filter pipeline
    # already sets aside (10% of the parsed corpus, never part of any training
    # run) when the caller passes one.
    # Wikipedia has no held-out split — setup_wikipedia_data.py writes only
    # train_full.txt — so it falls back to that carve-out.
    local name="$1" plain_text="$2" held_out_text="${3:-}"
    local out_dir="$WORK_DIR/models/$name"
    local tokenized="$WORK_DIR/tokenized_data/${name}.txt"
    local shuffled="$WORK_DIR/shuffled_tokenized_data/${name}.txt"
    local eval_file="$WORK_DIR/tokenized_data_split/${name}_eval.txt"
    local train_file

    make_dir "$out_dir" "$(dirname "$tokenized")" "$(dirname "$shuffled")" "$(dirname "$eval_file")"

    # --- Tokenize (max 512 tokens per sequence, both architectures) ---
    if [[ -f "$tokenized" ]]; then
        log "Tokenized data exists, skipping."
    else
        log "Tokenizing → $tokenized"
        run_cmd "$GOLDFISH_PYTHON" "$GOLDFISH_DIR/tokenize_dataset.py" \
            --tokenizer="$ARCH_TOKENIZER" \
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

    # --- Evaluation set ---
    # EVAL_SEQUENCES caps the cost: the full held-out split runs to ~1.6M
    # sequences.  The cap is applied to the raw text first.  
    # Sentences are sampled
    # rather than taken from the head, which would draw them all from the
    # lowest-numbered chunk.
    local eval_seqs="${EVAL_SEQUENCES:-2000}"

    # A dry run reports the held-out path without requiring the file, which
    # concat_corpus has not written yet at that point; a real run falls back
    # only if the configured held-out text is genuinely missing, and says so.
    if [[ -n "$held_out_text" ]] && ! is_dry && [[ ! -s "$held_out_text" ]]; then
        log "WARNING: held-out text $held_out_text is missing or empty —"
        log "         falling back to carving the eval set out of training data."
        held_out_text=""
    fi

    if [[ -n "$held_out_text" ]]; then
        # Train on the whole corpus; evaluate on data it never contained.
        train_file="$shuffled"
        local ho_sample="$WORK_DIR/raw_text/${name}_heldout_sample.txt"
        local ho_tokenized="$WORK_DIR/tokenized_data/${name}_heldout.txt"
        # 40 sentences per sequence: ~13x headroom over the 512-token sequence
        # length at this corpus's ~39 tokens per sentence, so the sample still
        # yields $eval_seqs sequences after packing.
        local ho_sents=$((eval_seqs * 40))

        if [[ -f "$eval_file" ]]; then
            log "Eval set exists, skipping."
        elif is_dry; then
            echo "  [dry-run] would sample $ho_sents sentences from $held_out_text,"
            echo "  [dry-run]   tokenize them, and keep $eval_seqs sequences → $eval_file"
            echo "  [dry-run] train file is the full shuffled corpus: $train_file"
        else
            [[ -s "$ho_sample" ]] || shuf -n "$ho_sents" "$held_out_text" -o "$ho_sample"
            [[ -s "$ho_tokenized" ]] || "$GOLDFISH_PYTHON" "$GOLDFISH_DIR/tokenize_dataset.py" \
                --tokenizer="$ARCH_TOKENIZER" \
                --input_file="$ho_sample" \
                --output_file="$ho_tokenized" \
                --max_segments=-1 --max_seq_len=512 --max_examples=-1
            head -n "$eval_seqs" "$ho_tokenized" > "$eval_file"
            log "Eval: $(wc -l < "$eval_file") sequences from the held-out split."
        fi
    else
        # No held-out split available (Wikipedia): carve the eval set off the
        # end of the shuffled training data, as every earlier run did.
        train_file="$WORK_DIR/tokenized_data_split/${name}.txt"
        if [[ -f "$train_file" ]]; then
            log "Train/eval split exists, skipping."
        elif is_dry; then
            echo "  [dry-run] no held-out split — would carve $eval_seqs sequences"
            echo "  [dry-run]   off $shuffled into $train_file + $eval_file"
        else
            local total train_lines
            total=$(wc -l < "$shuffled")
            train_lines=$((total - eval_seqs))
            head -n "$train_lines" "$shuffled" > "$train_file"
            tail -n "$eval_seqs"   "$shuffled" > "$eval_file"
            log "No held-out split — Train: $train_lines sequences, Eval: $eval_seqs sequences."
        fi
    fi

    # --- Train ---
    # train_results.json is written only by trainer.save_metrics("train", ...)
    # after trainer.train() returns. config.json is NOT a completion marker:
    # the training script writes it (and the tokenizer files) to $out_dir at
    # the START of a fresh run too, so a job that crashed mid-training leaves
    # one behind — checking for it here would skip an incomplete model.
    if [[ -f "$out_dir/train_results.json" ]]; then
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
    # --standalone binds a fresh local rendezvous port per process instead of
    # the torchrun default (29500), which collides when concurrent array
    # tasks land on the same node. dataloader_num_workers is 2, not 4: cn19
    # warns 4 exceeds its suggested max, and concurrent array tasks training
    # at once make that memory overhead worth avoiding.
    run_cmd "$GOLDFISH_TORCHRUN" --standalone --nproc_per_node=1 \
        "$GOLDFISH_DIR/lm_code/run_transformer_language_modeling.py" \
        --tokenizer_name="$ARCH_TOKENIZER" \
        --config_name="$ARCH_MODEL_CONFIG" \
        --do_train --train_iterable --eval_iterable \
        --train_data_file="$train_file" \
        --eval_data_file="$eval_file" \
        --per_device_train_batch_size="$BUDGET_BATCH_PER_DEVICE" \
        --gradient_accumulation_steps="$BUDGET_GRAD_ACCUM" \
        --per_device_eval_batch_size=64 \
        --dataloader_num_workers=2 \
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
    # (probe_models.py, attribution) can load it alongside the model.  A local
    # tokenizer directory is copied as-is; a hub-resolved one is materialised
    # with save_pretrained.
    if [[ "$ARCH_TOKENIZER_IS_LOCAL" == 1 ]]; then
        run_cmd cp -r "$ARCH_TOKENIZER"/. "$out_dir"/
    else
        run_cmd "$GOLDFISH_PYTHON" -c \
            "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$ARCH_TOKENIZER').save_pretrained('$out_dir')"
    fi
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
