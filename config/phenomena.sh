# =============================================================================
# phenomena.sh — the four linguistic phenomena, plus the factual setting.
#
# Each phenomenon X is defined by three things:
#   1. a syntactic filter over the parsed corpus, which splits it into
#      D_X (train_matched.txt) and D_base (train_clean.txt).  The filter
#      classes live in vendor/corpus_filtering/src/corpus_filtering/filters/base.py;
#      PHENOMENON_FILTER_DIR is the output directory that filter writes to.
#   2. a held-out minimal-pair test set.  Three phenomena take theirs from
#      BLiMP (PHENOMENON_PARADIGM selects the paradigm file); NPI uses the
#      manually constructed set in data/minimal_pairs_npi.tsv instead, for the
#      reasons given in the README under "Data".
#   3. a phenomenon key for probe_models.py (PHENOMENON_PROBE_KEY), which
#      decides which minimal pairs the Step-2 verification scores.
# =============================================================================

ALL_PHENOMENA=(binding_reflexives existential_there wh_islands npi)

phenomenon_config() {
    # Sets: PHENOMENON_FILTER_DIR PHENOMENON_PROBE_KEY PHENOMENON_PARADIGM
    local phenomenon="$1"
    case "$phenomenon" in
        binding_reflexives)
            # "The boy hurt himself." vs "*The boy hurt herself."
            PHENOMENON_FILTER_DIR="Binding-reflexive"
            PHENOMENON_PROBE_KEY="binding"
            PHENOMENON_PARADIGM='{"binding": ["principle_A_c_command"]}' ;;
        existential_there)
            # "There are few cars." vs "*There are all cars."
            PHENOMENON_FILTER_DIR="existential-there-quantifier"
            PHENOMENON_PROBE_KEY="quantifiers"
            PHENOMENON_PARADIGM='{"quantifiers": ["existential_there_quantifiers_1"]}' ;;
        wh_islands)
            # "Which girls ran?" vs "*Which ran girls?"
            PHENOMENON_FILTER_DIR="interrogative-wh-modifier"
            PHENOMENON_PROBE_KEY="island_effects"
            PHENOMENON_PARADIGM='{"island_effects": ["left_branch_island_simple_question"]}' ;;
        npi)
            # "I'm not playing anymore." vs "*I'm playing anymore."
            # The paradigm key is only a label here: prepare/linguistic.py serves
            # npi_licensing from data/minimal_pairs_npi.tsv and ignores the
            # (empty) subitem list.
            PHENOMENON_FILTER_DIR="nuke-npi"
            PHENOMENON_PROBE_KEY="manual_npi"
            PHENOMENON_PARADIGM='{"npi_licensing": []}' ;;
        facts)
            # BEAR entity-relation triplets; no syntactic filter and no BLiMP
            # paradigm — D_X is built per fact by run/filter_facts.sh.
            PHENOMENON_FILTER_DIR=""
            PHENOMENON_PROBE_KEY="facts_bear"
            PHENOMENON_PARADIGM="" ;;
        *)
            echo "ERROR: unknown phenomenon '$phenomenon'." >&2
            echo "       Valid: ${ALL_PHENOMENA[*]} facts" >&2
            return 1 ;;
    esac
}

# ── Attribution methods ──────────────────────────────────────────────────────
# CLI name → (python --method value, extra flags, output directory name).
# "kfac" is the fused application scheme — whitening folded into the projection
# matrices — which is the default everywhere except run/ablation_kfac.sh, where
# the alternatives are compared.  See that script's header for what the schemes
# are and why this one is the default.
ALL_METHODS=(gradsim trackstar kfac bm25)

method_config() {
    # Sets: METHOD_PY METHOD_ARGS METHOD_DIR
    local method="$1"
    case "$method" in
        gradsim)        METHOD_PY=vanilla;   METHOD_ARGS="";                                                        METHOD_DIR=vanilla ;;
        trackstar)      METHOD_PY=trackstar; METHOD_ARGS="";                                                        METHOD_DIR=trackstar ;;
        kfac)           METHOD_PY=ekfac;     METHOD_ARGS="--ekfac_method kfac  --ekfac_modified_projections true";   METHOD_DIR=kfac_effic_precond ;;
        bm25)           METHOD_PY=bm25;      METHOD_ARGS="";                                                        METHOD_DIR=bm25 ;;
        # ── ablation-only variants, see run/ablation_kfac.sh ─────────────────
        kfac-sketched)  METHOD_PY=ekfac;     METHOD_ARGS="--ekfac_method kfac  --ekfac_modified_projections false";  METHOD_DIR=kfac_full_precond ;;
        ekfac-sketched) METHOD_PY=ekfac;     METHOD_ARGS="--ekfac_method ekfac --ekfac_modified_projections false";  METHOD_DIR=ekfac_full_precond ;;
        *)
            echo "ERROR: unknown method '$method'. Valid: ${ALL_METHODS[*]} kfac-sketched ekfac-sketched" >&2
            return 1 ;;
    esac
}
