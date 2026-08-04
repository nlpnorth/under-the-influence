from abc import ABC, abstractmethod
from conllu import TokenList


class CorpusFilter(ABC):
    
    @abstractmethod
    def _exclude_sent(self, sent: TokenList) -> bool:
        """Return True if sentence should be excluded."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the phenomenon this filter targets."""
        ...


# ---------------------------------------------------------------------------
# HELPERS (self-contained, no external imports)
# ---------------------------------------------------------------------------

def _get_root(id_map):
    """Return the root token."""
    return next((t for t in id_map.values() if t["deprel"] == "root"), None)


def _get_clause_head_id(token_id, id_map):
    """Walk up the tree until we hit a VERB, AUX, or root."""
    current_id = id_map[token_id]["head"]
    seen = set()
    while current_id != 0 and current_id in id_map:
        if current_id in seen:
            return None
        seen.add(current_id)
        tok = id_map[current_id]
        if tok["upos"] in ("VERB", "AUX"):
            return current_id
        current_id = tok["head"]
    return current_id if current_id != 0 else None


def _has_neg_dependent(clause_head_id, id_map, neg_lemmas):
    """Return token id of negation dependent if present, else None."""
    for tid, token in id_map.items():
        if token["head"] != clause_head_id:
            continue
        if token["deprel"] in ("advmod", "advmod:neg", "aux", "aux:pass", "cop"):
            feats = token.get("feats") or {}
            if feats.get("Polarity") == "Neg":
                return tid
            if token.get("lemma") in neg_lemmas:
                return tid
    return None


def _is_negated_base(clause_head_id, id_map, neg_lemmas):
    """Return clause_head_id if the verb itself carries Polarity=Neg."""
    if clause_head_id is None or clause_head_id == 0:
        return None
    feats = id_map[clause_head_id].get("feats") or {}
    if feats.get("Polarity") == "Neg":
        return clause_head_id
    return None


def _negated_matrix_clause(clause_head_id, id_map, neg_lemmas):
    """Return neg token id if clause is embedded under a negated matrix verb."""
    clause_token = id_map.get(clause_head_id)
    if clause_token is None:
        return None
    if clause_token["deprel"] not in ("ccomp", "csubj", "csubj:cop", "xcomp"):
        return None
    matrix_head_id = clause_token["head"]
    if matrix_head_id == 0 or matrix_head_id not in id_map:
        return None
    feats = id_map[matrix_head_id].get("feats") or {}
    if feats.get("Polarity") == "Neg":
        return matrix_head_id
    return _has_neg_dependent(matrix_head_id, id_map, neg_lemmas)


def _detect_environment_en(clause_head_id, id_map):
    """
    Detect the licensing environment for English.
    Returns one of: 'neg', 'negated_matrix_clause', 'question',
                    'comparative', 'conditional', 'affirmative'
    """
    neg_lemmas = {"not", "never", "no"}

    if clause_head_id == 0 or clause_head_id is None:
        root = _get_root(id_map)
        if root is None:
            return "affirmative"
        clause_head_id = root["id"]

    # Question — ? punct child of clause head
    for tok in id_map.values():
        if tok.get("head") == clause_head_id and tok.get("form") == "?":
            return True

    # Question — interrogative wh-word anywhere
    for tok in id_map.values():
        feats = tok.get("feats") or {}
        if feats.get("PronType") == "Int":
            return True

    # Comparative
    for tok in id_map.values():
        if tok.get("lemma") == "than":
            return True

    # Conditional
    for tok in id_map.values():
        if tok.get("lemma") == "if":
            return True
        
    # Negation on the verb itself
    if _is_negated_base(clause_head_id, id_map, neg_lemmas):
        return True

    # Negation as dependent
    if _has_neg_dependent(clause_head_id, id_map, neg_lemmas):
        return True

    # Negated matrix clause
    if _negated_matrix_clause(clause_head_id, id_map, neg_lemmas):
        return True

    return False


# ---------------------------------------------------------------------------
# FILTERS
# ---------------------------------------------------------------------------

class NotFilter(CorpusFilter):
    
    @property
    def name(self) -> str:
        return "negation"
    
    def _exclude_sent(self, sent: TokenList) -> bool:
        return any(token["form"].lower() == "not" for token in sent)


class ExistentialThereQuantifierFilter(CorpusFilter):

    quantifiers = ["a", "an", "no", "some", "few", "many", "all", "most", "every", "each"]
    
    @property
    def name(self) -> str:
        return "existential-there-quantifier"
    
    def _exclude_sent(self, sent: TokenList) -> bool:
        id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}
        there_copulas = set()
        quantifier_head_head_verbs = set()

        for tok in id_map.values():
            head_id = tok["head"]
            head = id_map.get(head_id)
            if tok["lemma"].lower() == "there" and tok["deprel"] == "expl" \
                    and head is not None and head["lemma"] == "be":
                there_copulas.add(head["id"])
            if tok["lemma"] in self.quantifiers and head is not None \
                    and head["deprel"] is not None \
                    and head["deprel"].startswith("nsubj"):
                quantifier_head_head_verbs.add(head["head"])

        return bool(there_copulas & quantifier_head_head_verbs)


class BindingReflexive(CorpusFilter):

    #source wikipeadi
    reflexives = {"myself", "thyself", "thyselves", "himself", "herself", "itself", "ourselves", "yourself", "yourselves", "themself", "themselves", "oneself"}

    @property
    def name(self) -> str:
        return "Binding-reflexive"

    def _exclude_sent(self, sent: TokenList) -> bool:
        id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}
        for tok in id_map.values():
            feats = tok.get("feats") or {}
            if feats.get("Reflex") == "Yes":
                return True
            if (tok.get("form") or "").lower() in self.reflexives:
                return True
        return False


class InterrogativeWhModifierFilter(CorpusFilter):

    wh_lemmas = {"how", "whose", "what", "which"}
    n_upos = {"ADJ", "ADV", "NOUN", "PRON"}

    @property
    def name(self) -> str:
        return "interrogative-wh-modifier"

    def _exclude_sent(self, sent: TokenList) -> bool:
        id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}
        root = _get_root(id_map)
        if root is None:
            return False

        has_qmark = any(
            t["head"] == root["id"]
            and t["deprel"] == "punct"
            and "?" in (t["form"] or "")
            for t in id_map.values()
        )
        if not has_qmark:
            return False

        for n in id_map.values():
            if n["head"] != root["id"]:
                continue
            if n["upos"] not in self.n_upos:
                continue
            for w in id_map.values():
                if w["head"] != n["id"]:
                    continue
                feats = w.get("feats") or {}
                if feats.get("PronType") != "Int":
                    continue
                if (w["lemma"] or "").lower() in self.wh_lemmas:
                    return True
        return False


class LicensedNPI(CorpusFilter):

    NPI_lemmas = {
        "any", "ever", "remotely", "exactly", "squat", "yet",
        "anymore", "anyone", "anywhere", "anything", "anybody",
    }

    EXTRA_LICENSORS = {
    # negative quantifiers
    "nobody", "nowhere", "nothing", "neither", "nor",
    # downward entailing
    "no", "few", "little", "rarely", "seldom",
    "without", "doubt", "deny", "refuse", "fail",
    # scalar
    "barely", "hardly", "scarcely",
    # restrictor
    "only","barely"
}

    @property
    def name(self) -> str:
        return "Licensed-NPI"

    def _has_extra_licensor(self, id_map):
        """Return True if any extra licensor is present in the sentence."""
        return any(
            (token.get("lemma") or "").lower() in self.EXTRA_LICENSORS
            for token in id_map.values()
        )

    def _exclude_sent(self, sent: TokenList) -> bool:
        id_map = {t["id"]: t for t in sent if isinstance(t["id"], int)}


        for token in id_map.values():
            if token.get("lemma") not in self.NPI_lemmas:
                continue
            clause_head_id = _get_clause_head_id(token["id"], id_map)
            if clause_head_id is None:
                continue
            if self._has_extra_licensor(id_map):
                return True
            env = _detect_environment_en(clause_head_id, id_map)
            if env:
                return True

        return False


class NukeNPI(CorpusFilter):

    NPI_forms = {
        "any", "ever", "remotely", "exactly", "squat", "yet",
        "anymore", "anyone", "anywhere", "anything", "anybody",
    }

    @property
    def name(self) -> str:
        return "nuke-npi"

    def _exclude_sent(self, sent: TokenList) -> bool:
        return any(
            (tok.get("form") or "").lower() in self.NPI_forms
            for tok in sent
            if isinstance(tok["id"], int)
        )
