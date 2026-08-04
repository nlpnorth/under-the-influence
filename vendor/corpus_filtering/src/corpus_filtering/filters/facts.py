import re
import unicodedata
from collections import defaultdict
from typing import Callable, Iterable

from corpus_filtering.filters.base import CorpusFilter

# ── BEAR facts filter ─────────────────────────────────────────────────────────


def _build_bear_match_fn(
    pairs_with_aliases: list[tuple[list[str], list[str]]],
) -> Callable[[str], bool]:
    """Return a function that returns True when any (subject, object) pair
    co-occurs in the sentence.

    pairs_with_aliases: list of (subject_surface_forms, object_surface_forms).
    Each surface-form list should include the canonical label plus any Wikidata
    aliases.  The matching logic mirrors _build_match_fn: scan for the object
    first (as the more distinctive term) then verify the subject is present.
    """
    records = []
    for subj_terms, obj_terms in pairs_with_aliases:
        subj_cs, subj_ci = _split_case(subj_terms)
        obj_cs, obj_ci = _split_case(obj_terms)
        if not (subj_cs or subj_ci) or not (obj_cs or obj_ci):
            continue
        records.append({
            "subj_cs": subj_cs,
            "subj_ci": subj_ci,
            "obj_cs": obj_cs,
            "obj_ci": obj_ci,
        })

    all_obj_ci = sorted({t for r in records for t in r["obj_ci"]}, key=len, reverse=True)
    all_obj_cs = sorted({t for r in records for t in r["obj_cs"]}, key=len, reverse=True)
    obj_re_ci = _compile_alt(all_obj_ci, re.IGNORECASE)
    obj_re_cs = _compile_alt(all_obj_cs, 0)

    obj_to_records: dict[str, list] = defaultdict(list)
    for r in records:
        for t in r["obj_ci"]:
            obj_to_records[t].append(r)
        for t in r["obj_cs"]:
            obj_to_records[t].append(r)

    def _subj_present(text_raw: str, text_lower: str, rec: dict) -> bool:
        for t in rec["subj_cs"]:
            if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text_raw):
                return True
        for t in rec["subj_ci"]:
            if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text_lower):
                return True
        return False

    def match(text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        if obj_re_ci is not None:
            for m in obj_re_ci.finditer(text_lower):
                for rec in obj_to_records.get(m.group(0), []):
                    if _subj_present(text, text_lower, rec):
                        return True
        if obj_re_cs is not None:
            for m in obj_re_cs.finditer(text):
                for rec in obj_to_records.get(m.group(0), []):
                    if _subj_present(text, text_lower, rec):
                        return True
        return False

    return match


def _build_subject_match_fn(
    subj_terms_list: list[list[str]],
) -> Callable[[str], bool]:
    """Return a function that returns True when any subject surface form is
    present in the sentence, regardless of whether the object is present.

    subj_terms_list: list of subject surface-form lists (canonical label +
    Wikidata aliases). Unlike _build_bear_match_fn, this is a flat "any
    subject term present" check — no cross-referencing against an object is
    needed.
    """
    all_cs: list[str] = []
    all_ci: list[str] = []
    for terms in subj_terms_list:
        cs, ci = _split_case(terms)
        all_cs.extend(cs)
        all_ci.extend(ci)

    re_ci = _compile_alt(all_ci, re.IGNORECASE)
    re_cs = _compile_alt(all_cs, 0)

    def match(text: str) -> bool:
        if not text:
            return False
        if re_ci is not None and re_ci.search(text.lower()):
            return True
        if re_cs is not None and re_cs.search(text):
            return True
        return False

    return match


class BearFactsFilter(CorpusFilter):
    """Marks sentences where a confidently-learned BEAR subject and object co-occur,
    and optionally also sentences that merely mention an occurrence-eligible subject
    or object.

    Accepts pairs_with_aliases: a list of (subject_terms, object_terms) where
    each element is a list of surface forms (canonical label + Wikidata aliases).

    occurrence_subj_terms / occurrence_obj_terms (optional): surface-form lists
    for the subset of facts eligible for subject-only / object-only removal
    (e.g. low corpus subject-count or object-count). When given, a sentence is
    also excluded if it merely mentions one of these subjects or objects, even
    without the paired counterpart.
    """

    def __init__(
        self,
        pairs_with_aliases: list[tuple[list[str], list[str]]],
        occurrence_subj_terms: list[list[str]] | None = None,
        occurrence_obj_terms: list[list[str]] | None = None,
        name: str = "BearFacts",
    ) -> None:
        self._match = _build_bear_match_fn(pairs_with_aliases)
        self._occurrence_match = (
            _build_subject_match_fn(occurrence_subj_terms) if occurrence_subj_terms else None
        )
        self._occurrence_obj_match = (
            _build_subject_match_fn(occurrence_obj_terms) if occurrence_obj_terms else None
        )
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _exclude_sent(self, sent) -> bool:
        text = sent.metadata.get("text", "")
        if self._match(text):
            return True
        if self._occurrence_match and self._occurrence_match(text):
            return True
        return bool(self._occurrence_obj_match and self._occurrence_obj_match(text))

# Curated country/capital pairs, expanded with demonyms and historical names
# below. Used as a fallback source of entity surface forms alongside the
# Wikidata lookups, for entities whose aliases the API returns incompletely.
_PAIRS = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("United States", "Washington"),
    ("United Kingdom", "London"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Australia", "Canberra"),
    ("Brazil", "Brasilia"),
    ("India", "New Delhi"),
    ("Russia", "Moscow"),
    ("Canada", "Ottawa"),
    ("Mexico", "Mexico City"),
    ("Argentina", "Buenos Aires"),
    ("South Korea", "Seoul"),
    ("Netherlands", "Amsterdam"),
    ("Poland", "Warsaw"),
    ("Turkey", "Ankara"),
    ("Sweden", "Stockholm"),
    ("Norway", "Oslo"),
    ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"),
    ("Portugal", "Lisbon"),
    ("Switzerland", "Bern"),
    ("Austria", "Vienna"),
    ("Greece", "Athens"),
    ("Hungary", "Budapest"),
    ("Egypt", "Cairo"),
    ("South Africa", "Pretoria"),
    ("Nigeria", "Abuja"),
    ("Kenya", "Nairobi"),
    ("Thailand", "Bangkok"),
    ("Indonesia", "Jakarta"),
    ("Saudi Arabia", "Riyadh"),
    ("Colombia", "Bogota"),
    ("Peru", "Lima"),
    ("Czech Republic", "Prague"),
    ("Chile", "Santiago"),
    ("Romania", "Bucharest"),
    ("Ukraine", "Kyiv"),
    ("Belgium", "Brussels"),
    ("Ireland", "Dublin"),
    ("New Zealand", "Wellington"),
    ("Philippines", "Manila"),
    ("Pakistan", "Islamabad"),
    ("Vietnam", "Hanoi"),
    ("Israel", "Jerusalem"),
    ("Morocco", "Rabat"),
]

# Short all-caps abbreviations matched case-sensitively to avoid catching
# pronouns ("us", "uk") or URL fragments.
_CASE_SENSITIVE_TOKENS: frozenset[str] = frozenset(
    {
        "US",
        "U.S.",
        "U.S.A.",
        "USA",
        "UK",
        "U.K.",
        "GB",
        "NZ",
        "DC",
        "D.C.",
        "PRC",
        "ROK",
        "KSA",
        "FRG",
        "BRD",
        "RSA",
        "CDMX",
        "USSR",
    }
)


def _normalize(s: str) -> str:
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


def _is_case_sensitive(term: str) -> bool:
    if term in _CASE_SENSITIVE_TOKENS:
        return True
    return (
        len(term) <= 4
        and term.isascii()
        and term.isupper()
        and any(c.isalpha() for c in term)
    )


def _compile_alt(terms: Iterable[str], flags: int = 0) -> re.Pattern | None:
    terms = [t for t in terms if t]
    if not terms:
        return None
    terms = sorted(set(terms), key=len, reverse=True)
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)"
    return re.compile(pattern, flags)


def _split_case(terms: Iterable[str]) -> tuple[list[str], list[str]]:
    """Partition terms into (case_sensitive, case_insensitive)."""
    cs: list[str] = []
    ci: list[str] = []
    for t in terms:
        if not t:
            continue
        if _is_case_sensitive(t):
            cs.append(t)
        else:
            ci.append(t.lower())
            norm = _normalize(t)
            if norm and norm != t.lower():
                ci.append(norm)
    return cs, ci


def _build_match_fn(
    pairs,
    demonyms,
    capital_aliases=None,
    country_aliases=None,
    historical_capitals=None,
) -> Callable[[str], bool]:
    """Return a function that returns True if a capital and its country/demonym
    co-occur in the sentence."""
    capital_aliases = capital_aliases or {}
    country_aliases = country_aliases or {}
    historical_capitals = historical_capitals or {}

    records = []
    for country, capital in pairs:
        country_raw = {country}
        country_raw.update(country_aliases.get(country, []))
        country_raw.update(demonyms.get(country, []))

        capital_raw = {capital}
        capital_raw.update(capital_aliases.get(country, []))
        capital_raw.update(historical_capitals.get(country, []))

        country_cs, country_ci = _split_case(country_raw)
        capital_cs, capital_ci = _split_case(capital_raw)

        records.append(
            {
                "country": country,
                "country_cs": country_cs,
                "country_ci": country_ci,
                "capital_cs": capital_cs,
                "capital_ci": capital_ci,
            }
        )

    all_caps_ci = sorted(
        {t for r in records for t in r["capital_ci"]}, key=len, reverse=True
    )
    all_caps_cs = sorted(
        {t for r in records for t in r["capital_cs"]}, key=len, reverse=True
    )
    cap_re_ci = _compile_alt(all_caps_ci, re.IGNORECASE)
    cap_re_cs = _compile_alt(all_caps_cs, 0)

    cap_to_country_terms: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        for cap in r["capital_ci"]:
            cap_to_country_terms[cap].append(r)
        for cap in r["capital_cs"]:
            cap_to_country_terms[cap].append(r)

    def _country_present(text_raw: str, text_lower: str, rec: dict) -> bool:
        for term in rec["country_cs"]:
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text_raw):
                return True
        for term in rec["country_ci"]:
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text_lower):
                return True
        return False

    def match(text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        if cap_re_ci is not None:
            for m in cap_re_ci.finditer(text_lower):
                for rec in cap_to_country_terms.get(m.group(0), []):
                    if _country_present(text, text_lower, rec):
                        return True
        if cap_re_cs is not None:
            for m in cap_re_cs.finditer(text):
                for rec in cap_to_country_terms.get(m.group(0), []):
                    if _country_present(text, text_lower, rec):
                        return True
        return False

    return match


class CapitalFactsFilter(CorpusFilter):
    """Excludes sentences where a capital city and its country/demonym co-occur."""

    def __init__(
        self,
        pairs=None,
        demonyms=None,
        capital_aliases=None,
        country_aliases=None,
        historical_capitals=None,
    ):
        self._match = _build_match_fn(
            pairs if pairs is not None else _PAIRS,
            demonyms if demonyms is not None else _DEMONYMS_EXPANDED,
            capital_aliases if capital_aliases is not None else _CAPITAL_ALIASES,
            country_aliases if country_aliases is not None else _COUNTRY_ALIASES,
            historical_capitals
            if historical_capitals is not None
            else _HISTORICAL_CAPITALS,
        )

    @property
    def name(self) -> str:
        return "CapitalFacts"

    def _exclude_sent(self, sent) -> bool:
        text = sent.metadata.get("text", "")
        return self._match(text)


# ---------------------------------------------------------------------------
# Data: capital aliases, country aliases, demonyms, historical capitals.
# ---------------------------------------------------------------------------

_CAPITAL_ALIASES: dict[str, list[str]] = {
    "France": ["Paris"],
    "Germany": ["Berlin"],
    "Japan": ["Tokyo", "東京", "Tōkyō"],
    "China": ["Beijing", "Peking", "北京"],
    "United States": ["Washington", "Washington D.C.", "Washington DC", "D.C.", "DC"],
    "United Kingdom": ["London"],
    "Italy": ["Rome", "Roma"],
    "Spain": ["Madrid"],
    "Australia": ["Canberra"],
    "Brazil": ["Brasilia", "Brasília"],
    "India": ["New Delhi", "Delhi"],
    "Russia": ["Moscow", "Москва", "Moskva"],
    "Canada": ["Ottawa"],
    "Mexico": ["Mexico City", "Ciudad de México", "CDMX", "México D.F.", "Mexico D.F."],
    "Argentina": ["Buenos Aires"],
    "South Korea": ["Seoul", "서울"],
    "Netherlands": ["Amsterdam", "The Hague", "Den Haag"],
    "Poland": ["Warsaw", "Warszawa"],
    "Turkey": ["Ankara"],
    "Sweden": ["Stockholm"],
    "Norway": ["Oslo"],
    "Denmark": ["Copenhagen", "København", "Kobenhavn"],
    "Finland": ["Helsinki", "Helsingfors"],
    "Portugal": ["Lisbon", "Lisboa"],
    "Switzerland": ["Bern", "Berne"],
    "Austria": ["Vienna", "Wien"],
    "Greece": ["Athens", "Athína", "Athina", "Αθήνα"],
    "Hungary": ["Budapest"],
    "Egypt": ["Cairo", "Al-Qāhirah", "القاهرة"],
    "South Africa": ["Pretoria", "Cape Town", "Bloemfontein"],
    "Nigeria": ["Abuja"],
    "Kenya": ["Nairobi"],
    "Thailand": ["Bangkok", "Krung Thep", "กรุงเทพ"],
    "Indonesia": ["Jakarta", "Djakarta"],
    "Saudi Arabia": ["Riyadh", "Ar-Riyāḍ", "الرياض"],
    "Colombia": ["Bogota", "Bogotá", "Santa Fe de Bogotá"],
    "Peru": ["Lima"],
    "Czech Republic": ["Prague", "Praha"],
    "Chile": ["Santiago", "Santiago de Chile"],
    "Romania": ["Bucharest", "București", "Bucuresti"],
    "Ukraine": ["Kyiv", "Kiev", "Київ"],
    "Belgium": ["Brussels", "Bruxelles", "Brussel"],
    "Ireland": ["Dublin", "Baile Átha Cliath"],
    "New Zealand": ["Wellington"],
    "Philippines": ["Manila"],
    "Pakistan": ["Islamabad"],
    "Vietnam": ["Hanoi", "Hà Nội"],
    "Israel": ["Jerusalem", "Tel Aviv", "Yerushalayim", "ירושלים"],
    "Morocco": ["Rabat", "الرباط"],
}


_COUNTRY_ALIASES: dict[str, list[str]] = {
    "France": ["France"],
    "Germany": ["Germany", "Deutschland", "FRG", "BRD"],
    "Japan": ["Japan", "Nippon", "Nihon", "日本"],
    "China": ["China", "PRC", "People's Republic of China", "中国", "中國"],
    "United States": [
        "United States",
        "USA",
        "U.S.A.",
        "U.S.",
        "America",
        "United States of America",
    ],
    "United Kingdom": [
        "United Kingdom",
        "UK",
        "U.K.",
        "Britain",
        "Great Britain",
        "GB",
        "England",
    ],
    "Italy": ["Italy", "Italia"],
    "Spain": ["Spain", "España", "Espana"],
    "Australia": ["Australia"],
    "Brazil": ["Brazil", "Brasil"],
    "India": ["India", "Bharat", "भारत"],
    "Russia": [
        "Russia",
        "Russian Federation",
        "Россия",
        "Rossiya",
        "USSR",
        "Soviet Union",
    ],
    "Canada": ["Canada"],
    "Mexico": ["Mexico", "México"],
    "Argentina": ["Argentina"],
    "South Korea": [
        "South Korea",
        "Korea",
        "ROK",
        "Republic of Korea",
        "대한민국",
        "한국",
    ],
    "Netherlands": ["Netherlands", "Holland", "Nederland"],
    "Poland": ["Poland", "Polska"],
    "Turkey": ["Turkey", "Türkiye", "Turkiye"],
    "Sweden": ["Sweden", "Sverige"],
    "Norway": ["Norway", "Norge", "Noreg"],
    "Denmark": ["Denmark", "Danmark"],
    "Finland": ["Finland", "Suomi"],
    "Portugal": ["Portugal"],
    "Switzerland": ["Switzerland", "Schweiz", "Suisse", "Svizzera"],
    "Austria": ["Austria", "Österreich", "Osterreich"],
    "Greece": ["Greece", "Hellas", "Ελλάδα", "Ellada"],
    "Hungary": ["Hungary", "Magyarország", "Magyarorszag"],
    "Egypt": ["Egypt", "Misr", "مصر"],
    "South Africa": ["South Africa", "RSA", "Suid-Afrika"],
    "Nigeria": ["Nigeria"],
    "Kenya": ["Kenya"],
    "Thailand": ["Thailand", "Siam", "ประเทศไทย"],
    "Indonesia": ["Indonesia"],
    "Saudi Arabia": ["Saudi Arabia", "KSA", "Kingdom of Saudi Arabia", "السعودية"],
    "Colombia": ["Colombia"],
    "Peru": ["Peru", "Perú"],
    "Czech Republic": ["Czech Republic", "Czechia", "Česko", "Česká republika"],
    "Chile": ["Chile"],
    "Romania": ["Romania", "România"],
    "Ukraine": ["Ukraine", "Україна", "Ukrayina"],
    "Belgium": ["Belgium", "België", "Belgique", "Belgien"],
    "Ireland": ["Ireland", "Éire", "Eire"],
    "New Zealand": ["New Zealand", "NZ", "Aotearoa"],
    "Philippines": ["Philippines", "Pilipinas"],
    "Pakistan": ["Pakistan"],
    "Vietnam": ["Vietnam", "Viet Nam", "Việt Nam"],
    "Israel": ["Israel", "ישראל", "Yisrael"],
    "Morocco": ["Morocco", "Maroc", "Al-Maghrib", "المغرب"],
}


_DEMONYMS_EXPANDED: dict[str, list[str]] = {
    "France": ["French", "Frenchman", "Frenchwoman", "Frenchmen", "Frenchwomen"],
    "Germany": ["German", "Germans"],
    "Japan": ["Japanese"],
    "China": ["Chinese"],
    "United States": ["American", "Americans", "US", "U.S.", "USA"],
    "United Kingdom": [
        "British",
        "Brit",
        "Brits",
        "Briton",
        "Britons",
        "English",
        "Englishman",
        "Englishwoman",
        "UK",
        "U.K.",
    ],
    "Italy": ["Italian", "Italians"],
    "Spain": ["Spanish", "Spaniard", "Spaniards"],
    "Australia": ["Australian", "Australians", "Aussie", "Aussies"],
    "Brazil": ["Brazilian", "Brazilians"],
    "India": ["Indian", "Indians"],
    "Russia": ["Russian", "Russians"],
    "Canada": ["Canadian", "Canadians"],
    "Mexico": ["Mexican", "Mexicans"],
    "Argentina": ["Argentine", "Argentinian", "Argentines", "Argentinians"],
    "South Korea": ["Korean", "Koreans", "South Korean", "South Koreans"],
    "Netherlands": ["Dutch", "Dutchman", "Dutchwoman", "Netherlander", "Hollander"],
    "Poland": ["Polish", "Pole", "Poles"],
    "Turkey": ["Turkish", "Turk", "Turks"],
    "Sweden": ["Swedish", "Swede", "Swedes"],
    "Norway": ["Norwegian", "Norwegians"],
    "Denmark": ["Danish", "Dane", "Danes"],
    "Finland": ["Finnish", "Finn", "Finns"],
    "Portugal": ["Portuguese"],
    "Switzerland": ["Swiss"],
    "Austria": ["Austrian", "Austrians"],
    "Greece": ["Greek", "Greeks", "Hellenic"],
    "Hungary": ["Hungarian", "Hungarians", "Magyar", "Magyars"],
    "Egypt": ["Egyptian", "Egyptians"],
    "South Africa": ["South African", "South Africans"],
    "Nigeria": ["Nigerian", "Nigerians"],
    "Kenya": ["Kenyan", "Kenyans"],
    "Thailand": ["Thai", "Thais", "Siamese"],
    "Indonesia": ["Indonesian", "Indonesians"],
    "Saudi Arabia": ["Saudi", "Saudis", "Saudi Arabian"],
    "Colombia": ["Colombian", "Colombians"],
    "Peru": ["Peruvian", "Peruvians"],
    "Czech Republic": ["Czech", "Czechs"],
    "Chile": ["Chilean", "Chileans"],
    "Romania": ["Romanian", "Romanians"],
    "Ukraine": ["Ukrainian", "Ukrainians"],
    "Belgium": ["Belgian", "Belgians"],
    "Ireland": ["Irish", "Irishman", "Irishwoman"],
    "New Zealand": ["New Zealander", "New Zealanders", "Kiwi", "Kiwis"],
    "Philippines": ["Filipino", "Filipinos", "Filipina", "Philippine", "Pinoy"],
    "Pakistan": ["Pakistani", "Pakistanis"],
    "Vietnam": ["Vietnamese"],
    "Israel": ["Israeli", "Israelis"],
    "Morocco": ["Moroccan", "Moroccans"],
}


_HISTORICAL_CAPITALS: dict[str, list[str]] = {
    "Germany": ["Bonn"],
    "Brazil": ["Rio de Janeiro", "Salvador"],
    "Russia": ["St. Petersburg", "Saint Petersburg", "Leningrad", "Petrograd"],
    "Pakistan": ["Karachi"],
    "Turkey": ["Istanbul", "Constantinople"],
    "Japan": ["Kyoto", "Edo"],
    "China": ["Nanjing", "Nanking", "Xi'an", "Chang'an"],
    "Italy": ["Florence", "Turin"],
    "India": ["Calcutta", "Kolkata"],
    "Australia": ["Melbourne"],
    "Nigeria": ["Lagos"],
    "South Korea": ["Busan"],
    "Vietnam": ["Saigon", "Ho Chi Minh City"],
    "Greece": ["Nafplio"],
    "Poland": ["Krakow", "Kraków"],
}


def get_predefined_aliases(label: str) -> list[str]:
    """Return any predefined surface forms for a label, or an empty list.

    Checks _COUNTRY_ALIASES, _DEMONYMS_EXPANDED, _CAPITAL_ALIASES (values),
    and _HISTORICAL_CAPITALS (values) so that BEAR entities which overlap with
    the capital-facts vocabulary get their full alias coverage for free.
    """
    terms: list[str] = []

    # Country label → country surface forms + demonyms
    if label in _COUNTRY_ALIASES:
        terms.extend(_COUNTRY_ALIASES[label])
    if label in _DEMONYMS_EXPANDED:
        terms.extend(_DEMONYMS_EXPANDED[label])

    # Capital label → find which country it belongs to, add capital aliases
    for caps in _CAPITAL_ALIASES.values():
        if label in caps:
            terms.extend(caps)
            break

    # Historical capital label → same treatment
    for hist_caps in _HISTORICAL_CAPITALS.values():
        if label in hist_caps:
            terms.extend(hist_caps)
            break

    # Always include the label itself
    if label not in terms:
        terms.insert(0, label)

    return list(dict.fromkeys(terms))  # deduplicate, preserve order
