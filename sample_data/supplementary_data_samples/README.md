# Data samples (supplementary material)

Samples of the data used in the paper. All files are *excerpts* — the full
parsed corpus and the filtered training splits are too large to enclose and will
be released with the code upon publication.

Source corpus: the English part of **Common Corpus** (public domain /
permissively licensed text), sentence-segmented with NLTK and dependency-parsed
with MaChAmp (deberta-v3-large backbone), as described in the paper. The factual setting additionally uses an English Wikipedia
dump.

## `parsed_corpus/`

`parser_output_sample.conllu` — a parser output sample. 146 sentences from 22
distinct Common Corpus collections (USPTO, Caselaw Access Project, US-PD-Books,
US/NZ-PD-Newspapers, LoC-PD-Books, Eurlex, UN-Digital-Library, OpenAlex, SEC,
StackExchange, Wikipedia, Youtube-Commons, …), sampled by taking the first
document of each collection encountered in the corpus.

Standard CoNLL-U: `# newdoc id` / `# meta::*` headers carry the original Common
Corpus licensing and provenance metadata (identifier, collection, open type,
license, date, language), which we retain in our pre-processed version;
`# sent_id` / `# text` per sentence; then the ten CoNLL-U columns with UPOS,
XPOS, morphological features, head and dependency relation.

## `linguistic_filters/`

One directory per phenomenon, each holding the
output of that phenomenon's filter over one corpus shard of 1,799,943
sentences:

- `matched_sample.txt` — 150 sentences the filter matched, i.e. D_X, the split
  removed to build the No-X model.
- `base_sample.txt` — 150 sentences from D_base, what the No-X model is trained
  on after removal.

## `factual_filter/`

- `bear_facts/` — same three files for the BEAR-facts filter over the same
  shard.
- `wikipedia_corpus_sample.txt` — 150 sentences of the Wikipedia corpus, the
  second training corpus in the factual setting.

