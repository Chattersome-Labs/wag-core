# Changelog

All notable changes to `wag_core` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-13

### Added
- **Bundled optional English stopword list** (`wag_core/data/stopwords_en.txt`, ~800 words).
  Opt in with `--exclude-words builtin:en` (or `builtin` for the default language). Removes
  low-information English "glue" words (function words, contractions, fillers, profanity
  intensifiers) before graph construction, so anchor-word harvesting stays focused on topical
  vocabulary and the iterative-pruning loop converges in fewer passes. By design the list holds
  only words that could never be a topic's core subject in any community — domain terms and even
  time-generics (*days*, *years*) are left to per-corpus dynamic pruning.
- `builtin_stopwords_path(lang='en')` helper (exported from the package) for programmatic use.
- `--exclude-words` now accepts a `builtin:<lang>` selector in addition to a file path.

### Notes
- **Non-breaking / language-agnostic default preserved.** No word list is applied unless opted
  into; runs without `--exclude-words` behave exactly as in 0.1.0. The bundled list is English
  only — for other languages, omit the flag or supply your own file.

## [0.1.0] - 2026-07-11

### Added
- Initial release: WAG topic detection with Leiden clustering, iterative pruning, cluster size
  cap, orphan removal, and per-topic `generic` flag.
