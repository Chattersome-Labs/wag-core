# wag_core

**Weighted Adjacency Graph (WAG)** topic detection engine for text collections.

## Abstract

Weighted Adjacency Graph (WAG) modeling introduces a population-weighted, graph-based framework for topic detection in document collections. Unlike traditional probabilistic models that require pre-specifying the number of topics, WAG modeling uses a minimum unique-user threshold as its primary control parameter, allowing topics to emerge naturally from vocabulary shared across participants in a discourse community. The method constructs a weighted adjacency graph over population-qualified n-grams, applies network dismantling to reduce the influence of overconnected vocabulary, and employs community detection to identify coherent topic clusters. Documents lacking population-qualified terms define a principled "miscellaneous" boundary, enabling exclusion of idiosyncratic or low-signal material prior to summarization. Because the approach relies solely on internal co-occurrence structure and author participation counts, it is domain-agnostic and applicable to multilingual corpora as well as structured token sequences such as system logs. WAG modeling offers an interpretable, parameter-light alternative to optimization-heavy topic models, emphasizing emergence, transparency, and population-representative discourse structure.

## Overview

Detects topics by building word co-occurrence graphs from a corpus of posts and clustering them with the [Leiden algorithm](https://www.nature.com/articles/s41598-019-41695-z). Works with any language, any domain — social media, logs, technical documentation, or anything else made of text.

## How It Works

1. **Tokenize** input posts (lightweight regex, no NLP dependencies)
2. **Detect stopwords** automatically using the document frequency elbow method (language-agnostic)
3. **Select anchor words** — tokens used by a minimum percentage of unique users
4. **Build a co-occurrence graph** — word pairs within a sliding window, weighted by distinct user count
5. **Cluster with Leiden** — community detection finds natural topic groupings
6. **Iterative pruning** — optionally removes words that bridge too many clusters, re-runs until clean
7. **Classify posts** — each post scored against each cluster; assigned Strong, Weak, or None
8. **N-gram analysis** — top unigrams, bigrams, trigrams per topic
9. **Output** — TSV files for analysis, Gephi-ready graph files, per-topic post lists

## Requirements

```
pip install igraph leidenalg numpy
```

Python 3.8+. No other dependencies.

## Input Format

Tab-separated, one post per line:

```
user_id<TAB>post_text
```

No header row. User IDs can be any string. Example:

```
alice	Just set up Tailscale on my home server, SSH works great now
bob	Anyone tried running Proxmox in a VM? Curious about nested virtualization
alice	The supply chain attack on that npm package was wild
```

## Usage

### Basic run

```bash
python3 -m wag_core --input posts.tsv --output-dir ./output
```

### Typical run with all defaults shown

```bash
python3 -m wag_core \
  --input posts.tsv \
  --output-dir ./output \
  --min-user-pct 1.0 \
  --min-pair-user-pct 0.1 \
  --radius 3 \
  --stopword-sensitivity 0.6 \
  --resolution 1.0 \
  --weight-by users \
  --max-adjacent-topics 3
```

### Tuning for more or fewer topics

More topics (tighter pairs, sparser graph):

```bash
python3 -m wag_core \
  --input my_data/reddit_posts.tsv \
  --output-dir ./results_fine \
  --min-pair-user-pct 0.3 \
  --radius 2
```

Fewer, broader topics (looser pairs, denser graph):

```bash
python3 -m wag_core \
  --input my_data/reddit_posts.tsv \
  --output-dir ./results_broad \
  --min-pair-user-pct 0.05 \
  --radius 5
```

### With a custom exclude-words file

```bash
python3 -m wag_core \
  --input my_data/forum_posts.tsv \
  --output-dir ./output \
  --exclude-words my_data/exclude_words.txt
```

### Disable iterative pruning

```bash
python3 -m wag_core \
  --input my_data/chat_logs.tsv \
  --output-dir ./output \
  --max-adjacent-topics 0
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to tab-separated input file |
| `--output-dir` | *(required)* | Directory for all output files |
| `--min-user-pct` | `1.0` | Min % of unique users for a word to be an anchor word |
| `--min-pair-user-pct` | `0.1` | Min % of users for a word pair to become a graph edge |
| `--radius` | `3` | Co-occurrence window size in tokens |
| `--stopword-sensitivity` | `0.6` | Stopword aggressiveness: 0.0 = permissive, 1.0 = aggressive |
| `--resolution` | `1.0` | Leiden clustering resolution (higher = more clusters) |
| `--weight-by` | `users` | Edge weight method: `users` (distinct users) or `frequency` (raw count) |
| `--max-adjacent-topics` | `3` | Max clusters a word can bridge before pruning. Set to 0 to disable |
| `--exclude-words` | *(none)* | Path to file with words to exclude, one per line |

## Output Files

| File | Description |
|------|-------------|
| `summary_table.tsv` | Master summary: one row per topic with anchor words, counts, top n-grams, top post |
| `all_posts_classified.tsv` | Every post with topic assignment, match score, confidence |
| `clusters.txt` | Word-to-cluster assignments |
| `node_list.tsv` | Graph nodes for Gephi (word, cluster, mentions, users) |
| `edge_list.tsv` | Graph edges for Gephi (source, target, weight) |
| `run_stats.txt` | Run summary: input size, parameters, topic count, classification breakdown |
| `stopwords_detected.txt` | Auto-detected stopwords for this corpus |
| `overconnected_words.txt` | Words sorted by cross-cluster connectivity |
| `exclude_words.txt` | Final exclude word set |
| `wag_core.log` | Detailed run log |
| `post_lists/posts_in_topic_NNN.tsv` | Strong posts per topic, sorted by score |
| `post_lists/posts_in_topic_weak.tsv` | Weakly classified posts |
| `post_lists/posts_in_topic_none.tsv` | Unclassified posts |

## Classification

Each post is scored against each topic cluster:

- **Strong** — score meets the minimum threshold for that cluster (`2 + sqrt(cluster_size) - 1`)
- **Weak** — post contains anchor words but doesn't meet the Strong threshold
- **None** — post contains no anchor words at all

## Module Structure

```
wag_core/
    __init__.py       # Package metadata, public API
    __main__.py       # CLI entry point
    exceptions.py     # Error hierarchy
    tokenizer.py      # Ingestion, tokenization, stopword detection
    graph.py          # Anchor words, co-occurrence, Leiden, connectivity
    classifier.py     # Post scoring and classification
    ngrams.py         # Per-cluster n-gram analysis
    output.py         # All output file writers
    pipeline.py       # Orchestrator with iterative pruning loop
```

## Programmatic Use

```python
from wag_core import WagPipeline

pipeline = WagPipeline(
    input_path='my_data/posts.tsv',
    output_dir='./output',
    min_pair_user_pct=0.2,
    max_adjacent_topics=3,
)
result = pipeline.run()

print(f"Found {result['topic_count']} topics")
print(f"Strong: {result['strong_count']}, Weak: {result['weak_count']}, None: {result['none_count']}")
```

## License

Copyright 2026 Chattersome Labs

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
