# Copyright 2026 Chattersome Labs
# https://github.com/Chattersome-Labs/wag-core
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text ingestion, tokenization, and dynamic stopword detection."""

import logging
import re
from collections import defaultdict
from pathlib import Path

from .exceptions import InputError, TokenizationError

logger = logging.getLogger('wag_core')


class TokenizedCorpus:
    """Holds all ingested post data and word statistics."""

    def __init__(self):
        self.posts = []             # list of dicts: {user_id, raw_text, tokens_full, tokens_filtered}
        self.unique_users = set()
        self.word_mentions = defaultdict(int)    # word -> total mention count
        self.word_users = defaultdict(set)       # word -> set of user_ids
        self.word_doc_freq = defaultdict(int)    # word -> number of posts containing it
        self.total_posts = 0
        self.total_users = 0
        self.stopwords = set()


def tokenize_text(text):
    """Tokenize a single post into a list of cleaned tokens.

    Preserves # and @ prefixes. Removes URLs. Lowercases everything.
    Returns the full token list (before stopword filtering).
    """
    tokens = []
    for raw_token in text.split():
        # skip URLs
        if raw_token.lower().startswith('http'):
            continue

        if raw_token.startswith('#') or raw_token.startswith('@'):
            # preserve prefix, strip trailing non-word chars
            cleaned = re.sub(r'[^\w#@]+$', '', raw_token).lower()
        else:
            # strip leading and trailing non-word chars
            cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', raw_token).lower()

        if cleaned:
            tokens.append(cleaned)

    return tokens


def detect_stopwords(word_doc_freq, total_posts, sensitivity=0.6):
    """Detect stopwords using a document-frequency percentage threshold.

    A word is a stopword if it appears in more than a certain percentage
    of all posts. The sensitivity parameter controls that threshold:

        threshold = 50% - (sensitivity * 40%)

    So:
        sensitivity 0.0 -> only words in >50% of posts (just core function words)
        sensitivity 0.6 -> only words in >26% of posts (default, conservative)
        sensitivity 1.0 -> only words in >10% of posts (more aggressive)

    This is simple, predictable, and avoids flagging high-information
    domain words that happen to be common in a topic-focused corpus.

    Args:
        word_doc_freq: dict of word -> number of posts containing it
        total_posts: total number of posts
        sensitivity: float 0.0-1.0 controlling the DF threshold

    Returns:
        set of stopwords
    """
    if not word_doc_freq or total_posts == 0:
        return set()

    # compute the DF threshold as a fraction
    # sensitivity 0.0 -> 0.50, sensitivity 0.6 -> 0.26, sensitivity 1.0 -> 0.10
    threshold = 0.50 - (sensitivity * 0.40)
    threshold = max(0.05, min(0.50, threshold))  # clamp to 5%-50%

    stopwords = set()
    for word, count in word_doc_freq.items():
        df = count / total_posts
        if df >= threshold:
            stopwords.add(word)

    logger.info("Stopword detection: sensitivity=%.2f -> DF threshold=%.1f%%, "
                "%d stopwords detected out of %d unique tokens",
                sensitivity, threshold * 100,
                len(stopwords), len(word_doc_freq))

    return stopwords


def load_exclude_words(exclude_path):
    """Load exclude words from a file, one word per line.

    Returns empty set if path is None or file doesn't exist.
    """
    if exclude_path is None:
        return set()

    path = Path(exclude_path)
    if not path.exists():
        logger.warning("Exclude words file not found: %s", path)
        return set()

    words = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word and not word.startswith('#'):
                words.add(word)

    logger.info("Loaded %d exclude words from %s", len(words), path)
    return words


def ingest_corpus(input_path, stopword_sensitivity=0.5, exclude_words=None):
    """Read input file, tokenize all posts, detect stopwords, build corpus.

    Args:
        input_path: Path to tab-separated file (user_id<TAB>post_text)
        stopword_sensitivity: float 0.0-1.0 for dynamic stopword detection
        exclude_words: set of words to exclude (or None)

    Returns:
        TokenizedCorpus instance with all statistics populated
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise InputError("Input file not found: %s" % input_path)

    if exclude_words is None:
        exclude_words = set()

    corpus = TokenizedCorpus()

    # first pass: read all posts, tokenize, build word stats
    logger.info("Reading input file: %s", input_path)
    line_num = 0
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if not line:
                continue

            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue

            user_id = parts[0].strip()
            raw_text = parts[1].strip()
            if not raw_text:
                continue

            tokens_full = tokenize_text(raw_text)
            # remove exclude words from full list too
            tokens_full = [t for t in tokens_full if t not in exclude_words]

            post = {
                'user_id': user_id,
                'raw_text': raw_text,
                'tokens_full': tokens_full,
                'tokens_filtered': [],  # populated after stopword detection
            }
            corpus.posts.append(post)
            corpus.unique_users.add(user_id)

            # track word stats (pre-stopword)
            seen_in_post = set()
            for token in tokens_full:
                corpus.word_mentions[token] += 1
                corpus.word_users[token].add(user_id)
                if token not in seen_in_post:
                    corpus.word_doc_freq[token] += 1
                    seen_in_post.add(token)

            line_num += 1
            if line_num % 5000 == 0:
                logger.info("  read %d posts...", line_num)

    corpus.total_posts = len(corpus.posts)
    corpus.total_users = len(corpus.unique_users)

    if corpus.total_posts == 0:
        raise InputError("No valid posts found in %s" % input_path)

    logger.info("Ingested %d posts from %d unique users, %d unique tokens",
                corpus.total_posts, corpus.total_users, len(corpus.word_mentions))

    # detect stopwords dynamically
    corpus.stopwords = detect_stopwords(
        corpus.word_doc_freq, corpus.total_posts, stopword_sensitivity
    )

    # second pass: build filtered token lists (excluding stopwords)
    for post in corpus.posts:
        post['tokens_filtered'] = [
            t for t in post['tokens_full'] if t not in corpus.stopwords
        ]

    filtered_token_count = sum(len(p['tokens_filtered']) for p in corpus.posts)
    logger.info("After stopword removal: %d stopwords filtered, "
                "%d tokens remain across all posts",
                len(corpus.stopwords), filtered_token_count)

    return corpus
