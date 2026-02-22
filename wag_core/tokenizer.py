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

import numpy as np

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


def detect_stopwords(word_doc_freq, total_posts, sensitivity=0.5):
    """Detect stopwords dynamically using elbow method on document frequency.

    Computes the document frequency (fraction of posts containing each word),
    ranks words by DF descending, finds the knee in the curve, and uses
    the sensitivity parameter to shift the cutoff.

    Args:
        word_doc_freq: dict of word -> number of posts containing it
        total_posts: total number of posts
        sensitivity: float 0.0-1.0
            0.0 = very permissive (few stopwords)
            0.5 = moderate (cut at natural elbow)
            1.0 = aggressive (many stopwords)

    Returns:
        set of stopwords
    """
    if not word_doc_freq or total_posts == 0:
        return set()

    # compute document frequency as fraction of total posts
    words_and_df = []
    for word, count in word_doc_freq.items():
        words_and_df.append((word, count / total_posts))

    # sort by DF descending
    words_and_df.sort(key=lambda x: x[1], reverse=True)

    if len(words_and_df) < 3:
        return set()

    words = [w for w, _ in words_and_df]
    df_values = np.array([df for _, df in words_and_df])

    # find the knee using second derivative on log-scaled values
    # log scale handles the zipfian distribution better
    log_df = np.log1p(df_values * 1000)  # scale up before log for better resolution
    ranks = np.arange(len(log_df))

    # compute second derivative (curvature)
    if len(log_df) < 5:
        # too few words for meaningful elbow detection
        return set()

    # smooth with a small window to reduce noise
    window = max(3, len(log_df) // 50)
    if window % 2 == 0:
        window += 1
    if window >= len(log_df):
        window = max(3, len(log_df) // 3)
        if window % 2 == 0:
            window += 1

    # simple moving average smoothing
    kernel = np.ones(window) / window
    if len(log_df) > window:
        smoothed = np.convolve(log_df, kernel, mode='valid')
        offset = window // 2
    else:
        smoothed = log_df
        offset = 0

    # first derivative (rate of change)
    first_deriv = np.diff(smoothed)

    # second derivative (acceleration of change)
    second_deriv = np.diff(first_deriv)

    if len(second_deriv) == 0:
        return set()

    # the knee is where the second derivative is maximized
    # (greatest change in slope = sharpest bend in curve)
    knee_idx = np.argmax(np.abs(second_deriv)) + offset + 1

    # apply sensitivity to shift the cutoff
    # sensitivity 0.0 -> cut at 20% of the way toward knee (very few stopwords)
    # sensitivity 0.5 -> cut at knee point
    # sensitivity 1.0 -> cut at up to 3x knee position (moderately more stopwords)
    total_words = len(words)
    if sensitivity <= 0.5:
        # interpolate between a minimal cutoff and the knee
        min_cutoff = max(1, int(knee_idx * 0.2))
        adjusted_idx = int(min_cutoff + (knee_idx - min_cutoff) * (sensitivity / 0.5))
    else:
        # extend past knee proportionally (up to 2x further, capped at total)
        # at sensitivity 1.0, go to 3x knee_idx (so 2x further past the knee)
        extra = int(knee_idx * 2.0 * ((sensitivity - 0.5) / 0.5))
        adjusted_idx = min(knee_idx + extra, total_words - 1)

    adjusted_idx = max(0, min(adjusted_idx, total_words - 1))

    # everything above the cutoff DF is a stopword
    if adjusted_idx <= 0:
        return set()

    cutoff_df = df_values[adjusted_idx]
    stopwords = set()
    for word, df in words_and_df:
        if df >= cutoff_df:
            stopwords.add(word)
        else:
            break  # sorted descending, so we can stop early

    logger.info("Stopword detection: knee at rank %d (DF=%.4f), "
                "sensitivity=%.2f -> cutoff at rank %d (DF=%.4f), "
                "%d stopwords detected out of %d unique tokens",
                knee_idx, df_values[min(knee_idx, len(df_values) - 1)],
                sensitivity, adjusted_idx, cutoff_df,
                len(stopwords), total_words)

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
