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

"""N-gram analysis per cluster."""

import logging
from collections import defaultdict

logger = logging.getLogger('wag_core')


def compute_ngrams(corpus, classifications, clusters, stopwords):
    """Compute top unigrams, bigrams, trigrams per cluster.

    Uses tokens_full (not tokens_filtered) for bigrams and trigrams.
    Uses tokens_full minus stopwords for unigrams.

    Scoring: group_proportion = 100 * posts_with_gram / total_posts_in_cluster

    Returns:
        dict of cluster_id -> {
            'unigrams': [(gram, score), ...],
            'bigrams': [(gram, score), ...],
            'trigrams': [(gram, score), ...],
        }
        Each list sorted by score descending, top 21 entries.
    """
    # group posts by cluster
    cluster_posts = defaultdict(list)  # cluster_id -> list of post indices
    for pc in classifications:
        if isinstance(pc.cluster_id, int):
            cluster_posts[pc.cluster_id].append(pc.post_index)

    cluster_ngrams = {}

    for cluster_id in sorted(clusters.keys()):
        post_indices = cluster_posts.get(cluster_id, [])
        total_in_cluster = len(post_indices)

        if total_in_cluster == 0:
            cluster_ngrams[cluster_id] = {
                'unigrams': [],
                'bigrams': [],
                'trigrams': [],
            }
            continue

        # track which posts contain each gram
        unigram_posts = defaultdict(set)
        bigram_posts = defaultdict(set)
        trigram_posts = defaultdict(set)

        for pidx in post_indices:
            tokens = corpus.posts[pidx]['tokens_full']

            seen_uni = set()
            seen_bi = set()
            seen_tri = set()

            for j in range(len(tokens)):
                # unigrams: skip stopwords
                token = tokens[j]
                if token not in stopwords and token not in seen_uni:
                    unigram_posts[token].add(pidx)
                    seen_uni.add(token)

                # bigrams
                if j + 1 < len(tokens):
                    bigram = tokens[j] + ' ' + tokens[j + 1]
                    if bigram not in seen_bi:
                        bigram_posts[bigram].add(pidx)
                        seen_bi.add(bigram)

                # trigrams
                if j + 2 < len(tokens):
                    trigram = tokens[j] + ' ' + tokens[j + 1] + ' ' + tokens[j + 2]
                    if trigram not in seen_tri:
                        trigram_posts[trigram].add(pidx)
                        seen_tri.add(trigram)

        # score by group proportion and take top 21
        def score_and_rank(gram_posts):
            scored = []
            for gram, posts_set in gram_posts.items():
                proportion = 100.0 * len(posts_set) / total_in_cluster
                scored.append((gram, proportion))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:21]

        cluster_ngrams[cluster_id] = {
            'unigrams': score_and_rank(unigram_posts),
            'bigrams': score_and_rank(bigram_posts),
            'trigrams': score_and_rank(trigram_posts),
        }

    logger.info("N-gram analysis complete for %d clusters", len(cluster_ngrams))

    return cluster_ngrams
