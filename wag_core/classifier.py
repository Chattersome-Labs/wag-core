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

"""Post classification against topic clusters."""

import logging
from collections import defaultdict

from .exceptions import ClassificationError

logger = logging.getLogger('wag_core')


class PostClassification:
    """Classification result for a single post."""

    def __init__(self):
        self.post_index = 0
        self.cluster_id = None      # int, 'weak', or 'none'
        self.match_score = 0
        self.confidence = 'high'    # 'high' or 'low'
        self.top_groups = ''        # space-separated cluster ids for ties
        self.post_length = 0
        self.score_per_100_char = 0.0
        self.user_id = ''
        self.raw_text = ''


def compute_pair_cluster_map(pair_freq, word_to_cluster):
    """Map each pair to its cluster if both words share the same cluster.

    Returns:
        dict of (word_a, word_b) -> cluster_id, only for intra-cluster pairs
    """
    pair_cluster = {}
    for pair in pair_freq:
        w1, w2 = pair
        c1 = word_to_cluster.get(w1)
        c2 = word_to_cluster.get(w2)
        if c1 is not None and c2 is not None and c1 == c2:
            pair_cluster[pair] = c1
    return pair_cluster


def classify_posts(corpus, clusters, word_to_cluster, pair_freq, post_pairs,
                    anchor_words=None):
    """Classify every post against topic clusters.

    Scoring: +1 per anchor word match, +1 per intra-cluster pair match.
    Ties broken by highest cluster ID.

    Classification:
        Strong: score >= min_score for winning cluster (assigned to that cluster)
        Weak: post contains anchor words but score is below threshold, or
              post contains anchor words not assigned to any cluster
        None: post contains NO anchor words at all

    Args:
        anchor_words: set of ALL anchor words (including those not in graph).
            Used to distinguish Weak from None for posts with orphaned anchors.

    Returns:
        list of PostClassification objects, parallel to corpus.posts
    """
    pair_cluster = compute_pair_cluster_map(pair_freq, word_to_cluster)
    if anchor_words is None:
        anchor_words = set(word_to_cluster.keys())
    classifications = []

    for i, post in enumerate(corpus.posts):
        pc = PostClassification()
        pc.post_index = i
        pc.user_id = post['user_id']
        pc.raw_text = post['raw_text']
        pc.post_length = len(post['raw_text'])

        # tally scores per cluster
        tallies = defaultdict(int)

        # word matches (only words assigned to a cluster)
        for token in post['tokens_filtered']:
            cluster_id = word_to_cluster.get(token)
            if cluster_id is not None:
                tallies[cluster_id] += 1

        # pair matches (intra-cluster only)
        for pair in post_pairs[i]:
            cluster_id = pair_cluster.get(pair)
            if cluster_id is not None:
                tallies[cluster_id] += 1

        if not tallies:
            # no cluster-assigned words matched; check if ANY anchor words
            # are present (including orphaned ones not in the graph)
            has_any_anchor = any(
                t in anchor_words for t in post['tokens_filtered']
            )
            if has_any_anchor:
                pc.cluster_id = 'weak'
                pc.match_score = 0
            else:
                pc.cluster_id = 'none'
                pc.match_score = 0
            pc.confidence = 'high'
            pc.top_groups = ''
        else:
            # find max score
            max_score = max(tallies.values())
            top_clusters = [cid for cid, score in tallies.items()
                            if score == max_score]
            top_clusters.sort(reverse=True)  # highest cluster ID first

            winner = top_clusters[0]
            pc.match_score = max_score
            pc.top_groups = ' '.join(str(c) for c in sorted(top_clusters))
            pc.confidence = 'high' if len(top_clusters) == 1 else 'low'

            # check against min_score threshold
            min_score = clusters.get(winner, {}).get('min_score', 1)
            if max_score >= min_score:
                pc.cluster_id = winner
            else:
                pc.cluster_id = 'weak'

        # score density
        if pc.post_length > 0:
            pc.score_per_100_char = 100.0 * pc.match_score / pc.post_length

        classifications.append(pc)

    # log summary
    strong = sum(1 for c in classifications if isinstance(c.cluster_id, int))
    weak = sum(1 for c in classifications if c.cluster_id == 'weak')
    none_ = sum(1 for c in classifications if c.cluster_id == 'none')
    logger.info("Classification: %d Strong, %d Weak, %d None (of %d posts)",
                strong, weak, none_, len(classifications))

    return classifications


def compute_class_stats(classifications, clusters, corpus):
    """Compute per-cluster and overall classification statistics.

    Returns:
        dict with:
            'per_cluster': {cluster_id: {post_count, user_count, ...}}
            'strong_count', 'weak_count', 'none_count'
            'strong_users', 'weak_users', 'none_users'
            'total_posts', 'total_users'
    """
    per_cluster = {}

    # initialize for each cluster
    for cid in clusters:
        per_cluster[cid] = {
            'post_count': 0,
            'user_count': 0,
            'users': set(),
            'high_conf': 0,
            'low_conf': 0,
            'top_score': 0,
            'top_post_length': 0,
            'top_score_per_100_char': 0.0,
            'top_post_text': '',
            'top_post_index': -1,
        }

    strong_users = set()
    weak_users = set()
    none_users = set()
    weak_count = 0
    none_count = 0

    for pc in classifications:
        if isinstance(pc.cluster_id, int):
            stats = per_cluster.get(pc.cluster_id)
            if stats is None:
                continue
            stats['post_count'] += 1
            stats['users'].add(pc.user_id)
            strong_users.add(pc.user_id)
            if pc.confidence == 'high':
                stats['high_conf'] += 1
            else:
                stats['low_conf'] += 1
            if pc.score_per_100_char > stats['top_score_per_100_char']:
                stats['top_score'] = pc.match_score
                stats['top_post_length'] = pc.post_length
                stats['top_score_per_100_char'] = pc.score_per_100_char
                stats['top_post_text'] = pc.raw_text
                stats['top_post_index'] = pc.post_index
        elif pc.cluster_id == 'weak':
            weak_count += 1
            weak_users.add(pc.user_id)
        else:
            none_count += 1
            none_users.add(pc.user_id)

    # finalize user counts
    for cid, stats in per_cluster.items():
        stats['user_count'] = len(stats['users'])
        if stats['user_count'] > 0:
            stats['posts_per_user'] = stats['post_count'] / stats['user_count']
        else:
            stats['posts_per_user'] = 0.0
        total = stats['high_conf'] + stats['low_conf']
        if total > 0:
            stats['pct_high_conf'] = 100.0 * stats['high_conf'] / total
        else:
            stats['pct_high_conf'] = 0.0

    strong_count = sum(s['post_count'] for s in per_cluster.values())

    return {
        'per_cluster': per_cluster,
        'strong_count': strong_count,
        'weak_count': weak_count,
        'none_count': none_count,
        'strong_users': len(strong_users),
        'weak_users': len(weak_users),
        'none_users': len(none_users),
        'total_posts': corpus.total_posts,
        'total_users': corpus.total_users,
    }
