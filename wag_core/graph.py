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

"""Graph construction, Leiden clustering, and connectivity analysis."""

import logging
import math
from collections import defaultdict

import igraph as ig
import leidenalg as la

from .exceptions import GraphError

logger = logging.getLogger('wag_core')


def select_anchor_words(corpus, min_user_pct):
    """Select anchor words: tokens used by >= min_user_pct% of unique users.

    Args:
        corpus: TokenizedCorpus instance
        min_user_pct: float (e.g. 1.0 means 1% of users)

    Returns:
        set of anchor word strings
    """
    min_users_needed = math.ceil(corpus.total_users * min_user_pct / 100.0)
    if min_users_needed < 1:
        min_users_needed = 1

    anchors = set()
    for word, users in corpus.word_users.items():
        if word in corpus.stopwords:
            continue
        if len(users) >= min_users_needed:
            anchors.add(word)

    logger.info("Anchor word selection: min_user_pct=%.2f%% -> "
                "min_users=%d -> %d anchor words selected",
                min_user_pct, min_users_needed, len(anchors))

    if len(anchors) < 2:
        raise GraphError(
            "Only %d anchor words found (need at least 2). "
            "Try lowering --min-user-pct." % len(anchors)
        )

    return anchors


def build_cooccurrence_pairs(corpus, anchor_words, radius=5, min_pair_users=0):
    """Generate co-occurrence pairs within configurable radius.

    For each post's filtered token list, pair each token with tokens
    up to `radius` positions ahead. Only keep pairs where both words
    are anchor words and (if min_pair_users > 0) the pair was produced
    by at least that many distinct users.

    Args:
        corpus: TokenizedCorpus instance
        anchor_words: set of anchor word strings
        radius: int, max distance between paired tokens (default 5)
        min_pair_users: int, minimum unique users per pair to qualify
                        as a graph edge (default 0 = no filtering)

    Returns:
        pair_freq: dict of (word_a, word_b) -> total frequency count
        pair_users: dict of (word_a, word_b) -> set of user_ids
        post_pairs: list of lists, parallel to corpus.posts, each a list
                    of qualifying pairs found in that post
    """
    pair_freq = defaultdict(int)
    pair_users = defaultdict(set)
    raw_post_pairs = []  # unfiltered, for re-filtering after pair cutoff

    for post in corpus.posts:
        tokens = post['tokens_filtered']
        user_id = post['user_id']
        pairs_in_post = []

        for i in range(len(tokens)):
            if tokens[i] not in anchor_words:
                continue
            end = min(i + radius + 1, len(tokens))
            for j in range(i + 1, end):
                if tokens[j] not in anchor_words:
                    continue
                # sort pair alphabetically
                pair = tuple(sorted((tokens[i], tokens[j])))
                pair_freq[pair] += 1
                pair_users[pair].add(user_id)
                pairs_in_post.append(pair)

        raw_post_pairs.append(pairs_in_post)

    total_before = len(pair_freq)

    # apply pair-level user count filter
    if min_pair_users > 0:
        qualifying_pairs = {
            pair for pair, users in pair_users.items()
            if len(users) >= min_pair_users
        }
        pair_freq = {p: f for p, f in pair_freq.items() if p in qualifying_pairs}
        pair_users = {p: u for p, u in pair_users.items() if p in qualifying_pairs}

        # rebuild post_pairs with only qualifying pairs
        post_pairs = []
        for pairs_in_post in raw_post_pairs:
            post_pairs.append([p for p in pairs_in_post if p in qualifying_pairs])

        logger.info("Co-occurrence pairs: radius=%d, %d raw pairs -> "
                    "%d after min_pair_users=%d filter",
                    radius, total_before, len(pair_freq), min_pair_users)
    else:
        post_pairs = raw_post_pairs
        logger.info("Co-occurrence pairs: radius=%d, %d unique pairs found",
                    radius, len(pair_freq))

    return pair_freq, pair_users, post_pairs


def build_graph(pair_freq, pair_users, weight_by='frequency'):
    """Build an igraph Graph from word pairs.

    Args:
        pair_freq: dict of (word_a, word_b) -> frequency count
        pair_users: dict of (word_a, word_b) -> set of user_ids
        weight_by: 'frequency' or 'users'

    Returns:
        graph: igraph.Graph (undirected, weighted)
        word_to_index: dict mapping word -> vertex index
        index_to_word: dict mapping vertex index -> word
    """
    # collect all unique words from pairs and sort for stable indexing
    all_words = set()
    for w1, w2 in pair_freq:
        all_words.add(w1)
        all_words.add(w2)
    all_words = sorted(all_words)

    word_to_index = {w: i for i, w in enumerate(all_words)}
    index_to_word = {i: w for w, i in word_to_index.items()}

    edges = []
    weights = []
    for pair, freq in pair_freq.items():
        w1, w2 = pair
        edges.append((word_to_index[w1], word_to_index[w2]))
        if weight_by == 'users':
            weights.append(len(pair_users[pair]))
        else:
            weights.append(freq)

    graph = ig.Graph(n=len(all_words), edges=edges, directed=False)
    graph.vs['name'] = all_words
    graph.es['weight'] = weights

    logger.info("Graph built: %d vertices, %d edges", graph.vcount(), graph.ecount())

    return graph, word_to_index, index_to_word


def run_leiden(graph, resolution=1.0):
    """Run Leiden community detection on the graph.

    Args:
        graph: igraph.Graph (weighted)
        resolution: float, Leiden resolution parameter (default 1.0)

    Returns:
        membership: list of cluster IDs, one per vertex
        num_clusters: int
    """
    if graph.vcount() == 0:
        raise GraphError("Cannot cluster an empty graph")

    if resolution == 1.0:
        partition = la.find_partition(
            graph,
            la.ModularityVertexPartition,
            weights='weight',
            n_iterations=-1,
            seed=42,
        )
    else:
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            n_iterations=-1,
            seed=42,
        )

    membership = partition.membership
    num_clusters = len(set(membership))

    logger.info("Leiden clustering: resolution=%.2f, found %d communities "
                "(quality=%.4f)", resolution, num_clusters, partition.quality())

    return membership, num_clusters


def compute_cluster_info(membership, index_to_word, corpus):
    """Build cluster metadata from Leiden output.

    Returns:
        clusters: dict of cluster_id -> {
            'words': list of words sorted by user count descending,
            'word_count': int,
            'min_score': int
        }
        word_to_cluster: dict of word -> cluster_id
    """
    word_to_cluster = {}
    cluster_words = defaultdict(list)

    for idx, cluster_id in enumerate(membership):
        word = index_to_word[idx]
        word_to_cluster[word] = cluster_id
        cluster_words[cluster_id].append(word)

    clusters = {}
    for cluster_id, words in sorted(cluster_words.items()):
        # sort by unique user count descending
        words.sort(key=lambda w: len(corpus.word_users.get(w, set())), reverse=True)
        word_count = len(words)
        min_score = int(2 + math.sqrt(word_count)) - 1
        clusters[cluster_id] = {
            'words': words,
            'word_count': word_count,
            'min_score': min_score,
        }

    logger.info("Cluster info: %d clusters, sizes: %s",
                len(clusters),
                ', '.join('%d:%d' % (cid, c['word_count'])
                          for cid, c in sorted(clusters.items())[:10]))

    return clusters, word_to_cluster


def compute_connectivity(graph, word_to_cluster, word_to_index, corpus):
    """Compute cross-cluster connectivity for each word.

    For each word in the graph, count how many distinct clusters its
    graph neighbors belong to.

    Returns:
        connectivity: list of dicts sorted by adjacent_clusters descending:
            [{word, adjacent_clusters, degree, mentions, users}, ...]
    """
    connectivity = []

    for word, idx in word_to_index.items():
        neighbors = graph.neighbors(idx)
        neighbor_clusters = set()
        for n_idx in neighbors:
            n_word = graph.vs[n_idx]['name']
            if n_word in word_to_cluster:
                neighbor_clusters.add(word_to_cluster[n_word])

        connectivity.append({
            'word': word,
            'adjacent_clusters': len(neighbor_clusters),
            'degree': len(neighbors),
            'mentions': corpus.word_mentions.get(word, 0),
            'users': len(corpus.word_users.get(word, set())),
        })

    connectivity.sort(key=lambda x: x['adjacent_clusters'], reverse=True)

    if connectivity:
        logger.info("Connectivity: max adjacent clusters = %d (word: '%s')",
                    connectivity[0]['adjacent_clusters'],
                    connectivity[0]['word'])

    return connectivity


def find_overconnected_words(connectivity, max_adjacent_topics):
    """Find words exceeding the adjacent-topics threshold.

    Only returns words at the MAXIMUM connectivity level (matching
    the Perl meta_opt.pl behavior of adding one tier at a time).

    Returns:
        words_to_exclude: list of word strings
        max_connectivity: int (current maximum)
        converged: bool (True if max <= threshold)
    """
    if not connectivity:
        return [], 0, True

    max_conn = connectivity[0]['adjacent_clusters']

    if max_conn <= max_adjacent_topics:
        return [], max_conn, True

    words_to_exclude = [
        entry['word'] for entry in connectivity
        if entry['adjacent_clusters'] == max_conn
    ]

    return words_to_exclude, max_conn, False
