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
from collections import defaultdict

import igraph as ig
import leidenalg as la

from .exceptions import GraphError

logger = logging.getLogger('wag_core')


def select_anchor_words(corpus, min_users):
    """Select anchor words: tokens used by >= min_users unique users.

    Args:
        corpus: TokenizedCorpus instance
        min_users: int, absolute minimum user count

    Returns:
        set of anchor word strings
    """
    anchors = set()
    for word, users in corpus.word_users.items():
        if word in corpus.stopwords:
            continue
        if len(users) >= min_users:
            anchors.add(word)

    logger.info("Anchor word selection: min_users=%d -> "
                "%d anchor words selected",
                min_users, len(anchors))

    if len(anchors) < 2:
        raise GraphError(
            "Only %d anchor words found (need at least 2). "
            "Try raising --detail." % len(anchors)
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
        clusters[cluster_id] = {
            'words': words,
            'word_count': word_count,
        }

    logger.info("Cluster info: %d clusters, sizes: %s",
                len(clusters),
                ', '.join('%d:%d' % (cid, c['word_count'])
                          for cid, c in sorted(clusters.items())[:10]))

    return clusters, word_to_cluster


def enforce_cluster_size_cap(graph, clusters, word_to_cluster,
                             index_to_word, word_to_index, corpus,
                             max_cluster_words, base_resolution=1.0):
    """Enforce maximum cluster size via sub-clustering.

    For each cluster exceeding max_cluster_words:
    1. Extract subgraph of that cluster's vertices
    2. Re-run Leiden at progressively higher resolution (2x, 4x, 8x, 16x)
    3. If all sub-clusters fit within the cap: adopt them
    4. If words resist splitting even at 16x: exclude them

    Returns:
        clusters: updated cluster dict (renumbered from 0)
        word_to_cluster: updated mapping
        words_to_exclude: set of words that couldn't be split below the cap
    """
    if max_cluster_words is None:
        return clusters, word_to_cluster, set()

    oversized = {cid: info for cid, info in clusters.items()
                 if info['word_count'] > max_cluster_words}

    if not oversized:
        return clusters, word_to_cluster, set()

    logger.info("Cluster size cap (%d): %d clusters exceed limit: %s",
                max_cluster_words, len(oversized),
                ', '.join('c%d(%dw)' % (cid, info['word_count'])
                          for cid, info in sorted(oversized.items())))

    words_to_exclude = set()
    new_sub_clusters = []  # list of word-lists for adopted sub-clusters

    # collect small clusters unchanged
    kept_clusters = []
    for cid in sorted(clusters.keys()):
        if cid not in oversized:
            kept_clusters.append(clusters[cid]['words'])

    for cid in sorted(oversized.keys()):
        words = oversized[cid]['words']

        # get vertex indices for this cluster's words
        vertex_ids = [word_to_index[w] for w in words if w in word_to_index]

        if len(vertex_ids) < 2:
            words_to_exclude.update(words)
            continue

        # extract subgraph
        subgraph = graph.induced_subgraph(vertex_ids)

        # try progressively higher resolutions
        split_success = False
        best_membership = None

        for res_multiplier in [2.0, 4.0, 8.0, 16.0]:
            sub_resolution = base_resolution * res_multiplier
            sub_membership, sub_num = run_leiden(subgraph, sub_resolution)

            # check if all sub-clusters fit within the cap
            sub_sizes = defaultdict(int)
            for m in sub_membership:
                sub_sizes[m] += 1
            max_sub_size = max(sub_sizes.values())

            best_membership = sub_membership

            if max_sub_size <= max_cluster_words:
                # success — all sub-clusters fit
                for sub_cid in sorted(set(sub_membership)):
                    sub_words = [subgraph.vs[i]['name']
                                 for i, m in enumerate(sub_membership)
                                 if m == sub_cid]
                    sub_words.sort(
                        key=lambda w: len(corpus.word_users.get(w, set())),
                        reverse=True)
                    new_sub_clusters.append(sub_words)

                split_success = True
                logger.info("  Cluster %d (%d words) -> %d sub-clusters "
                            "at resolution=%.1f",
                            cid, len(words), sub_num, sub_resolution)
                break

        if not split_success:
            # use the last (highest resolution) attempt
            # keep sub-clusters that fit, exclude words from ones that don't
            kept_sub = 0
            excluded_sub = 0
            for sub_cid in sorted(set(best_membership)):
                sub_words = [subgraph.vs[i]['name']
                             for i, m in enumerate(best_membership)
                             if m == sub_cid]
                if len(sub_words) <= max_cluster_words:
                    sub_words.sort(
                        key=lambda w: len(corpus.word_users.get(w, set())),
                        reverse=True)
                    new_sub_clusters.append(sub_words)
                    kept_sub += 1
                else:
                    words_to_exclude.update(sub_words)
                    excluded_sub += 1

            logger.info("  Cluster %d (%d words) partially split at 16x: "
                        "%d sub-clusters kept, %d sub-clusters excluded "
                        "(%d words)",
                        cid, len(words), kept_sub, excluded_sub,
                        len([w for w in words if w in words_to_exclude]))

    # rebuild clusters dict with new sequential IDs
    all_cluster_words = kept_clusters + new_sub_clusters

    # remove single-word orphans (can't match posts via pair scoring)
    orphan_words = set()
    filtered = []
    for wl in all_cluster_words:
        if len(wl) <= 1:
            orphan_words.update(wl)
        else:
            filtered.append(wl)
    if orphan_words:
        words_to_exclude.update(orphan_words)
        logger.info("Excluding %d single-word orphan clusters: %s",
                    len(orphan_words),
                    ', '.join(sorted(orphan_words)[:10]))
    all_cluster_words = filtered

    # sort by size descending, then first word alphabetically for stability
    all_cluster_words.sort(key=lambda wl: (-len(wl), wl[0] if wl else ''))

    new_clusters = {}
    new_word_to_cluster = {}
    for new_cid, words in enumerate(all_cluster_words):
        new_clusters[new_cid] = {
            'words': words,
            'word_count': len(words),
        }
        for w in words:
            new_word_to_cluster[w] = new_cid

    logger.info("Cluster size cap result: %d -> %d clusters, "
                "%d words to exclude",
                len(clusters), len(new_clusters), len(words_to_exclude))

    return new_clusters, new_word_to_cluster, words_to_exclude


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
