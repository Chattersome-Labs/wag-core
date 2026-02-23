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

"""Output file generation for all wag_core results."""

import csv
import logging
import math
from pathlib import Path

from .exceptions import OutputError

logger = logging.getLogger('wag_core')


def ensure_output_dirs(output_dir):
    """Create output directory and post_lists subdirectory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'post_lists').mkdir(exist_ok=True)
    return output_dir


def write_stopwords_detected(output_dir, stopwords):
    """Write auto-detected stopwords, one per line, sorted alphabetically."""
    path = Path(output_dir) / 'stopwords_detected.txt'
    with open(path, 'w', encoding='utf-8') as f:
        for word in sorted(stopwords):
            f.write(word + '\n')
    logger.info("Wrote %d stopwords to %s", len(stopwords), path)


def write_exclude_words(output_dir, exclude_words):
    """Write the current exclude word set, one per line."""
    path = Path(output_dir) / 'exclude_words.txt'
    with open(path, 'w', encoding='utf-8') as f:
        for word in sorted(exclude_words):
            f.write(word + '\n')
    logger.info("Wrote %d exclude words to %s", len(exclude_words), path)


def write_overconnected_words(output_dir, connectivity):
    """Write words sorted by cross-cluster connectivity (descending)."""
    path = Path(output_dir) / 'overconnected_words.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('mentions\tusers\tdegree\tadjacent_clusters\tword\n')
        for entry in connectivity:
            f.write('%d\t%d\t%d\t%d\t%s\n' % (
                entry['mentions'], entry['users'], entry['degree'],
                entry['adjacent_clusters'], entry['word']))
    logger.info("Wrote %d entries to %s", len(connectivity), path)


def write_clusters(output_dir, membership, index_to_word):
    """Write word-to-cluster assignments."""
    path = Path(output_dir) / 'clusters.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('word_index\tcluster_id\tword\n')
        for idx, cluster_id in enumerate(membership):
            word = index_to_word.get(idx, '')
            f.write('%d\t%d\t%s\n' % (idx, cluster_id, word))
    logger.info("Wrote %d cluster assignments to %s", len(membership), path)


def write_node_list(output_dir, word_to_index, word_to_cluster, corpus, connectivity):
    """Write node metadata TSV for graph visualization (e.g. Gephi)."""
    path = Path(output_dir) / 'node_list.tsv'

    # build connectivity lookup
    conn_lookup = {}
    for entry in connectivity:
        conn_lookup[entry['word']] = entry

    with open(path, 'w', encoding='utf-8') as f:
        f.write('Id\tLabel\tmod_class\tmentions\tusers\tmentions_per_user\n')
        for word in sorted(word_to_index.keys()):
            idx = word_to_index[word]
            cluster_id = word_to_cluster.get(word, -1)
            mentions = corpus.word_mentions.get(word, 0)
            users = len(corpus.word_users.get(word, set()))
            mpu = mentions / users if users > 0 else 0.0
            f.write('%d\t%s\t%d\t%d\t%d\t%.2f\n' % (
                idx, word, cluster_id, mentions, users, mpu))

    logger.info("Wrote node list to %s", path)


def write_edge_list(output_dir, pair_freq, pair_users, word_to_index, word_to_cluster):
    """Write edge metadata TSV for graph visualization."""
    path = Path(output_dir) / 'edge_list.tsv'

    with open(path, 'w', encoding='utf-8') as f:
        f.write('Source\tTarget\tmod_class_1\tmod_class_2\t'
                'same_mod_class\tmentions\tusers\tmentions_per_user\n')

        for pair in sorted(pair_freq.keys()):
            w1, w2 = pair
            idx1 = word_to_index[w1]
            idx2 = word_to_index[w2]
            c1 = word_to_cluster.get(w1, -1)
            c2 = word_to_cluster.get(w2, -1)
            same = 1 if c1 == c2 else 0
            mentions = pair_freq[pair]
            users = len(pair_users[pair])
            mpu = mentions / users if users > 0 else 0.0

            if same:
                mc_label = str(c1)
            else:
                mc_label = '%d,%d' % (min(c1, c2), max(c1, c2))

            f.write('%d\t%d\t%s\t%s\t%d\t%d\t%d\t%.2f\n' % (
                idx1, idx2, str(c1), str(c2), same, mentions, users, mpu))

    logger.info("Wrote edge list to %s", path)


def write_summary_table(output_dir, clusters, class_stats, cluster_ngrams,
                        classifications, corpus):
    """Write master summary table TSV with per-topic statistics."""
    path = Path(output_dir) / 'summary_table.tsv'
    per_cluster = class_stats['per_cluster']

    with open(path, 'w', encoding='utf-8') as f:
        # header
        f.write('mod_class\tanchor_word_count\tmin_match_score\t'
                'post_count\tuser_count\tposts_per_user\t'
                'high_conf\tlow_conf\tpct_high_conf\t'
                'top_score\ttop_post_length\ttop_score_per_100_char\t'
                'top_unigram_score\ttop_bigram_score\ttop_trigram_score\t'
                'anchor_words\ttop_5_unigrams\ttop_3_bigrams\ttop_3_trigrams\t'
                'top_post_text\n')

        for cid in sorted(clusters.keys()):
            c = clusters[cid]
            s = per_cluster.get(cid, {})
            ng = cluster_ngrams.get(cid, {'unigrams': [], 'bigrams': [], 'trigrams': []})

            # top gram scores
            top_uni_score = ng['unigrams'][0][1] if ng['unigrams'] else 0.0
            top_bi_score = ng['bigrams'][0][1] if ng['bigrams'] else 0.0
            top_tri_score = ng['trigrams'][0][1] if ng['trigrams'] else 0.0

            # format anchor words and n-grams
            anchor_str = ' '.join(c['words'])
            uni_str = ', '.join(g for g, _ in ng['unigrams'][:5])
            bi_str = ', '.join(g for g, _ in ng['bigrams'][:3])
            tri_str = ', '.join(g for g, _ in ng['trigrams'][:3])

            top_text = s.get('top_post_text', '').replace('\t', ' ').replace('\n', ' ')

            f.write('%03d\t%d\t%d\t%d\t%d\t%.2f\t%d\t%d\t%.1f\t'
                    '%d\t%d\t%.2f\t%.1f\t%.1f\t%.1f\t'
                    '%s\t%s\t%s\t%s\t%s\n' % (
                        cid,
                        c['word_count'],
                        c['min_score'],
                        s.get('post_count', 0),
                        s.get('user_count', 0),
                        s.get('posts_per_user', 0.0),
                        s.get('high_conf', 0),
                        s.get('low_conf', 0),
                        s.get('pct_high_conf', 0.0),
                        s.get('top_score', 0),
                        s.get('top_post_length', 0),
                        s.get('top_score_per_100_char', 0.0),
                        top_uni_score,
                        top_bi_score,
                        top_tri_score,
                        anchor_str,
                        uni_str,
                        bi_str,
                        tri_str,
                        top_text))

        # summary rows
        total = class_stats['total_posts']
        if total > 0:
            strong_pct = 100.0 * class_stats['strong_count'] / total
            weak_pct = 100.0 * class_stats['weak_count'] / total
            none_pct = 100.0 * class_stats['none_count'] / total
        else:
            strong_pct = weak_pct = none_pct = 0.0

        f.write('\n')
        f.write('strong\t\t\t%d\t%d\t\t\t\t\t\t\t\t\t\t\t%.1f%% of posts\n' % (
            class_stats['strong_count'], class_stats['strong_users'], strong_pct))
        f.write('weak\t\t\t%d\t%d\t\t\t\t\t\t\t\t\t\t\t%.1f%% of posts\n' % (
            class_stats['weak_count'], class_stats['weak_users'], weak_pct))
        f.write('none\t\t\t%d\t%d\t\t\t\t\t\t\t\t\t\t\t%.1f%% of posts\n' % (
            class_stats['none_count'], class_stats['none_users'], none_pct))

    logger.info("Wrote summary table to %s", path)


def write_all_posts_classified(output_dir, classifications):
    """Write every post with its classification."""
    path = Path(output_dir) / 'all_posts_classified.tsv'

    with open(path, 'w', encoding='utf-8') as f:
        f.write('row_id\tmod_class\tconfidence\ttop_groups\t'
                'match_score\tpost_length\tscore_per_100_char\t'
                'user_id\tpost_text\n')

        for pc in classifications:
            if isinstance(pc.cluster_id, int):
                mc = '%03d' % pc.cluster_id
            else:
                mc = str(pc.cluster_id)

            text = pc.raw_text.replace('\t', ' ').replace('\n', ' ')
            f.write('%d\t%s\t%s\t%s\t%d\t%d\t%.2f\t%s\t%s\n' % (
                pc.post_index, mc, pc.confidence, pc.top_groups,
                pc.match_score, pc.post_length, pc.score_per_100_char,
                pc.user_id, text))

    logger.info("Wrote %d classified posts to %s", len(classifications), path)


def write_per_topic_post_lists(output_dir, classifications, clusters):
    """Write per-topic post list files.

    Strong topics sorted by score_per_100_char descending.
    Weak/none sorted by post_length ascending.
    """
    output_dir = Path(output_dir)
    post_lists_dir = output_dir / 'post_lists'

    # group by cluster
    by_cluster = {}
    weak_posts = []
    none_posts = []

    for pc in classifications:
        if isinstance(pc.cluster_id, int):
            by_cluster.setdefault(pc.cluster_id, []).append(pc)
        elif pc.cluster_id == 'weak':
            weak_posts.append(pc)
        else:
            none_posts.append(pc)

    def write_post_list(filepath, posts):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('mod_class\tmatch_score\tconfidence\ttop_groups\t'
                    'post_length\tscore_per_100_char\tuser_id\tpost_text\n')
            for pc in posts:
                if isinstance(pc.cluster_id, int):
                    mc = '%03d' % pc.cluster_id
                else:
                    mc = str(pc.cluster_id)
                text = pc.raw_text.replace('\t', ' ').replace('\n', ' ')
                f.write('%s\t%d\t%s\t%s\t%d\t%.2f\t%s\t%s\n' % (
                    mc, pc.match_score, pc.confidence, pc.top_groups,
                    pc.post_length, pc.score_per_100_char, pc.user_id, text))

    # write each cluster's posts
    for cid in sorted(clusters.keys()):
        posts = by_cluster.get(cid, [])
        posts.sort(key=lambda p: p.score_per_100_char, reverse=True)
        filepath = post_lists_dir / ('posts_in_topic_%03d.tsv' % cid)
        write_post_list(filepath, posts)

    # weak and none
    weak_posts.sort(key=lambda p: p.post_length)
    write_post_list(post_lists_dir / 'posts_in_topic_weak.tsv', weak_posts)

    none_posts.sort(key=lambda p: p.post_length)
    write_post_list(post_lists_dir / 'posts_in_topic_none.tsv', none_posts)

    logger.info("Wrote %d per-topic post list files to %s",
                len(clusters) + 2, post_lists_dir)


def write_run_stats(output_dir, corpus, clusters, class_stats,
                    params, iterations=None):
    """Write session summary with input/output stats and parameters."""
    path = Path(output_dir) / 'run_stats.txt'
    total = class_stats['total_posts']

    with open(path, 'w', encoding='utf-8') as f:
        f.write('=== WAG Core Run Statistics ===\n\n')

        f.write('Input:\n')
        f.write('  Posts: %d\n' % corpus.total_posts)
        f.write('  Unique users: %d\n' % corpus.total_users)
        f.write('  Input file: %s\n' % params.get('input_path', ''))
        f.write('\n')

        f.write('Parameters:\n')
        f.write('  min_user_pct: %.2f%%\n' % params.get('min_user_pct', 1.0))
        f.write('  radius: %d\n' % params.get('radius', 5))
        f.write('  stopword_sensitivity: %.2f\n' % params.get('stopword_sensitivity', 0.5))
        f.write('  resolution: %.2f\n' % params.get('resolution', 1.0))
        f.write('  weight_by: %s\n' % params.get('weight_by', 'frequency'))
        if params.get('max_adjacent_topics') is not None:
            f.write('  max_adjacent_topics: %d\n' % params['max_adjacent_topics'])
        f.write('\n')

        f.write('Output:\n')
        f.write('  Topics found: %d\n' % len(clusters))
        f.write('  Stopwords detected: %d\n' % len(corpus.stopwords))
        f.write('\n')

        f.write('Classification:\n')
        if total > 0:
            f.write('  Strong: %d (%.1f%%)\n' % (
                class_stats['strong_count'],
                100.0 * class_stats['strong_count'] / total))
            f.write('  Weak:   %d (%.1f%%)\n' % (
                class_stats['weak_count'],
                100.0 * class_stats['weak_count'] / total))
            f.write('  None:   %d (%.1f%%)\n' % (
                class_stats['none_count'],
                100.0 * class_stats['none_count'] / total))
        f.write('\n')

        if iterations is not None:
            f.write('Iterative Pruning:\n')
            f.write('  Iterations: %d\n' % iterations['count'])
            f.write('  Final max connectivity: %d\n' % iterations['final_max_conn'])
            f.write('  Total words excluded: %d\n' % iterations['total_excluded'])
            f.write('\n')

    logger.info("Wrote run stats to %s", path)


def write_all_outputs(output_dir, corpus, clusters, word_to_cluster,
                      word_to_index, index_to_word, membership,
                      pair_freq, pair_users, connectivity,
                      classifications, class_stats, cluster_ngrams,
                      exclude_words, params, iterations=None):
    """Write all output files."""
    output_dir = ensure_output_dirs(output_dir)

    write_stopwords_detected(output_dir, corpus.stopwords)
    write_exclude_words(output_dir, exclude_words)
    write_overconnected_words(output_dir, connectivity)
    write_clusters(output_dir, membership, index_to_word)
    write_node_list(output_dir, word_to_index, word_to_cluster, corpus, connectivity)
    write_edge_list(output_dir, pair_freq, pair_users, word_to_index, word_to_cluster)
    write_summary_table(output_dir, clusters, class_stats, cluster_ngrams,
                        classifications, corpus)
    write_all_posts_classified(output_dir, classifications)
    write_per_topic_post_lists(output_dir, classifications, clusters)
    write_run_stats(output_dir, corpus, clusters, class_stats, params, iterations)

    logger.info("All output files written to %s", output_dir)
