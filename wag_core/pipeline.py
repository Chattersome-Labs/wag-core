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

"""Pipeline orchestrator with iterative pruning loop."""

import logging
import math
from pathlib import Path

from .exceptions import WagCoreError
from .tokenizer import ingest_corpus, load_exclude_words
from .graph import (select_anchor_words, build_cooccurrence_pairs,
                    build_graph, run_leiden, compute_cluster_info,
                    enforce_cluster_size_cap,
                    compute_connectivity, find_overconnected_words)
from .classifier import classify_posts, compute_class_stats
from .ngrams import compute_ngrams
from .output import ensure_output_dirs, write_exclude_words, write_all_outputs

logger = logging.getLogger('wag_core')


class WagPipeline:
    """Main pipeline orchestrator for WAG topic detection."""

    def __init__(self, input_path, output_dir, detail=1.0,
                 min_user_pct=None, radius=1, stopword_sensitivity=0.6,
                 resolution=1.0, exclude_words_path=None,
                 max_adjacent_topics=3, max_iterations=None,
                 weight_by='users', max_cluster_words=8):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.detail = detail
        self.min_user_pct_override = min_user_pct
        self.radius = radius
        self.stopword_sensitivity = stopword_sensitivity
        self.resolution = resolution
        self.exclude_words_path = exclude_words_path
        self.max_adjacent_topics = max_adjacent_topics
        self.max_iterations = max_iterations
        self.weight_by = weight_by
        self.max_cluster_words = max_cluster_words

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to both file and stderr."""
        ensure_output_dirs(self.output_dir)

        wag_logger = logging.getLogger('wag_core')
        if not wag_logger.handlers:
            wag_logger.setLevel(logging.INFO)

            # stderr handler
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
            console.setFormatter(fmt)
            wag_logger.addHandler(console)

            # file handler
            log_path = self.output_dir / 'wag_core.log'
            fh = logging.FileHandler(log_path, encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            wag_logger.addHandler(fh)

    def _get_params(self):
        """Return a dict of current parameters for output."""
        return {
            'input_path': str(self.input_path),
            'detail': self.detail,
            'min_user_pct_override': self.min_user_pct_override,
            'min_users': getattr(self, '_effective_min_users', None),
            'min_user_pct_effective': getattr(self, '_effective_min_user_pct', None),
            'radius': self.radius,
            'stopword_sensitivity': self.stopword_sensitivity,
            'resolution': self.resolution,
            'weight_by': self.weight_by,
            'max_iterations': self.max_iterations,
            'max_adjacent_topics': self.max_adjacent_topics,
            'max_cluster_words': self.max_cluster_words,
        }

    def _compute_min_users(self, total_users):
        """Compute the absolute minimum user count for anchors and pairs.

        If min_user_pct was explicitly set, use that directly.
        Otherwise, auto-scale from corpus size and detail level:
            base = 6 + 0.42 * sqrt(total_users)
            min_users = max(2, round(base / detail))

        Higher detail -> lower threshold -> more topics.
        Lower detail -> higher threshold -> fewer topics.
        """
        if self.min_user_pct_override is not None:
            return max(1, math.ceil(total_users * self.min_user_pct_override / 100.0))
        base = 6 + 0.42 * math.sqrt(total_users)
        scaled = base / self.detail
        return max(2, round(scaled))

    def _run_core_stages(self, exclude_words, min_users, corpus=None):
        """Run stages 1-6: ingest, stopwords, anchors, graph, Leiden, connectivity.

        Args:
            exclude_words: set of words to exclude
            min_users: int, minimum user count for anchors and pairs
            corpus: optional pre-ingested TokenizedCorpus (avoids re-reading)

        Returns dict with all intermediate results.
        """
        if corpus is None:
            corpus = ingest_corpus(
                self.input_path,
                stopword_sensitivity=self.stopword_sensitivity,
                exclude_words=exclude_words,
            )

        anchor_words = select_anchor_words(corpus, min_users)

        pair_freq, pair_users, post_pairs = build_cooccurrence_pairs(
            corpus, anchor_words, self.radius, min_users
        )

        if not pair_freq:
            raise WagCoreError(
                "No co-occurrence pairs found. Try raising --detail "
                "or increasing --radius."
            )

        graph, word_to_index, index_to_word = build_graph(
            pair_freq, pair_users, self.weight_by
        )

        membership, num_clusters = run_leiden(graph, self.resolution)

        clusters, word_to_cluster = compute_cluster_info(
            membership, index_to_word, corpus
        )

        # enforce cluster size cap via sub-clustering
        clusters, word_to_cluster, size_cap_exclusions = \
            enforce_cluster_size_cap(
                graph, clusters, word_to_cluster,
                index_to_word, word_to_index, corpus,
                self.max_cluster_words, self.resolution
            )

        # rebuild membership to reflect any sub-clustering changes
        membership = [word_to_cluster.get(index_to_word.get(i), -1)
                      for i in range(graph.vcount())]
        num_clusters = len(clusters)

        connectivity = compute_connectivity(
            graph, word_to_cluster, word_to_index, corpus
        )

        return {
            'corpus': corpus,
            'anchor_words': anchor_words,
            'pair_freq': pair_freq,
            'pair_users': pair_users,
            'post_pairs': post_pairs,
            'graph': graph,
            'word_to_index': word_to_index,
            'index_to_word': index_to_word,
            'membership': membership,
            'num_clusters': num_clusters,
            'clusters': clusters,
            'word_to_cluster': word_to_cluster,
            'connectivity': connectivity,
            'size_cap_exclusions': size_cap_exclusions,
        }

    def run(self):
        """Execute the full pipeline.

        If max_adjacent_topics is set, runs iterative pruning loop.
        Otherwise runs a single pass.

        Returns dict with summary info.
        """
        logger.info("Starting WAG pipeline")
        logger.info("  input: %s", self.input_path)
        logger.info("  output: %s", self.output_dir)
        logger.info("  detail: %.2f", self.detail)
        if self.min_user_pct_override is not None:
            logger.info("  min_user_pct override: %.2f%%",
                        self.min_user_pct_override)
        logger.info("  radius: %d", self.radius)
        logger.info("  stopword_sensitivity: %.2f", self.stopword_sensitivity)
        logger.info("  resolution: %.2f", self.resolution)
        logger.info("  weight_by: %s", self.weight_by)
        if self.max_cluster_words is not None:
            logger.info("  max_cluster_words: %d", self.max_cluster_words)

        # load initial exclude words
        exclude_words = load_exclude_words(self.exclude_words_path)
        initial_exclude_count = len(exclude_words)

        # do a quick ingest to get total_users for auto-scaling
        # (first iteration will re-ingest, but this is needed for the log)
        corpus_peek = ingest_corpus(
            self.input_path,
            stopword_sensitivity=self.stopword_sensitivity,
            exclude_words=exclude_words,
        )
        min_users = self._compute_min_users(corpus_peek.total_users)
        min_user_pct_effective = 100.0 * min_users / corpus_peek.total_users
        logger.info("  total_users: %d -> min_users: %d (%.2f%%)",
                    corpus_peek.total_users, min_users, min_user_pct_effective)

        # store for _get_params output
        self._effective_min_users = min_users
        self._effective_min_user_pct = min_user_pct_effective

        iterations_info = None

        if self.max_adjacent_topics is not None:
            logger.info("  max_adjacent_topics: %d (iterative pruning enabled)",
                        self.max_adjacent_topics)

            epoch = 0
            max_epochs = self.max_iterations  # None = unlimited
            reuse_corpus = corpus_peek  # reuse first ingest

            while max_epochs is None or epoch < max_epochs:
                epoch += 1
                logger.info("=== Iteration %d (exclude words: %d) ===",
                            epoch, len(exclude_words))

                result = self._run_core_stages(
                    exclude_words, min_users, corpus=reuse_corpus)
                reuse_corpus = None  # only reuse on first iteration

                overconnected, max_conn, converged = find_overconnected_words(
                    result['connectivity'], self.max_adjacent_topics
                )

                size_cap_exclusions = result.get('size_cap_exclusions', set())

                if converged and not size_cap_exclusions:
                    logger.info("Converged at iteration %d, "
                                "max connectivity = %d <= %d, "
                                "all clusters within size cap",
                                epoch, max_conn, self.max_adjacent_topics)
                    break

                new_excludes = set(overconnected) | size_cap_exclusions

                if not new_excludes:
                    logger.info("No new words to exclude, stopping at "
                                "iteration %d", epoch)
                    break

                if overconnected:
                    logger.info("Excluding %d words at connectivity=%d: %s",
                                len(overconnected), max_conn,
                                ', '.join(overconnected[:10]))
                if size_cap_exclusions:
                    logger.info("Excluding %d words from oversized clusters: %s",
                                len(size_cap_exclusions),
                                ', '.join(sorted(size_cap_exclusions)[:10]))

                exclude_words = exclude_words | new_excludes
                write_exclude_words(self.output_dir, exclude_words)
            else:
                logger.warning("Reached max iterations (%d) without convergence",
                               max_epochs)

            iterations_info = {
                'count': epoch,
                'final_max_conn': max_conn,
                'total_excluded': len(exclude_words),
            }
        else:
            result = self._run_core_stages(
                exclude_words, min_users, corpus=corpus_peek)

        # stages 7-9: classification, n-grams, output
        corpus = result['corpus']

        classifications = classify_posts(
            corpus, result['clusters'], result['word_to_cluster'],
            result['pair_freq'], result['post_pairs'],
            anchor_words=result['anchor_words'],
        )

        class_stats = compute_class_stats(
            classifications, result['clusters'], corpus
        )

        cluster_ngrams = compute_ngrams(
            corpus, classifications, result['clusters'], corpus.stopwords
        )

        write_all_outputs(
            self.output_dir,
            corpus=corpus,
            clusters=result['clusters'],
            word_to_cluster=result['word_to_cluster'],
            word_to_index=result['word_to_index'],
            index_to_word=result['index_to_word'],
            membership=result['membership'],
            pair_freq=result['pair_freq'],
            pair_users=result['pair_users'],
            connectivity=result['connectivity'],
            classifications=classifications,
            class_stats=class_stats,
            cluster_ngrams=cluster_ngrams,
            exclude_words=exclude_words,
            params=self._get_params(),
            iterations=iterations_info,
        )

        topic_count = len(result['clusters'])
        logger.info("Pipeline complete: %d topics found", topic_count)

        return {
            'topic_count': topic_count,
            'strong_count': class_stats['strong_count'],
            'weak_count': class_stats['weak_count'],
            'none_count': class_stats['none_count'],
            'total_posts': corpus.total_posts,
            'total_users': corpus.total_users,
            'stopwords_detected': len(corpus.stopwords),
            'exclude_words': len(exclude_words),
        }
