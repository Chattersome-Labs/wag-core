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

"""CLI entry point for wag_core. Run with: python -m wag_core"""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import WagPipeline
from .exceptions import WagCoreError


def parse_args():
    parser = argparse.ArgumentParser(
        prog='wag_core',
        description='Word Adjacency Graph topic detection engine. '
                    'Detects topics in text collections using word co-occurrence '
                    'graphs and Leiden community detection.',
    )

    parser.add_argument('--input', required=True, type=str,
                        help='Tab-separated input file: user_id<TAB>post_text')
    parser.add_argument('--output-dir', required=True, type=str,
                        help='Directory for all output files')
    parser.add_argument('--min-user-pct', type=float, default=1.0,
                        help='Minimum %% of unique users required per anchor '
                             'word and per word pair (default: 1.0)')
    parser.add_argument('--radius', type=int, default=1,
                        help='Co-occurrence radius in tokens (default: 1)')
    parser.add_argument('--stopword-sensitivity', type=float, default=0.6,
                        help='Stopword detection aggressiveness 0.0-1.0 '
                             '(default: 0.6). 0=permissive, 1=aggressive')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Leiden resolution parameter (default: 1.0)')
    parser.add_argument('--max-adjacent-topics', type=int, default=3,
                        help='Max adjacent clusters per word before exclusion '
                             '(iterative pruning). Set to 0 to disable. '
                             '(default: 3)')
    parser.add_argument('--max-iterations', type=int, default=0,
                        help='Max pruning iterations. 0 = unlimited (default: 0)')
    parser.add_argument('--exclude-words', type=str, default=None,
                        help='Path to file with words to exclude (one per line)')
    parser.add_argument('--weight-by', type=str, default='users',
                        choices=['frequency', 'users'],
                        help='Edge weight method (default: users)')

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        pipeline = WagPipeline(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            min_user_pct=args.min_user_pct,
            radius=args.radius,
            stopword_sensitivity=args.stopword_sensitivity,
            resolution=args.resolution,
            exclude_words_path=Path(args.exclude_words) if args.exclude_words else None,
            max_adjacent_topics=args.max_adjacent_topics or None,
            max_iterations=args.max_iterations or None,
            weight_by=args.weight_by,
        )

        result = pipeline.run()

        print('\n=== Results ===')
        print('Topics found: %d' % result['topic_count'])
        print('Posts: %d total, %d strong, %d weak, %d none' % (
            result['total_posts'], result['strong_count'],
            result['weak_count'], result['none_count']))
        print('Stopwords detected: %d' % result['stopwords_detected'])
        print('Exclude words: %d' % result['exclude_words'])
        print('Output: %s' % args.output_dir)

    except WagCoreError as e:
        logging.getLogger('wag_core').error("Pipeline failed: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logging.getLogger('wag_core').info("Interrupted by user")
        sys.exit(130)


if __name__ == '__main__':
    main()
