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

"""
wag_core - Word Adjacency Graph topic detection engine

A Python package for detecting topics in text collections using
word co-occurrence graphs and Leiden community detection.

Works with any language, any domain (social media, logs, technical docs).
"""

__version__ = '0.1.0'

from .pipeline import WagPipeline
from .exceptions import WagCoreError, InputError
