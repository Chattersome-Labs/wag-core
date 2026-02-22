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

"""Custom exception hierarchy for wag_core."""


class WagCoreError(Exception):
    """Base exception for all wag_core errors."""
    pass


class InputError(WagCoreError):
    """Raised for invalid or missing input files."""
    pass


class TokenizationError(WagCoreError):
    """Raised when tokenization or stopword detection fails."""
    pass


class GraphError(WagCoreError):
    """Raised when graph construction or clustering fails."""
    pass


class ClassificationError(WagCoreError):
    """Raised when post classification encounters errors."""
    pass


class OutputError(WagCoreError):
    """Raised when output file writing fails."""
    pass
