# SPDX-FileCopyrightText: 2025-present U.N. Owen <void@some.where>
#
# SPDX-License-Identifier: MIT
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
