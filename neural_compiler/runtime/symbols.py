############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# symbols.py: Symbol interning: bidirectional mapping between string names and integer IDs. Symbols are represented as...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Symbol interning: bidirectional mapping between string names and integer IDs.

Symbols are represented as integer IDs in the tensor representation.
The SymbolTable maintains the mapping so we can convert back to names
for display/debugging. Used at compile time to assign IDs to all symbol
literals in a program.
"""

from __future__ import annotations


class SymbolTable:
    """Intern string names to integer IDs for tensor-backed symbol values."""

    def __init__(self) -> None:
        self._str_to_id: dict[str, int] = {}
        self._id_to_str: list[str] = []

    def intern(self, name: str) -> int:
        """Get or create an integer ID for a symbol name."""
        if name not in self._str_to_id:
            idx = len(self._id_to_str)
            self._str_to_id[name] = idx
            self._id_to_str.append(name)
        return self._str_to_id[name]

    def name(self, sym_id: int) -> str:
        """Look up the string name for a symbol ID."""
        if sym_id < 0 or sym_id >= len(self._id_to_str):
            raise KeyError(f"Unknown symbol ID: {sym_id}")
        return self._id_to_str[sym_id]

    def contains(self, name: str) -> bool:
        return name in self._str_to_id

    def __len__(self) -> int:
        return len(self._id_to_str)

    def __repr__(self) -> str:
        return f"SymbolTable({self._id_to_str})"
