############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# __init__.py: Runtime support for full Scheme: tagged values, heap, symbol table.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Runtime support for full Scheme: tagged values, heap, symbol table."""

from neural_compiler.runtime.tagged_value import (
    TAG_DIM,
    PAYLOAD_DIM,
    VALUE_DIM,
    NIL,
    BOOL,
    INT,
    FLOAT,
    CHAR,
    SYMBOL,
    PAIR,
    STRING,
    CLOSURE,
    VECTOR,
    make_nil,
    make_bool,
    make_int,
    make_float,
    make_char,
    make_symbol,
    make_pair,
    make_string,
    make_closure,
    make_vector,
    extract_tag,
    extract_payload,
    type_index,
    type_name,
    is_type,
    is_nil,
    is_pair,
    is_number,
    is_symbol,
    is_closure,
    unwrap_number,
    unwrap_bool,
    unwrap_pair_addrs,
    unwrap_closure,
    tagged_if,
    soft_select,
    from_scalar,
    to_scalar,
)
from neural_compiler.runtime.heap import TensorHeap
from neural_compiler.runtime.symbols import SymbolTable

__all__ = [
    "TAG_DIM", "PAYLOAD_DIM", "VALUE_DIM",
    "NIL", "BOOL", "INT", "FLOAT", "CHAR", "SYMBOL", "PAIR",
    "STRING", "CLOSURE", "VECTOR",
    "make_nil", "make_bool", "make_int", "make_float", "make_char",
    "make_symbol", "make_pair", "make_string", "make_closure", "make_vector",
    "extract_tag", "extract_payload", "type_index", "type_name",
    "is_type", "is_nil", "is_pair", "is_number", "is_symbol", "is_closure",
    "unwrap_number", "unwrap_bool", "unwrap_pair_addrs", "unwrap_closure",
    "tagged_if", "soft_select", "from_scalar", "to_scalar",
    "TensorHeap", "SymbolTable",
]
