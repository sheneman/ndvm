############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# scheme_parser.py: Tokenizer and recursive-descent parser for the Scheme subset.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tokenizer and recursive-descent parser for the Scheme subset."""

from __future__ import annotations

from neural_compiler.parser.ast_nodes import (
    ASTNode,
    App,
    Begin,
    Const,
    Define,
    If,
    Lambda,
    Let,
    Letrec,
    Loop,
    Program,
    Quote,
    Recur,
    SchemeChar,
    SoftChoice,
    Var,
    PRIMITIVES,
)

NAMED_CHARS = {
    "space": " ", "newline": "\n", "tab": "\t",
    "return": "\r", "nul": "\x00", "null": "\x00",
    "alarm": "\x07", "backspace": "\x08", "delete": "\x7f",
    "escape": "\x1b",
}


def tokenize(source: str) -> list[str]:
    """Split Scheme source into tokens (strings, parens, atoms, quote)."""
    tokens: list[str] = []
    i = 0
    while i < len(source):
        c = source[i]
        if c.isspace():
            i += 1
        elif c == ";":
            while i < len(source) and source[i] != "\n":
                i += 1
        elif c == "'":
            tokens.append("'")
            i += 1
        elif c == "`":
            tokens.append("`")
            i += 1
        elif c == ",":
            if i + 1 < len(source) and source[i + 1] == "@":
                tokens.append(",@")
                i += 2
            else:
                tokens.append(",")
                i += 1
        elif c in ("(", ")", "[", "]"):
            tokens.append(c)
            i += 1
        elif c == "#":
            if i + 1 < len(source) and source[i + 1] == "\\":
                if i + 2 >= len(source):
                    raise SyntaxError(f"Incomplete character literal at position {i}")
                j = i + 2
                while j < len(source) and source[j] not in ("(", ")", "[", "]", "'", "`", ",", " ", "\t", "\n", "\r", ";"):
                    j += 1
                tokens.append(source[i:j])
                i = j
            elif i + 1 < len(source) and source[i + 1] in ("t", "f"):
                tokens.append(source[i : i + 2])
                i += 2
            else:
                raise SyntaxError(f"Unexpected '#' at position {i}")
        else:
            j = i
            while j < len(source) and source[j] not in ("(", ")", "[", "]", "'", "`", ",", " ", "\t", "\n", "\r", ";"):
                j += 1
            tokens.append(source[i:j])
            i = j
    return tokens


def _parse_sexpr(tokens: list[str], pos: int) -> tuple[object, int]:
    """Parse one S-expression, returning (sexpr, new_pos).

    An S-expression is either an atom (str) or a list of S-expressions.
    """
    if pos >= len(tokens):
        raise SyntaxError("Unexpected end of input")

    token = tokens[pos]

    if token == "(":
        pos += 1
        items = []
        while pos < len(tokens) and tokens[pos] != ")":
            item, pos = _parse_sexpr(tokens, pos)
            items.append(item)
        if pos >= len(tokens):
            raise SyntaxError("Missing closing parenthesis")
        pos += 1  # skip ')'
        return items, pos

    if token == "[":
        pos += 1
        items = ["vec"]
        while pos < len(tokens) and tokens[pos] != "]":
            item, pos = _parse_sexpr(tokens, pos)
            items.append(item)
        if pos >= len(tokens):
            raise SyntaxError("Missing closing bracket")
        pos += 1  # skip ']'
        return items, pos

    if token == "'":
        quoted, pos = _parse_sexpr(tokens, pos + 1)
        return ["quote", quoted], pos

    if token == "`":
        datum, pos = _parse_sexpr(tokens, pos + 1)
        return _expand_quasiquote(datum), pos

    if token == ",":
        datum, pos = _parse_sexpr(tokens, pos + 1)
        return ["unquote", datum], pos

    if token == ",@":
        datum, pos = _parse_sexpr(tokens, pos + 1)
        return ["unquote-splicing", datum], pos

    if token in (")", "]"):
        raise SyntaxError(f"Unexpected '{token}' at token position {pos}")

    return token, pos + 1


def _atom_to_ast(atom: str) -> ASTNode:
    """Convert an atom string to a Const or Var node."""
    if atom == "#t":
        return Const(True)
    if atom == "#f":
        return Const(False)
    if atom.startswith("#\\"):
        char_name = atom[2:]
        if char_name in NAMED_CHARS:
            return Const(SchemeChar(ord(NAMED_CHARS[char_name])))
        if len(char_name) == 1:
            return Const(SchemeChar(ord(char_name)))
        raise SyntaxError(f"Unknown character literal: {atom}")
    try:
        return Const(int(atom))
    except ValueError:
        pass
    try:
        return Const(float(atom))
    except ValueError:
        pass
    return Var(atom)


def _sexpr_to_ast(sexpr: object) -> ASTNode:
    """Convert a nested S-expression (lists and strings) to an AST."""
    if isinstance(sexpr, str):
        return _atom_to_ast(sexpr)

    if not isinstance(sexpr, list) or len(sexpr) == 0:
        raise SyntaxError(f"Empty application: {sexpr}")

    head = sexpr[0]

    if head == "quote":
        if len(sexpr) != 2:
            raise SyntaxError(f"'quote' requires 1 argument, got {len(sexpr) - 1}")
        return Quote(datum=sexpr[1])

    if head == "quasiquote":
        if len(sexpr) != 2:
            raise SyntaxError(f"'quasiquote' requires 1 argument")
        expanded = _expand_quasiquote(sexpr[1])
        return _sexpr_to_ast(expanded)

    if head == "begin":
        if len(sexpr) < 2:
            raise SyntaxError("'begin' requires at least one expression")
        exprs = tuple(_sexpr_to_ast(e) for e in sexpr[1:])
        return Begin(exprs=exprs)

    if head == "define":
        if len(sexpr) < 3:
            raise SyntaxError(f"'define' requires name and value, got {len(sexpr) - 1} parts")
        target = sexpr[1]
        if isinstance(target, list):
            name = target[0]
            params = target[1:]
            if not isinstance(name, str):
                raise SyntaxError(f"'define' function name must be identifier: {name}")
            if not all(isinstance(p, str) for p in params):
                raise SyntaxError(f"'define' params must be identifiers: {params}")
            body = _sexpr_to_ast(sexpr[2]) if len(sexpr) == 3 else Begin(
                exprs=tuple(_sexpr_to_ast(e) for e in sexpr[2:])
            )
            return Define(name=name, value=Lambda(params=tuple(params), body=body))
        if not isinstance(target, str):
            raise SyntaxError(f"'define' target must be identifier or (name params...): {target}")
        return Define(name=target, value=_sexpr_to_ast(sexpr[2]))

    if head == "cond":
        return _desugar_cond(sexpr[1:])

    if head == "if":
        if len(sexpr) != 4:
            raise SyntaxError(f"'if' requires 3 arguments, got {len(sexpr) - 1}")
        return If(
            test=_sexpr_to_ast(sexpr[1]),
            then_=_sexpr_to_ast(sexpr[2]),
            else_=_sexpr_to_ast(sexpr[3]),
        )

    if head == "lambda":
        if len(sexpr) < 3:
            raise SyntaxError(f"'lambda' requires params and body, got {len(sexpr) - 1} parts")
        params = sexpr[1]
        if not isinstance(params, list) or not all(isinstance(p, str) for p in params):
            raise SyntaxError(f"'lambda' params must be a list of identifiers: {params}")
        body = _sexpr_to_ast(sexpr[2]) if len(sexpr) == 3 else Begin(
            exprs=tuple(_sexpr_to_ast(e) for e in sexpr[2:])
        )
        return Lambda(
            params=tuple(params),
            body=body,
        )

    if head == "let":
        if len(sexpr) < 3:
            raise SyntaxError(f"'let' requires bindings and body, got {len(sexpr) - 1} parts")
        raw_bindings = sexpr[1]
        if not isinstance(raw_bindings, list):
            raise SyntaxError(f"'let' bindings must be a list: {raw_bindings}")
        bindings = []
        for b in raw_bindings:
            if not isinstance(b, list) or len(b) != 2 or not isinstance(b[0], str):
                raise SyntaxError(f"Invalid let binding: {b}")
            bindings.append((b[0], _sexpr_to_ast(b[1])))
        body = _sexpr_to_ast(sexpr[2]) if len(sexpr) == 3 else Begin(
            exprs=tuple(_sexpr_to_ast(e) for e in sexpr[2:])
        )
        return Let(
            bindings=tuple(bindings),
            body=body,
        )

    if head == "letrec":
        if len(sexpr) < 3:
            raise SyntaxError(f"'letrec' requires bindings and body, got {len(sexpr) - 1} parts")
        raw_bindings = sexpr[1]
        if not isinstance(raw_bindings, list):
            raise SyntaxError(f"'letrec' bindings must be a list: {raw_bindings}")
        bindings = []
        for b in raw_bindings:
            if not isinstance(b, list) or len(b) != 2 or not isinstance(b[0], str):
                raise SyntaxError(f"Invalid letrec binding: {b}")
            rhs = _sexpr_to_ast(b[1])
            if not isinstance(rhs, Lambda):
                raise SyntaxError(f"letrec binding '{b[0]}' must be a lambda expression")
            bindings.append((b[0], rhs))
        body = _sexpr_to_ast(sexpr[2]) if len(sexpr) == 3 else Begin(
            exprs=tuple(_sexpr_to_ast(e) for e in sexpr[2:])
        )
        return Letrec(
            bindings=tuple(bindings),
            body=body,
        )

    if head == "let*":
        if len(sexpr) < 3:
            raise SyntaxError(f"'let*' requires bindings and body")
        raw_bindings = sexpr[1]
        if not isinstance(raw_bindings, list):
            raise SyntaxError(f"'let*' bindings must be a list")
        body = _sexpr_to_ast(sexpr[2]) if len(sexpr) == 3 else Begin(
            exprs=tuple(_sexpr_to_ast(e) for e in sexpr[2:])
        )
        for b in reversed(raw_bindings):
            if not isinstance(b, list) or len(b) != 2 or not isinstance(b[0], str):
                raise SyntaxError(f"Invalid let* binding: {b}")
            body = Let(bindings=((b[0], _sexpr_to_ast(b[1])),), body=body)
        return body

    if head == "when":
        if len(sexpr) < 3:
            raise SyntaxError("'when' requires test and body")
        test = _sexpr_to_ast(sexpr[1])
        body = _sexpr_to_ast(sexpr[2]) if len(sexpr) == 3 else Begin(
            exprs=tuple(_sexpr_to_ast(e) for e in sexpr[2:])
        )
        return If(test=test, then_=body, else_=Const(False))

    if head == "unless":
        if len(sexpr) < 3:
            raise SyntaxError("'unless' requires test and body")
        test = _sexpr_to_ast(sexpr[1])
        body = _sexpr_to_ast(sexpr[2]) if len(sexpr) == 3 else Begin(
            exprs=tuple(_sexpr_to_ast(e) for e in sexpr[2:])
        )
        return If(test=test, then_=Const(False), else_=body)

    if head == "soft-choice":
        if len(sexpr) != 3:
            raise SyntaxError(
                f"'soft-choice' requires options list and weights, got {len(sexpr) - 1} parts"
            )
        raw_options = sexpr[1]
        if not isinstance(raw_options, list) or len(raw_options) < 2:
            raise SyntaxError("'soft-choice' options must be a list of at least 2 expressions")
        options = tuple(_sexpr_to_ast(o) for o in raw_options)
        weights = _sexpr_to_ast(sexpr[2])
        return SoftChoice(options=options, weights=weights)

    if head == "loop":
        if len(sexpr) != 3:
            raise SyntaxError(f"'loop' requires bindings and body, got {len(sexpr) - 1} parts")
        raw_bindings = sexpr[1]
        if not isinstance(raw_bindings, list):
            raise SyntaxError(f"'loop' bindings must be a list: {raw_bindings}")
        bindings = []
        for b in raw_bindings:
            if not isinstance(b, list) or len(b) != 2 or not isinstance(b[0], str):
                raise SyntaxError(f"Invalid loop binding: {b}")
            bindings.append((b[0], _sexpr_to_ast(b[1])))
        return Loop(
            bindings=tuple(bindings),
            body=_sexpr_to_ast(sexpr[2]),
        )

    if head == "recur":
        if len(sexpr) < 2:
            raise SyntaxError("'recur' requires at least one argument")
        args = tuple(_sexpr_to_ast(a) for a in sexpr[1:])
        return Recur(args=args)

    func = _sexpr_to_ast(head)
    args = tuple(_sexpr_to_ast(a) for a in sexpr[1:])
    return App(func=func, args=args)


def _expand_quasiquote(datum: object) -> object:
    """Expand quasiquoted datum to S-expression using list/cons/append/quote."""
    if isinstance(datum, str):
        return ["quote", datum]
    if not isinstance(datum, list):
        return ["quote", datum]
    if len(datum) == 0:
        return ["quote", []]
    if len(datum) == 2 and datum[0] == "unquote":
        return datum[1]
    has_splicing = any(
        isinstance(el, list) and len(el) == 2 and el[0] == "unquote-splicing"
        for el in datum
    )
    if has_splicing:
        segments = []
        for el in datum:
            if isinstance(el, list) and len(el) == 2 and el[0] == "unquote-splicing":
                segments.append(el[1])
            elif isinstance(el, list) and len(el) == 2 and el[0] == "unquote":
                segments.append(["list", el[1]])
            else:
                segments.append(["list", _expand_quasiquote(el)])
        return ["append"] + segments
    else:
        return ["list"] + [
            el[1] if isinstance(el, list) and len(el) == 2 and el[0] == "unquote"
            else _expand_quasiquote(el)
            for el in datum
        ]


def _desugar_cond(clauses: list) -> ASTNode:
    """Desugar (cond (test expr)... (else expr)) to nested if."""
    if not clauses:
        raise SyntaxError("'cond' requires at least one clause")

    clause = clauses[0]
    if not isinstance(clause, list) or len(clause) < 2:
        raise SyntaxError(f"Invalid cond clause: {clause}")

    if clause[0] == "else":
        if len(clauses) > 1:
            raise SyntaxError("'else' clause must be last in cond")
        return _sexpr_to_ast(clause[1])

    test = _sexpr_to_ast(clause[0])
    then_expr = _sexpr_to_ast(clause[1])

    if len(clauses) == 1:
        return If(test=test, then_=then_expr, else_=Const(False))

    return If(test=test, then_=then_expr, else_=_desugar_cond(clauses[1:]))


def parse(source: str) -> ASTNode:
    """Parse a Scheme source string into an AST (single expression)."""
    tokens = tokenize(source)
    if not tokens:
        raise SyntaxError("Empty input")
    sexpr, pos = _parse_sexpr(tokens, 0)
    if pos < len(tokens):
        raise SyntaxError(f"Unexpected tokens after expression: {tokens[pos:]}")
    return _sexpr_to_ast(sexpr)


def parse_program(source: str) -> ASTNode:
    """Parse a Scheme program (multiple top-level forms) into a Program AST.

    If only one expression is present and it's not a define, returns it directly
    for backward compatibility.
    """
    tokens = tokenize(source)
    if not tokens:
        raise SyntaxError("Empty input")

    forms = []
    pos = 0
    while pos < len(tokens):
        sexpr, pos = _parse_sexpr(tokens, pos)
        forms.append(_sexpr_to_ast(sexpr))

    if len(forms) == 1 and not isinstance(forms[0], Define):
        return forms[0]

    return Program(forms=tuple(forms))
