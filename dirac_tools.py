"""Routines for manipulation expressions containing DiracDeltas"""
import sympy


def find_dirac_delta_terms(expr):
    """Return all sub-expressions of the form "x * δ(y)" in `expr`"""
    _w = sympy.Wild('w')
    _w2 = sympy.Wild('w2')
    return list(
        term
        for term in expr.find(_w * sympy.DiracDelta(_w2))
        if isinstance(term, sympy.Mul)
    )


def split_dirac_delta(expr):
    """Given an expression of the form "x * δ(y) * z", return a tuple
    (x*z, δ(y))"""
    if not isinstance(expr, sympy.Mul):
        raise ValueError("Unexpected expr %s: not a product" % expr)
    coeff = sympy.sympify(1)
    delta = None
    for term in expr.args:
        if isinstance(term, sympy.DiracDelta):
            if delta is None:
                delta = term
            else:
                raise ValueError(
                    "Unexpected expr %s: more than one delta" % expr
                )
        else:
            coeff *= term
    if delta is None:
        raise ValueError(
            "Unexpected expr %s: not proportional to delta" % expr
        )
    return (coeff, delta)


def normalize_dirac_delta_term(term, target):
    """Given a term of the form "f(n) δ(g(n))", shift n such
    that g(n) is `target`"""
    n = sympy.symbols('n', integer=True)
    coeff, delta = split_dirac_delta(term)
    delta_arg = delta.args[0]
    if delta_arg == target:
        return term
    else:
        for shift in [1, -1]:
            mapping = {n: n + shift}
            delta_arg_new = delta_arg.subs(mapping).simplify()
            if delta_arg_new == target:
                return coeff.subs(mapping) * sympy.DiracDelta(delta_arg_new)
        raise ValueError("Cannot bring %s to δ(%s)" % (term, target))


def normalize_dirac_delta_terms(expr, target):
    """Normalize all Dirac delta functions such that they have the same
    argument `target`"""
    mapping = {
        term: normalize_dirac_delta_term(term, target)
        for term in find_dirac_delta_terms(expr)
    }
    return expr.subs(mapping)
