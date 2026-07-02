"""Tiny executable-function library for Win Rate tests.

The official NESTFUL scorer (calculate_ans) discovers function names by reading
lines that start with `def ` in this file and then `exec`-imports it, calling
each function positionally. Keep these as simple top-level `def`s.
"""


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b
