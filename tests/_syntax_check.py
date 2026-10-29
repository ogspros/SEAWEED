#!/usr/bin/env python3
"""Compile the Python files named on stdin, in memory.

Used by tests/check_static.sh.  compile() rather than an import, so that a
missing numpy or a missing SEG-Y cannot turn a syntax check into a failure,
and rather than py_compile, so that no __pycache__ appears in the working tree
of a repository whose .gitignore is deliberately narrow.
"""

import sys

bad = 0
paths = [line.strip() for line in sys.stdin if line.strip()]
if not paths:
    sys.exit("    no Python file given")
for path in paths:
    try:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        compile(src, path, "exec")
    except SyntaxError as exc:
        bad += 1
        print("    FAIL %s:%s: %s" % (path, exc.lineno, exc.msg))
    except (OSError, UnicodeDecodeError) as exc:
        bad += 1
        print("    FAIL %s: %s" % (path, exc))
print("    %d/%d compile" % (len(paths) - bad, len(paths)))
sys.exit(1 if bad else 0)
