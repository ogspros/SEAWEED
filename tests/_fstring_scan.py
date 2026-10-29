#!/usr/bin/env python3
"""Report f-strings whose braces hold an expression spread over two lines.

Used by tests/check_static.sh.  Python 3.12 accepts them; the 3.10 of the runs
raises SyntaxError, and a newer interpreter therefore cannot tell whether the
file still runs on Python 3.10.  The scan is textual and deliberately loud rather
than clever: it reports candidates for a person to look at.
"""

import re
import sys

TRIPLE = re.compile(r'(?:rf|fr|f|F)("""|\'\'\')(.*?)\1', re.S)
BRACE = re.compile(r"(?<!\{)\{([^{}]*)\}")

hits = []
for path in (line.strip() for line in sys.stdin):
    if not path:
        continue
    try:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
    except (OSError, UnicodeDecodeError):
        continue
    for m in TRIPLE.finditer(src):
        body = m.group(2)
        for expr in BRACE.finditer(body):
            if "\n" in expr.group(1):
                line = (src[:m.start()].count("\n") + 1
                        + body[:expr.start()].count("\n"))
                hits.append("%s:%d" % (path, line))
for h in hits:
    print("    warning %s" % h)
print("    %d suspect f-string%s" % (len(hits), "" if len(hits) == 1 else "s"))
