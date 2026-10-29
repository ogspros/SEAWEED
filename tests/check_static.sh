#!/usr/bin/env bash
# The part of the test that needs no seismic data.
#
# tests/smoke_test.sh can only run where the SEG-Y and the cube are.  This
# script runs anywhere, in seconds, and catches what a refactoring breaks most
# often: a file that no longer compiles, a shell script with a typo in it, a
# family letter that the figure script does not expect, and a constant that
# drifted apart between two scripts that must share it.
#
#   bash tests/check_static.sh
#
# Exit 0 when everything passes, 1 otherwise.  Run it before the smoke test:
# it costs nothing and it fails faster.
set -uo pipefail

cd "$(dirname "$0")/.."
status=0

PY=${PY:-python3}
version=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "[1] interpreter: $PY $version"
if [ "$version" != "3.10" ]; then
  echo "    the runs were made on Python 3.10.  Compiling under $version"
  echo "    accepts f-strings that 3.10 rejects, so a clean result here is not"
  echo "    proof that the code still runs on Python 3.10."
fi

# ------------------------------------------------------ 2. every file compiles
# compile() and not py_compile: the syntax is checked in memory, nothing is
# imported, so a missing numpy cannot turn this into a failure, and no
# __pycache__ is left in a working tree whose .gitignore is deliberately narrow.
echo "[2] compiling every tracked Python file"
if ! git ls-files '*.py' 2>/dev/null | "$PY" tests/_syntax_check.py; then
  status=1
fi

# The one construction this project is known to trip over: an expression that
# runs over several lines inside the braces of an f-string.  Valid from 3.12,
# a SyntaxError on the 3.10 of the runs.  Reported as a warning, because the
# scan is textual and a newer interpreter cannot decide it.
echo "[3] f-strings with a multi-line expression (3.10 rejects them)"
git ls-files '*.py' 2>/dev/null | "$PY" tests/_fstring_scan.py

# ------------------------------------------------------- 4. the shell scripts
echo "[4] parsing every tracked shell script"
scripts=$(git ls-files '*.sh' 2>/dev/null || find . -name '*.sh' -not -path './.git/*')
n=0; bad=0
for f in $scripts; do
  n=$((n + 1))
  if ! out=$(bash -n "$f" 2>&1); then
    echo "    FAIL $f"
    echo "$out" | sed 's/^/         /'
    bad=$((bad + 1))
  fi
done
echo "    $((n - bad))/$n parse"
[ "$bad" -eq 0 ] || status=1

# --------------------------------------------- 5. run names and shared constants
echo "[5] Viking families and reference constants"
if "$PY" tests/check_families.py > /tmp/check_families.$$ 2>&1; then
  tail -n 1 /tmp/check_families.$$ | sed 's/^/    /'
else
  sed 's/^/    /' /tmp/check_families.$$
  status=1
fi
rm -f /tmp/check_families.$$

# ------------------------------------------------- 6. the frozen reference itself
echo "[6] frozen reference"
if [ -e tests/reference_md5.txt ] && [ -d tests/reference ] \
     && [ -e tests/reference_stats.tsv ]; then
  if bash tests/smoke_test.sh --offline > /tmp/offline.$$ 2>&1; then
    echo "    $(grep -c ': OK' /tmp/offline.$$) frozen wavelets match their hashes"
  else
    sed 's/^/    /' /tmp/offline.$$
    status=1
  fi
  rm -f /tmp/offline.$$
else
  echo "    absent or incomplete: run tests/freeze_reference.sh on the machine"
  echo "    that holds the data"
fi

echo
[ "$status" -eq 0 ] && echo "PASS: nothing that can be checked without data is broken" \
                    || echo "FAIL: see above"
exit "$status"
