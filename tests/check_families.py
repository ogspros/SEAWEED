#!/usr/bin/env python3
"""Check that the Viking run names and the analysis scripts still agree.

This test needs no data and no numpy.  It exists because the one silent error
this project has actually suffered was of this kind: a family letter that the
figure script did not expect drops a point from Fig. 4b and pools two spans
into one, with no error message and a plot that still looks reasonable.

Three things are checked.

1.  Every run name produced by viking_rerun_conv.sh matches the regular
    expression used by make_paper_figures_vik.py and by viking_phase_test.py,
    and the fold and the span written in the name are the fold and the span
    that the shot selection of that run actually gives.

2.  Each family letter means one selection and one only, and the letters mean
    what the documentation says they mean.  The span table of the figure
    script, keyed by shot step, is checked against the spans derived from the
    selections, and the letters it treats as blind selections are checked
    against the letters the run script produces at fold 100.

3.  The constants that build the recorded reference -- band, sampling
    intervals, recording delay, receiver depth, water velocity, wavelet
    length, low-pass corner -- are the same in the figure script and in the
    phase test.  They are written out separately in the two files, and two
    different references would be compared with the same wavelets without
    anything failing.

    python3 tests/check_families.py            # from anywhere in the tree
"""

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RERUN = os.path.join(ROOT, "validation", "viking", "viking_rerun_conv.sh")
FIGS = os.path.join(ROOT, "validation", "viking", "make_paper_figures_vik.py")
PHASE = os.path.join(ROOT, "validation", "viking", "viking_phase_test.py")

# The meaning of each letter, as RUNS.md states it: fold, shot step and the
# span in kilometres that the paper quotes for it.  `None` for the step means
# the selection is made by the workflow itself (--auto_select) and cannot be
# derived from the arguments.
EXPECTED = {"A": (100, 1, 2.5),
            "B": (200, 1, 5.0),
            "C": (100, 10, 24.8),
            "D": (100, 5, 12.4),
            "E": (500, 1, 12.5),
            "F": (1001, 1, 25.0),
            "G": (None, None, None)}

SPAN_TOL_KM = 0.3          # the names round the span; 24.75 km is written L25km

# Constants that both scripts need to build the same reference wavelet.  Left:
# the module constant of make_paper_figures_vik.py.  Right: the argparse
# default of viking_phase_test.py.
SHARED = (("BAND", "band"),
          ("DT_EX", "dt"),
          ("DT_FF", "dt_ff"),
          ("I_DELAY", "delay"),
          ("Z_REC", "z_rec"),
          ("VW", "vw"),
          ("NW", "nw"))

ARITH = re.compile(r"^[\d\s+*()i]+$")
ADD = re.compile(r'^\s*add\s+"(?P<name>[^"]+)"\s+(?P<args>.+?)\s*;?\s*$')
ADD_AUTO = re.compile(r'^\s*add_auto\s+"(?P<name>[^"]+)"\s+(?P<fold>\d+)')
LOOP = re.compile(r'^\s*for\s+i\s+in\s+\$\(seq\s+(?P<a>\d+)\s+(?P<b>\d+)\)\s*;'
                  r'\s*do\s+(?P<body>.*?)\s*;\s*done\s*$')

errors = []
notes = []


def fail(msg):
    errors.append(msg)


def shell_number(token, i):
    """Evaluate one argument of `add`, which is a number, $i, or $((...))."""
    token = token.strip().strip('"')
    if token.startswith("$((") and token.endswith("))"):
        expr = token[3:-2]
        if not ARITH.match(expr):
            raise ValueError("unexpected arithmetic: %s" % token)
        return int(eval(expr, {"__builtins__": {}}, {"i": i}))
    if token in ("$i", "i"):
        return int(i)
    return int(token)


def parse_runs(path):
    """Return the list of runs the re-run script queues: name and selection."""
    runs = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.lstrip().startswith("#"):
                continue
            m = LOOP.match(line)
            if m:
                bodies = [b for b in m.group("body").split(";") if b.strip()]
                for i in range(int(m.group("a")), int(m.group("b")) + 1):
                    for body in bodies:
                        runs.extend(parse_one(body, i))
                continue
            runs.extend(parse_one(line, None))
    return runs


NAME_ARITH = re.compile(r"\$\(\(([^()]+)\)\)")


def expand_name(name, i):
    """Substitute $i and $((...)) in a run name, as the shell would."""
    if i is None:
        return name
    name = NAME_ARITH.sub(lambda m: str(shell_number("$((%s))" % m.group(1), i)),
                          name)
    return name.replace("$i", str(i))


def parse_one(line, i):
    m = ADD_AUTO.match(line)
    if m:
        return [{"name": expand_name(m.group("name"), i), "auto": True,
                 "fold": int(m.group("fold")), "step": None}]
    m = ADD.match(line)
    if m:
        name = expand_name(m.group("name"), i)
        args = m.group("args").split()
        if len(args) < 3:
            fail("cannot read the selection of %s" % name)
            return []
        first, last, step = (shell_number(a, i or 0) for a in args[:3])
        fold = len(range(first, last, step))
        return [{"name": name, "auto": False, "first": first, "last": last,
                 "step": step, "fold": fold}]
    return []


def module_constants(path):
    """Module-level assignments of a script, as far as they are literals."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                out[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
        elif isinstance(target, ast.Tuple):
            # DT_FF, DT_EX, UP = 1e-3, 4e-3, 4 is how this project writes the
            # sampling intervals, and they are exactly the constants to check
            try:
                values = ast.literal_eval(node.value)
            except ValueError:
                continue
            names = [e.id for e in target.elts if isinstance(e, ast.Name)]
            if len(names) == len(target.elts) == len(values):
                out.update(zip(names, values))
    return out


def argparse_defaults(path):
    """The `default=` of every add_argument call of a script."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = str(node.args[0].value).lstrip("-").replace("-", "_")
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    out[name] = ast.literal_eval(kw.value)
                except ValueError:
                    pass
    return out


def source_regex(path, varname="PAT"):
    """The regular expression a script matches run names with."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == varname:
            if isinstance(node.value, ast.Call):        # re.compile(...)
                parts = []
                for arg in node.value.args:
                    parts.append(literal_string(arg))
                return re.compile("".join(p for p in parts if p))
    return None


def literal_string(node):
    try:
        value = ast.literal_eval(node)
    except ValueError:
        return None
    return value if isinstance(value, str) else None


def blind_letters(path):
    """The letters the figure script treats as blind selections."""
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and node.ops \
                and isinstance(node.ops[0], ast.In) \
                and isinstance(node.comparators[0], (ast.Tuple, ast.List, ast.Set)):
            try:
                value = ast.literal_eval(node.comparators[0])
            except ValueError:
                continue
            if value and all(isinstance(v, str) and len(v) == 1 and v.isupper()
                             for v in value):
                return tuple(value)
    return None


def main():
    for path in (RERUN, FIGS, PHASE):
        if not os.path.exists(path):
            sys.exit("missing: %s" % path)

    # ------------------------------------------------------------ the runs
    text = open(RERUN).read()
    m = re.search(r"--dx\s+(\d+(?:\.\d+)?)", text)
    dx = float(m.group(1)) if m else 25.0
    runs = parse_runs(RERUN)
    if not runs:
        sys.exit("no run could be read from %s" % RERUN)

    pat_figs = source_regex(FIGS)
    pat_phase = source_regex(PHASE)
    if pat_figs is None or pat_phase is None:
        fail("the run-name regular expression could not be read from both scripts")
    if pat_figs and pat_phase and pat_figs.pattern != pat_phase.pattern:
        notes.append("the two scripts spell the pattern differently; both are "
                     "applied to every name below")

    print("%-26s %6s %5s %9s %9s  %s"
          % ("run", "fold", "step", "span km", "in name", "checks"))
    families = {}
    for r in runs:
        name = r["name"]
        fname = name + "_wavelet.npy"
        flags = []
        for label, pat in (("figures", pat_figs), ("phase", pat_phase)):
            if pat is None:
                continue
            m = pat.match(fname)
            if not m:
                flags.append("no match in %s" % label)
                fail("%s does not match the pattern of the %s script" % (fname, label))
        m = (pat_figs or pat_phase).match(fname)
        if not m:
            print("%-26s %6s %5s %9s %9s  %s"
                  % (name, r.get("fold", "?"), r.get("step", "-"), "-", "-",
                     ", ".join(flags)))
            continue
        fam, n_name, l_name = m.group("fam"), int(m.group("n")), float(m.group("L"))
        if r["auto"]:
            span = None
        else:
            span = (r["fold"] - 1) * r["step"] * dx / 1000.0
            if n_name != r["fold"]:
                flags.append("name says fold %d, the selection gives %d"
                             % (n_name, r["fold"]))
                fail("%s: fold in the name is %d, the selection gives %d"
                     % (name, n_name, r["fold"]))
            if abs(l_name - span) > SPAN_TOL_KM:
                flags.append("name says %.1f km, the selection gives %.3f km"
                             % (l_name, span))
                fail("%s: span in the name is %.1f km, the selection gives %.3f km"
                     % (name, l_name, span))
        families.setdefault(fam, set()).add((r["fold"], r["step"]))
        print("%-26s %6d %5s %9s %9.1f  %s"
              % (name, r["fold"], "-" if r["step"] is None else r["step"],
                 "-" if span is None else "%.3f" % span, l_name,
                 ", ".join(flags) if flags else "ok"))

    # -------------------------------------------------------- the letters
    print()
    for fam in sorted(families):
        sel = families[fam]
        auto = all(step is None for _fold, step in sel)
        if len(sel) > 1 and not auto:
            fail("family %s covers more than one selection: %s"
                 % (fam, sorted(sel)))
        fold, step = sorted(sel, key=lambda t: (t[0], t[1] or 0))[0]
        want = EXPECTED.get(fam)
        if want is None:
            fail("family %s is not in the documented table" % fam)
            continue
        if auto:
            print("family %s: selected by the workflow, folds %s"
                  % (fam, sorted(f for f, _s in sel)))
            continue
        if want[0] is not None and (fold, step) != (want[0], want[1]):
            fail("family %s is fold %s step %s here and fold %s step %s in "
                 "the documentation" % (fam, fold, step, want[0], want[1]))
        print("family %s: fold %s, step %s" % (fam, fold, step))
    for fam in EXPECTED:
        if fam not in families:
            fail("family %s is documented but no run produces it" % fam)

    # ------------------------------------------- the span table of the figure
    print()
    consts = module_constants(FIGS)
    span_km = consts.get("SPAN_KM")
    if span_km is None:
        fail("SPAN_KM could not be read from %s" % os.path.basename(FIGS))
    else:
        derived = {}
        for fam, sel in families.items():
            fold, step = sorted(sel)[0]
            if step is None or fold != 100:
                continue
            derived[step] = (fold - 1) * step * dx / 1000.0
        if set(span_km) != set(derived):
            fail("SPAN_KM is keyed on steps %s, the fold-100 runs use %s"
                 % (sorted(span_km), sorted(derived)))
        for step, km in sorted(span_km.items()):
            if step in derived and abs(km - derived[step]) > SPAN_TOL_KM:
                fail("SPAN_KM[%s] is %.1f km, the selection gives %.3f km"
                     % (step, km, derived[step]))
        print("SPAN_KM %s against the selections %s"
              % (span_km, dict((k, round(v, 3)) for k, v in sorted(derived.items()))))
        colours = consts.get("C_SPAN")
        if isinstance(colours, dict) and set(colours) != set(span_km):
            fail("C_SPAN and SPAN_KM are keyed differently: %s against %s"
                 % (sorted(colours), sorted(span_km)))

    blind = blind_letters(FIGS)
    fold100 = tuple(sorted(f for f, sel in families.items()
                           if sorted(sel)[0][0] == 100 and sorted(sel)[0][1] is not None))
    if blind is None:
        notes.append("the blind-family tuple could not be read from %s"
                     % os.path.basename(FIGS))
    else:
        print("blind families in the figure %s, fold-100 runs %s"
              % (tuple(blind), fold100))
        if tuple(sorted(blind)) != fold100:
            fail("the figure plots families %s, the run script produces %s at "
                 "fold 100" % (tuple(sorted(blind)), fold100))

    # ----------------------------------------------------- shared constants
    print()
    defaults = argparse_defaults(PHASE)
    for cname, aname in SHARED:
        if cname not in consts or aname not in defaults:
            notes.append("%s / --%s not found in both scripts" % (cname, aname))
            continue
        a, b = consts[cname], defaults[aname]
        a_l = list(a) if isinstance(a, (tuple, list)) else [a]
        b_l = list(b) if isinstance(b, (tuple, list)) else [b]
        same = len(a_l) == len(b_l) and all(abs(float(x) - float(y)) <= 1e-12
                                            for x, y in zip(a_l, b_l))
        print("%-9s %-22s %s" % (cname, a, "ok" if same else "DIFFERS from --%s = %s"
                                 % (aname, b)))
        if not same:
            fail("%s is %s in the figure script and --%s defaults to %s in the "
                 "phase test" % (cname, a, aname, b))

    print()
    for note in notes:
        print("note: %s" % note)
    if errors:
        print()
        for e in errors:
            print("FAIL: %s" % e)
        return 1
    print("PASS: run names, family letters and reference constants agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
