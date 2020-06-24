import os
import argparse
import sys
from pathlib import Path
from itertools import chain
import filecmp
import shutil

start_symbol = "!!"

def get_location(originpath, text):
    lines = text.split("\n")
    res = []
    flag = True
    for line in lines:
        if not flag:
            if line.startswith(start_symbol + " location"):
                raise RuntimeError("location command shoule be at the start of a file.")
            continue
        if not line.startswith(start_symbol + " location"):
            flag = False
            continue

        if line.startswith(start_symbol + " locationauto"):
            res.append(("auto", Path("../source/" + str(originpath))))
        else:
            _, _, key, path = line.split()
            res.append((key, Path(path)))

    if not res:
        res.append(("auto", Path("../source/" + str(originpath))))
    return res

def render(text, macros):
    lines = text.split("\n")
    res = []

    in_flag = [True]
    for line in lines:
        if not line.startswith(start_symbol + " "):
            if in_flag[-1]:
                res.append(line)
            continue
        args = line.split()
        if len(args) < 2:
            raise RuntimeError("Unknown tags when parsing %s" % line)
        if args[1] == "location":
            continue
        elif args[1] == "ifdef":
            in_flag.append((args[2] in macros))
        elif args[1] == "ifndef":
            in_flag.append(not (args[2] in macros))
        elif args[1] == "endif":
            in_flag.pop()
        elif args[1] == "include":
            if in_flag[-1]:
                res.extend(render(open(args[2], 'r', encoding='utf-8').read(), macros))
        else:
            raise RuntimeError("Unknown tags %s when parsing %s" % (args[1], line))
    return res

def check(text, path, filename):

    realtext = open(path, 'r', encoding='utf-8').read().split("\n")
    if realtext[-1] != "" or len(text) + 1 != len(realtext):
        print("Different line numbers")
        print("\t%s: %d" % (filename, len(text)))
        print("\t%s: %d" % (path, len(realtext) - 1))
        return ValueError("It seems docs [%s] is not synced with metadocs [%s]. Please change metadocs and then run update_docs !" % (path, filename))
    for i, (t, rt) in enumerate(zip(text, realtext)):
        if t != rt:
            print("Unsynced at Line %d:" % i)
            print("\t%s: %s" % (filename, t))
            print("\t%s: %s" % (path, rt))
            return ValueError("It seems docs [%s] is not synced with metadocs [%s]. Please change metadocs and then run update_docs !" % (path, filename))
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Preprocess all the meta files(.md, .rst) in CWD.
    Supported commands in meta files:
    !! location MACRO PATH: generate at path with defining a macro named MACRO. Should at the start of file.
    !! locationauto: generate at the default path ("../source/" + path relative to CWD) with defining a macro named auto.
        Should at the start of file. This is default if no `location` command is used.
    !! ifdef MACRO: the following lines will be shown if MACRO is defined (note this cannot apply to include)
    !! ifndef MACRO: the following lines will be shown if MACRO is defined (note this cannot apply to include)
    !! endif MECRO: the above lines will be shown if MACRO is defined (note it cannot be applied to include)
    !! include PATH: include a file (relative to CWD).
''')
    parser.add_argument('--check', action='store_true', dest='check', help="If the file is already existed, "
        "just check whether it is consistent with meta files.")
    parser.add_argument('--only_source', action='store_true', help="Only check/generate filed in ../source/")
    parser.add_argument('-D', '--define', nargs='*')
    cargs = parser.parse_args()

    defined_macros = cargs.define if cargs.define else []

    for originpath in chain(Path(".").glob('**/*')):
        if originpath.is_dir():
            continue
        print(str(originpath))
        if originpath.suffix not in [".rst", ".md"]:
            path = Path("../source/" + str(originpath))
            if path.is_file() and cargs.check:
                if not filecmp.cmp(str(path), str(originpath), shallow=False):
                    raise ValueError("It seems data [%s] is not sync with [%s]. Please update docs." % (str(path), str(originpath)))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(originpath), str(path))
            continue

        file = open(str(originpath), "r", encoding='utf-8').read()

        locations = get_location(originpath, file)
        for key, path in locations:
            if cargs.only_source:
                try:
                    path.relative_to("../source/")
                except:
                    continue
            path.parent.mkdir(parents=True, exist_ok=True)
            text = render(file, [key] + defined_macros)

            if path.is_file():
                err = check(text, path, originpath)
            else:
                err = True

            if path.is_file() and cargs.check:
                if err:
                    raise err
            elif err:
                open(path, 'w', encoding='utf-8').write("\n".join(text) + "\n")
                print("render %s with %s" % (path, [key] + defined_macros))
