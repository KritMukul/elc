import sys
from modules.utils.kicad_add_symbol import parse_sexp, format_sexp
from init_project import make_kicad_sch

def main():
    sch = make_kicad_sch()
    parsed = parse_sexp(sch)
    formatted = format_sexp(parsed)
    print("FORMATTED SCHEMATIC:")
    print(formatted)

if __name__ == "__main__":
    main()
