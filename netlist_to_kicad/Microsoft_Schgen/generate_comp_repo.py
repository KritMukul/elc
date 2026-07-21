import json
import os
from pathlib import Path

def main():
    # Paths
    project_path = os.environ.get("PROJECT_PATH", ".")
    export_path = Path(project_path) / "export" / "organized_lib.json"
    repo_path = Path(project_path) / "modules" / "component_repository.json"

    if not export_path.exists():
        print(f"Error: {export_path} not found!")
        print("Please run `python modules/utils/kicad_scan_lib.py` first to generate it.")
        return

    print(f"Loading {export_path}...")
    with open(export_path, "r", encoding="utf-8") as f:
        org_lib = json.load(f)

    # The expected structure is { Level1: { Level2: { lib_name: [ symbol_names... ] } } }
    # We will put everything under a default "KiCad" -> "Standard" hierarchy
    repo_data = {
        "KiCad": {
            "Standard": {}
        }
    }

    lib_dict = repo_data["KiCad"]["Standard"]
    total_symbols = 0

    for lib_name, symbols in org_lib.items():
        symbol_names = []
        for sym in symbols:
            if "name" in sym:
                # Remove surrounding quotes if present (e.g. '"LM3S6911-EQC50"')
                name = sym["name"].strip('"')
                symbol_names.append(name)
        
        lib_dict[lib_name] = symbol_names
        total_symbols += len(symbol_names)

    # Save to component_repository.json
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    with open(repo_path, "w", encoding="utf-8") as f:
        json.dump(repo_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully generated {repo_path}!")
    print(f"Total libraries: {len(lib_dict)}")
    print(f"Total symbols mapped: {total_symbols}")

if __name__ == "__main__":
    main()
