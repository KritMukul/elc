import json
import os

def verify_dataset(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Could not find '{filepath}'")
        return

    print(f"1. Verifying JSON syntax for '{filepath}'...")
    
    # Check 1: Raw Syntax Validation
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print("✅ JSON syntax is perfectly valid. No structural text errors.")
    except json.JSONDecodeError as e:
        print("\n❌ CRITICAL SYNTAX ERROR FOUND")
        print(f"Message: {e.msg}")
        print(f"Line:    {e.lineno}")
        print(f"Column:  {e.colno}")
        print("Tip: Use a command-line tool like 'sed' or 'head' to inspect this specific line.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Check 2: Graph Schema Validation
    print(f"\n2. Verifying graph schema across {len(dataset)} circuits...")
    
    if not isinstance(dataset, list):
        print("❌ Error: The root JSON structure must be a list (array) of circuits.")
        return

    schema_errors = 0
    for i, circuit in enumerate(dataset):
        # Check required keys
        if "nets" not in circuit or "devices" not in circuit:
            print(f"❌ Schema Error at index {i}: Missing 'nets' or 'devices' key.")
            print(f"   Source: {circuit.get('source_dataset', 'Unknown')}, File: {circuit.get('source_file', 'Unknown')}")
            schema_errors += 1
            continue
            
        # Check data types
        if not isinstance(circuit["nets"], list) or not isinstance(circuit["devices"], list):
            print(f"❌ Schema Error at index {i}: 'nets' or 'devices' must be lists.")
            schema_errors += 1
            continue

        # Check internal device structure
        for j, device in enumerate(circuit["devices"]):
            if not all(key in device for key in ("name", "type", "pins")):
                print(f"❌ Schema Error at index {i}, device {j}: Missing required device keys (name, type, pins).")
                schema_errors += 1
                break

        # Stop spamming the terminal if there are too many errors
        if schema_errors > 20:
            print("\n⚠️ Too many schema errors found. Halting validation.")
            break

    if schema_errors == 0:
        print("✅ All graph schemas are valid and structurally sound.")
    else:
        print(f"\nFound {schema_errors} schema errors.")

if __name__ == "__main__":
    TARGET_FILE = "master_dataset.json"
    verify_dataset(TARGET_FILE)