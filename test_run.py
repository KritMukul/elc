# test_run.py
import json
from spice_to_graph.parser import SpiceParser

def main():
    # 1. Initialize the parser
    parser = SpiceParser()
    
    # 2. Define the path to your test file
    test_file = "test.spice"
    
    try:
        # 3. Parse the file
        print(f"Parsing {test_file}...\n")
        graph_data = parser.parse(test_file)
        
        # 4. Print the output in a clean, readable JSON format
        print("Successfully generated Graph JSON:")
        print(json.dumps(graph_data, indent=4))
        
    except FileNotFoundError:
        print(f"Error: Could not find '{test_file}'. Make sure it's in the same directory.")
    except Exception as e:
        print(f"An error occurred during parsing: {e}")

if __name__ == "__main__":
    main()