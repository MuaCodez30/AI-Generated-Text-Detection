"""
Merge multiple JSON data files into a single file.
Useful for combining multiple processed datasets.
"""
import json
import sys
from pathlib import Path


def merge_json_files(input_files: list, output_path: str):
    """
    Merge multiple JSON files into one.
    
    Args:
        input_files: List of paths to JSON files to merge
        output_path: Path to save merged data
    """
    combined_data = []
    
    for file_path in input_files:
        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                combined_data.append(data)
        print(f"  Added {len(data) if isinstance(data, list) else 1} items")
    
    # Save merged data
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Merged data saved to {output_path}")
    print(f"  Total items: {len(combined_data)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/merge_data.py <input1.json> <input2.json> ... <output.json>")
        print("\nExample:")
        print("  python scripts/merge_data.py data/processed/ai_clean.json data/processed/ai_clean_extra.json data/processed/ai_merged.json")
        sys.exit(1)
    
    input_files = sys.argv[1:-1]
    output_path = sys.argv[-1]
    
    merge_json_files(input_files, output_path)
