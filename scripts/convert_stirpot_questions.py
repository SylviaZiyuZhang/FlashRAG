import json
import argparse

def convert_dataset(input_file, output_file):
    """
    Convert dataset from original stirpot format (JSON array)
    to JSONL format expected by FlashRAG Dataset class.
    """
    with open(input_file, "r") as f:
        data = json.load(f)   # input is a big JSON array

    with open(output_file, "w") as f:
        for item in data:
            new_item = {
                "question": item["question"],
                "judgement": "same",  # placeholder label
                "golden_answers": (
                    [item["answer"]] if isinstance(item["answer"], str) else item["answer"]
                ),
            }
            f.write(json.dumps(new_item) + "\n")   # JSONL format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert stirpot JSON dataset into JSONL format for FlashRAG"
    )
    parser.add_argument("input_file", help="Path to input JSON file (e.g. stirpot-300.json)")
    parser.add_argument("output_file", help="Path to output JSONL file (e.g. test.json)")

    args = parser.parse_args()

    convert_dataset(args.input_file, args.output_file)
    print(f"Saved to {args.output_file}")
