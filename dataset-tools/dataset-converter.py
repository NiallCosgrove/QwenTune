import argparse
import json
from datasets import load_from_disk, Dataset


def convert_dataset_to_text(dataset_dir: str, output_file: str, raw: bool = False) -> None:
    """
    Convert a Hugging Face dataset (saved via save_to_disk) to a human-readable text file.

    :param dataset_dir: Directory containing the saved dataset.
    :type dataset_dir: str
    :param output_file: Path to the output text file.
    :type output_file: str
    :param raw: Whether to output raw text instead of JSONL.
    :type raw: bool
    """
    dataset = load_from_disk(dataset_dir)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            if raw:
                f.write("\n".join(str(v) for v in example.values()) + "\n\n")
            else:
                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + "\n")

    print(f"Dataset converted to {'raw text' if raw else 'JSONL'} and saved as: {output_file}")


def convert_text_to_dataset(input_file: str, output_dir: str) -> None:
    """
    Convert a JSONL text file back into a Hugging Face dataset and save it to disk.

    :param input_file: Path to the input JSONL text file.
    :type input_file: str
    :param output_dir: Directory where the new dataset will be saved.
    :type output_dir: str
    """
    examples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(output_dir)

    print(f"Dataset recreated from text file and saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face dataset to a text file (JSONL or raw) and back.\n\n"
                    "Usage Examples:\n"
                    "  Convert dataset to JSONL:\n"
                    "    python dataset_converter.py to-text --dataset_dir ./datasets/oasst1_train --output_file oasst1_train.jsonl\n"
                    "  Convert dataset to raw text:\n"
                    "    python dataset_converter.py to-text --dataset_dir ./datasets/oasst1_train --output_file oasst1_train.txt --raw\n"
                    "  Convert text back to dataset:\n"
                    "    python dataset_converter.py to-dataset --input_file oasst1_train.jsonl --output_dir ./datasets/oasst1_train_new\n",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for dataset -> text
    to_text_parser = subparsers.add_parser("to-text", help="Convert a Hugging Face dataset to a JSONL or raw text file.")
    to_text_parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the directory containing the saved dataset.")
    to_text_parser.add_argument("--output_file", type=str, required=True, help="Path to the output text file.")
    to_text_parser.add_argument("--raw", action="store_true", help="If set, outputs a raw text file instead of JSONL.")
    
    # Subparser for text -> dataset
    to_dataset_parser = subparsers.add_parser("to-dataset", help="Convert a JSONL text file back into a Hugging Face dataset.")
    to_dataset_parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL text file containing dataset records.")
    to_dataset_parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where the new dataset will be saved.")
    
    args = parser.parse_args()
    
    if args.command == "to-text":
        convert_dataset_to_text(args.dataset_dir, args.output_file, args.raw)
    elif args.command == "to-dataset":
        convert_text_to_dataset(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()

