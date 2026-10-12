import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModel

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to ABC score using SheetSage2.")

    parser.add_argument("input_file", nargs="?",
                        help="Path to the input audio file (e.g., audio.mp3)")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Base output directory (default: current directory)")
    parser.add_argument(
        "-b",
        "--batch",
        help="Path to a text file containing 'input_path|output_dir' on each line")

    args = parser.parse_args()

    # 1. Build the list of tasks
    tasks = []
    if args.batch:
        with open(args.batch, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('|')
                if len(parts) == 2:
                    tasks.append((Path(parts[0].strip()), Path(parts[1].strip())))
    elif args.input_file:
        tasks.append((Path(args.input_file), Path(args.output_dir)))
    else:
        print("Error: You must provide either an input_file or a --batch file.")
        sys.exit(1)

    if not tasks:
        print("No tasks found to process.")
        sys.exit(0)

    # 2. Load the model ONCE
    print("Loading SheetSage2 model...")
    model = AutoModel.from_pretrained("m-a-p/SheetSage2",
                                      trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    runs = [{
        "melody_only": True,
        "subfolder": "melody",
        "suffix": "-melody"
    }, {
        "melody_only": False,
        "subfolder": "full",
        "suffix": "-full"
    }]

    # 3. Loop through all tasks using the already-loaded model
    for input_path, base_out_dir in tasks:
        print(f"\n--- Processing: {input_path.name} ---")
        
        if not input_path.exists():
            print(f"Error: Audio file not found at {input_path}. Skipping.")
            continue

        target_dir = base_out_dir / f"{input_path.stem}-score"
        target_dir.mkdir(parents=True, exist_ok=True)

        for run in runs:
            melody_only = run["melody_only"]
            subfolder = run["subfolder"]
            suffix = run["suffix"]

            run_dir = target_dir / subfolder
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"Transcribing (melody_only={melody_only}) to '{run_dir}'...")

            model.transcribe(str(input_path),
                             output_dir=str(run_dir),
                             melody_only=melody_only)

            default_output = run_dir / "score.abc"
            final_output = run_dir / f"{input_path.stem}{suffix}.abc"

            if default_output.exists():
                if final_output.exists() and default_output != final_output:
                    final_output.unlink()
                if default_output != final_output:
                    default_output.rename(final_output)
                print(f"Saved ABC file: {final_output}")
            else:
                print(f"Error: Expected output {default_output} was not generated.")

    print(f"\nSuccess! All {len(tasks)} batch tasks complete.")

if __name__ == "__main__":
    sys.exit(main())
