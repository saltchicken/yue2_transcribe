import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to ABC score using SheetSage2.")

    parser.add_argument("input_file",
                        help="Path to the input audio file (e.g., audio.mp3)")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Base output directory (default: current directory)")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    base_out_dir = Path(args.output_dir)

    # Create the new base directory: e.g., "audio-score"
    target_dir = base_out_dir / f"{input_path.stem}-score"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SheetSage2 model...")
    # Load the SheetSage2 model
    model = AutoModel.from_pretrained("m-a-p/SheetSage2",
                                      trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    # Define the two transcription runs and their respective subfolders
    runs = [{
        "melody_only": True,
        "subfolder": "melody",
        "suffix": "-melody"
    }, {
        "melody_only": False,
        "subfolder": "full",
        "suffix": "-full"
    }]

    for run in runs:
        melody_only = run["melody_only"]
        subfolder = run["subfolder"]
        suffix = run["suffix"]

        # Create a dedicated subfolder for this run to avoid overwriting files
        run_dir = target_dir / subfolder
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Transcribing '{args.input_file}' on {device} (melody_only={melody_only}) to '{run_dir}'..."
        )

        # Transcribe the audio file into the subfolder
        result = model.transcribe(args.input_file,
                                  output_dir=str(run_dir),
                                  melody_only=melody_only)

        # Find the default score.abc in the subfolder and rename it with the suffix
        default_output = run_dir / "score.abc"
        final_output = run_dir / f"{input_path.stem}{suffix}.abc"

        if default_output.exists():
            if final_output.exists() and default_output != final_output:
                final_output.unlink(
                )  # Remove if the target file already exists

            if default_output != final_output:
                default_output.rename(final_output)

            print(f"Saved ABC file: {final_output}")
        else:
            print(f"Error: Expected output {default_output} was not generated.")

    print(
        f"Success! ABC scores and accompanying files saved in subfolders under: {target_dir}"
    )


if __name__ == "__main__":
    sys.exit(main())
