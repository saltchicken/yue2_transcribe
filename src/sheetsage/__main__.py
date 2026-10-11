import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModel

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to ABC score using SheetSage2.")
    
    parser.add_argument("input_file", help="Path to the input audio file (e.g., audio.mp3)")
    # We use this as the base directory now
    parser.add_argument("-o", "--output-dir", default=".", help="Base output directory (default: current directory)")
    parser.add_argument("--chords", action="store_true", help="Include chord symbols (turns off melody_only)")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    base_out_dir = Path(args.output_dir)
    
    # Create the new directory: e.g., "audio-score"
    target_dir = base_out_dir / f"{input_path.stem}-score"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SheetSage2 model...")
    # Load the SheetSage2 model
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    # melody_only=True is required for YuE2 covers to omit chord symbols
    melody_only = not args.chords

    print(f"Transcribing '{args.input_file}' on {device} (melody_only={melody_only})...")
    
    # Transcribe the audio file (SheetSage dumps all its generated files here)
    result = model.transcribe(
        args.input_file,
        output_dir=str(target_dir),
        melody_only=melody_only
    )

    # Find the default score.abc and rename it to match the input filename
    default_output = target_dir / "score.abc"
    final_output = target_dir / f"{input_path.stem}.abc"

    if default_output.exists():
        if final_output.exists() and default_output != final_output:
            final_output.unlink() # Remove if the target file already exists
            
        if default_output != final_output:
            default_output.rename(final_output)
            
        print(f"Success! ABC score and accompanying files saved in: {target_dir}")
        print(f"Final ABC file: {final_output}")
    else:
        print(f"Error: Expected output {default_output} was not generated.")

if __name__ == "__main__":
    sys.exit(main())
