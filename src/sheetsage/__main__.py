import argparse
import sys
import torch
from transformers import AutoModel

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to ABC score using SheetSage2.")
    
    # Make the input file a required positional argument, but you can default it if you prefer
    parser.add_argument("input_file", help="Path to the input audio file (e.g., audio.mp3)")
    parser.add_argument("-o", "--output-dir", default="cover-score", help="Output directory for the score (default: cover-score)")
    parser.add_argument("--chords", action="store_true", help="Include chord symbols (turns off melody_only)")

    args = parser.parse_args()

    print("Loading SheetSage2 model...")
    # Load the SheetSage2 model
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    # melody_only=True is required for YuE2 covers to omit chord symbols
    # We set melody_only to True UNLESS the user passes the --chords flag
    melody_only = not args.chords

    print(f"Transcribing '{args.input_file}' on {device} (melody_only={melody_only})...")
    
    # Transcribe the audio file
    result = model.transcribe(
        args.input_file,
        output_dir=args.output_dir,
        melody_only=melody_only
    )

    print(f"Success! ABC score saved to {args.output_dir}/score.abc")

if __name__ == "__main__":
    sys.exit(main())
