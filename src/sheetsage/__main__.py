import argparse
import sys
import subprocess
import torch
import datetime
from pathlib import Path
from transformers import AutoModel

def parse_duration(duration_val):
    """Converts MM:SS, HH:MM:SS, or Excel time objects to total seconds."""
    if duration_val is None or str(duration_val).strip() == "":
        return 0.0
        
    # 1. Handle Excel's native time/datetime objects
    if isinstance(duration_val, (datetime.time, datetime.datetime)):
        # Excel famously parses input like "2:15" (2m 15s) as 2 hours, 15 minutes, 0 seconds.
        if duration_val.second == 0:
            # Shift everything down: Hours become Minutes, Minutes become Seconds
            return float(duration_val.hour * 60 + duration_val.minute)
        else:
            return float(duration_val.hour * 3600 + duration_val.minute * 60 + duration_val.second)

    # 2. Handle strings (e.g. from CSV or formatted text cells)
    duration_str = str(duration_val).strip()
    if ':' in duration_str:
        parts = duration_str.split(':')
        # If MM:SS
        if len(parts) == 2:
            return float(int(parts[0]) * 60 + float(parts[1]))
        # If HH:MM:SS
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            # If a string comes through as '00:25:00' due to Excel text formatting
            if s == 0 and h > 0:
                return float(h * 60 + m)
            return float(h * 3600 + m * 60 + s)

    # 3. Fallback if it's already just raw seconds (e.g., "25" or 25)
    return float(duration_str)

def download_audio(url, name, output_dir):
    print(f"\n==================================================")
    print(f"⬇️  Downloading: {name}")
    print(f"==================================================")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = f"{name}.%(ext)s"
    
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--cookies-from-browser", "firefox",
        "-P", str(output_dir),
        "-o", output_template,
        url
    ]
    
    # Run yt-dlp using the system installation
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: yt-dlp failed for {url}")
        return None
        
    return output_dir / f"{name}.mp3"

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe local audio or download and transcribe YouTube audio to ABC score using SheetSage2."
    )

    parser.add_argument("input_file", nargs="?",
                        help="Path to the local input audio file (e.g., audio.mp3)")
    parser.add_argument("-b", "--batch",
                        help="Path to a batch file. Supports .txt (local paths 'in|out'), .csv or .xlsx (YouTube 'URL, Name, Duration').")
    parser.add_argument("-u", "--url",
                        help="Single YouTube URL")
    parser.add_argument("-n", "--name",
                        help="Single Song Name (required if using --url)")
    parser.add_argument("-s", "--subfolder", default="",
                        help="Subfolder to save the YouTube audio/scores in (optional)")
    parser.add_argument("-o", "--output-dir", default=".",
                        help="Base output directory (default: current directory)")

    args = parser.parse_args()
    base_dir = Path(args.output_dir).resolve()

    # 1. Build the list of tasks
    # Each task is a dict containing type ("local" or "youtube"), name, output_dir, and either input_path or url.
    tasks = []

    if args.batch:
        batch_path = Path(args.batch)
        
        # --- XLSX LOGIC ---
        if batch_path.suffix.lower() == '.xlsx':
            try:
                import openpyxl
            except ImportError:
                print("Error: 'openpyxl' is required to read .xlsx files. Please run: pip install openpyxl")
                sys.exit(1)
                
            spreadsheet_name = batch_path.stem
            print(f"📂 Loading Workbook: {spreadsheet_name}")
            
            # data_only=True ensures we get calculated values, not formulas
            wb = openpyxl.load_workbook(batch_path, data_only=True)
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                batch_subfolder = Path(spreadsheet_name.strip()) / sheet_name.strip()
                target_dir = base_dir / batch_subfolder
                
                for row in sheet.iter_rows(values_only=True):
                    if not row or not row[0]:
                        continue
                        
                    url = str(row[0]).strip()
                    if not url.startswith('http'):
                        continue
                        
                    name = str(row[1]).strip() if len(row) > 1 and row[1] else "Unknown"
                    
                    if len(row) > 2 and row[2] is not None:
                        try:
                            duration = parse_duration(row[2])
                            if duration < 30:
                                print(f"⏭️ Skipping '{name}' (Tab: {sheet_name}): Duration ({duration}s) is less than 30 seconds.")
                                continue
                        except ValueError:
                            print(f"⚠️ Warning: Could not parse duration '{row[2]}' for '{name}'. Proceeding anyway.")
                            
                    tasks.append({"type": "youtube", "url": url, "name": name, "output_dir": target_dir})

        # --- CSV LOGIC ---
        elif batch_path.suffix.lower() == '.csv':
            filename_stem = batch_path.stem
            dash_count = filename_stem.count(" - ")
            
            if dash_count > 1:
                print(f"Error: Unexpected filename format. Found multiple ' - ' in '{filename_stem}.csv'. Expected '<Spreadsheet> - <GameName>.csv'.")
                sys.exit(1)
            elif dash_count == 1:
                spreadsheet, game_name = filename_stem.split(" - ")
                batch_subfolder = Path(spreadsheet.strip()) / game_name.strip()
            else:
                batch_subfolder = Path(filename_stem.strip())

            target_dir = base_dir / batch_subfolder

            with open(args.batch, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        url = parts[0]
                        name = parts[1]
                        
                        if len(parts) > 2 and parts[2]:
                            try:
                                duration = parse_duration(parts[2])
                                if duration < 30:
                                    print(f"⏭️ Skipping '{name}': Duration ({duration}s) is less than 30 seconds.")
                                    continue
                            except ValueError:
                                print(f"⚠️ Warning: Could not parse duration '{parts[2]}' for '{name}'. Proceeding anyway.")
                                
                        tasks.append({"type": "youtube", "url": url, "name": name, "output_dir": target_dir})

        # --- LOCAL TEXT BATCH LOGIC ---
        else:
            with open(args.batch, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('|')
                    if len(parts) == 2:
                        input_path = Path(parts[0].strip())
                        out_dir = Path(parts[1].strip())
                        tasks.append({
                            "type": "local",
                            "input_path": input_path,
                            "name": input_path.name,
                            "output_dir": out_dir
                        })

    elif args.input_file:
        input_path = Path(args.input_file)
        tasks.append({
            "type": "local",
            "input_path": input_path,
            "name": input_path.name,
            "output_dir": base_dir
        })
    elif args.url and args.name:
        subfolder = Path(args.subfolder) if args.subfolder else None
        target_dir = base_dir / subfolder if subfolder else base_dir
        tasks.append({
            "type": "youtube",
            "url": args.url,
            "name": args.name,
            "output_dir": target_dir
        })
    else:
        print("Error: You must provide a local input_file, a --batch file, or both --url and --name.")
        sys.exit(1)

    if not tasks:
        print("No tasks found to process.")
        sys.exit(0)

    # 2. Load the model ONCE
    print("\n==================================================")
    print("🧠 Loading SheetSage2 Model...")
    print("==================================================")
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    runs = [
        {"melody_only": True, "subfolder": "melody", "suffix": "-melody"},
        {"melody_only": False, "subfolder": "full", "suffix": "-full"}
    ]

    # 3. Interleave Downloads and Transcriptions
    for task in tasks:
        print(f"\n--- Processing: {task['name']} ---")
        
        output_dir = task["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- RESOLVE AUDIO PATH ---
        if task["type"] == "youtube":
            audio_path = download_audio(task["url"], task["name"], output_dir)
            if not audio_path or not audio_path.exists():
                print(f"Skipping transcription for {task['name']} due to download failure.")
                continue
        else:
            audio_path = task["input_path"]
            if not audio_path.exists():
                print(f"Error: Audio file not found at {audio_path}. Skipping.")
                continue

        # --- TRANSCRIBE ---
        target_dir = output_dir / f"{audio_path.stem}-score"
        target_dir.mkdir(parents=True, exist_ok=True)

        for run in runs:
            melody_only = run["melody_only"]
            subfolder = run["subfolder"]
            suffix = run["suffix"]

            run_dir = target_dir / subfolder
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"Transcribing (melody_only={melody_only}) to '{run_dir}'...")

            try:
                model.transcribe(
                    str(audio_path),
                    output_dir=str(run_dir),
                    melody_only=melody_only
                )
            except RuntimeError as e:
                # Catch RuntimeErrors from SheetSage and skip instead of crashing the batch
                print(f"⚠️ Skipping transcription run: SheetSage encountered a runtime error. Error: {e}")
                continue
            except Exception as e:
                print(f"❌ An unexpected error occurred during transcription: {e}")
                continue

            default_output = run_dir / "score.abc"
            final_output = run_dir / f"{audio_path.stem}{suffix}.abc"

            if default_output.exists():
                if final_output.exists() and default_output != final_output:
                    final_output.unlink()
                if default_output != final_output:
                    default_output.rename(final_output)
                print(f"Saved ABC file: {final_output}")
            else:
                print(f"Error: Expected output {default_output} was not generated.")

    print(f"\n✅ Success! All {len(tasks)} tasks complete.")

if __name__ == "__main__":
    sys.exit(main())
