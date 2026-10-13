import argparse
import sys
import logging
import datetime
import subprocess
from pathlib import Path
import torch
from transformers import AutoModel

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger("sheetsage_cli")


def parse_duration(duration_val) -> float:
    """Converts MM:SS, HH:MM:SS, or Excel time objects to total seconds."""
    if duration_val is None or str(duration_val).strip() == "":
        return 0.0
        
    if isinstance(duration_val, (datetime.time, datetime.datetime)):
        if duration_val.second == 0:
            return float(duration_val.hour * 60 + duration_val.minute)
        return float(duration_val.hour * 3600 + duration_val.minute * 60 + duration_val.second)

    duration_str = str(duration_val).strip()
    if ':' in duration_str:
        parts = duration_str.split(':')
        if len(parts) == 2:
            return float(int(parts[0]) * 60 + float(parts[1]))
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            if s == 0 and h > 0:
                return float(h * 60 + m)
            return float(h * 3600 + m * 60 + s)

    return float(duration_str)


def download_audio(url: str, name: str, output_dir: Path):
    """Downloads audio using the original yt-dlp subprocess command, with the required JS solver flag."""
    logger.info(f"\n==================================================")
    logger.info(f"⬇️  Downloading: {name}")
    logger.info(f"==================================================")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = f"{name}.%(ext)s"
    
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--cookies-from-browser", "firefox",
        "--remote-components", "ejs:github",  # <--- Added to solve the [jsc:deno] error
        "-P", str(output_dir),
        "-o", output_template,
        url
    ]
    
    # Run yt-dlp using the system installation
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Error: yt-dlp failed for {url}")
        return None
        
    return output_dir / f"{name}.mp3"


def parse_excel_batch(batch_path: Path, base_dir: Path, min_duration: float) -> list:
    try:
        import openpyxl
    except ImportError:
        logger.error("Error: 'openpyxl' is required to read .xlsx files. Run: pip install openpyxl")
        sys.exit(1)
        
    tasks = []
    logger.info(f"📂 Loading Workbook: {batch_path.stem}")
    wb = openpyxl.load_workbook(batch_path, data_only=True)
    
    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        target_dir = base_dir / batch_path.stem.strip() / sheet_name.strip()
        
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
                    if duration < min_duration:
                        logger.info(f"⏭️ Skipping '{name}': Duration ({duration}s) < {min_duration}s.")
                        continue
                except ValueError:
                    logger.warning(f"⚠️ Warning: Could not parse duration '{row[2]}' for '{name}'.")
                    
            tasks.append({"type": "youtube", "url": url, "name": name, "output_dir": target_dir})
    return tasks


def parse_csv_batch(batch_path: Path, base_dir: Path, min_duration: float) -> list:
    tasks = []
    filename_stem = batch_path.stem
    dash_count = filename_stem.count(" - ")
    
    if dash_count > 1:
        logger.error(f"Error: Found multiple ' - ' in '{filename_stem}.csv'. Expected '<Spreadsheet> - <GameName>.csv'.")
        sys.exit(1)
    elif dash_count == 1:
        spreadsheet, game_name = filename_stem.split(" - ")
        target_dir = base_dir / spreadsheet.strip() / game_name.strip()
    else:
        target_dir = base_dir / filename_stem.strip()

    with open(batch_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                url, name = parts[0], parts[1]
                if len(parts) > 2 and parts[2]:
                    try:
                        duration = parse_duration(parts[2])
                        if duration < min_duration:
                            logger.info(f"⏭️ Skipping '{name}': Duration ({duration}s) < {min_duration}s.")
                            continue
                    except ValueError:
                        logger.warning(f"⚠️ Warning: Could not parse duration '{parts[2]}' for '{name}'.")
                        
                tasks.append({"type": "youtube", "url": url, "name": name, "output_dir": target_dir})
    return tasks


def parse_text_batch(batch_path: Path) -> list:
    tasks = []
    with open(batch_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) == 2:
                input_path = Path(parts[0].strip())
                tasks.append({
                    "type": "local",
                    "input_path": input_path,
                    "name": input_path.name,
                    "output_dir": Path(parts[1].strip())
                })
    return tasks


def process_audio_file(model, audio_path: Path, target_dir: Path):
    """Handles the transcription of a single audio file via SheetSage2."""
    runs = [
        {"melody_only": True, "subfolder": "melody", "suffix": "-melody"},
        {"melody_only": False, "subfolder": "full", "suffix": "-full"}
    ]
    
    for run in runs:
        run_dir = target_dir / run["subfolder"]
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Transcribing (melody_only={run['melody_only']}) to '{run_dir}'...")

        try:
            model.transcribe(
                str(audio_path),
                output_dir=str(run_dir),
                melody_only=run["melody_only"]
            )
        except RuntimeError as e:
            logger.warning(f"⚠️ SheetSage runtime error. Skipping run. Error: {e}")
            continue
        except Exception as e:
            logger.error(f"❌ Unexpected transcription error: {e}")
            continue

        default_output = run_dir / "score.abc"
        final_output = run_dir / f"{audio_path.stem}{run['suffix']}.abc"

        if default_output.exists():
            if final_output.exists() and default_output != final_output:
                final_output.unlink()
            if default_output != final_output:
                default_output.rename(final_output)
            logger.info(f"✅ Saved ABC file: {final_output}")
        else:
            logger.error(f"❌ Error: Expected output {default_output} was not generated.")


def main():
    parser = argparse.ArgumentParser(description="Transcribe local or YouTube audio to ABC score using SheetSage2.")
    parser.add_argument("input_file", nargs="?", help="Path to local input audio (e.g., audio.mp3)")
    parser.add_argument("-b", "--batch", help="Batch file (.txt, .csv, .xlsx)")
    parser.add_argument("-u", "--url", help="Single YouTube URL")
    parser.add_argument("-n", "--name", help="Single Song Name (required with --url)")
    parser.add_argument("-s", "--subfolder", default="", help="Subfolder to save YouTube audio/scores")
    parser.add_argument("-o", "--output-dir", default=".", help="Base output directory")
    parser.add_argument("--min-duration", type=float, default=30.0, help="Skip YouTube videos shorter than this (seconds)")
    parser.add_argument("--delete-audio", action="store_true", help="Delete downloaded .mp3 after transcription")

    args = parser.parse_args()
    base_dir = Path(args.output_dir).resolve()
    tasks = []

    # 1. Build Task List
    if args.batch:
        batch_path = Path(args.batch)
        if batch_path.suffix.lower() == '.xlsx':
            tasks = parse_excel_batch(batch_path, base_dir, args.min_duration)
        elif batch_path.suffix.lower() == '.csv':
            tasks = parse_csv_batch(batch_path, base_dir, args.min_duration)
        else:
            tasks = parse_text_batch(batch_path)
    elif args.input_file:
        input_path = Path(args.input_file)
        tasks.append({"type": "local", "input_path": input_path, "name": input_path.name, "output_dir": base_dir})
    elif args.url and args.name:
        subfolder = Path(args.subfolder) if args.subfolder else None
        tasks.append({"type": "youtube", "url": args.url, "name": args.name, "output_dir": base_dir / subfolder if subfolder else base_dir})
    else:
        logger.error("Error: Provide a local input_file, a --batch file, or both --url and --name.")
        sys.exit(1)

    if not tasks:
        logger.info("No tasks found to process.")
        sys.exit(0)

    # 2. Load Model
    logger.info("\n==================================================")
    logger.info("🧠 Loading SheetSage2 Model...")
    logger.info("==================================================")
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    # 3. Process Tasks
    for task in tasks:
        logger.info(f"\n--- Processing: {task['name']} ---")
        output_dir = task["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve Audio
        if task["type"] == "youtube":
            audio_path = download_audio(task["url"], task["name"], output_dir)
            if not audio_path or not audio_path.exists():
                logger.warning(f"Skipping transcription for {task['name']} due to download failure.")
                continue
        else:
            audio_path = task["input_path"]
            if not audio_path.exists():
                logger.error(f"Error: Audio file not found at {audio_path}. Skipping.")
                continue

        # Transcribe
        target_dir = output_dir / f"{audio_path.stem}-score"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        process_audio_file(model, audio_path, target_dir)
        
        # Cleanup
        if args.delete_audio and task["type"] == "youtube" and audio_path.exists():
            audio_path.unlink()
            logger.info(f"🗑️ Deleted audio file: {audio_path.name}")

    logger.info(f"\n🎉 Success! All {len(tasks)} tasks complete.")

if __name__ == "__main__":
    sys.exit(main())
