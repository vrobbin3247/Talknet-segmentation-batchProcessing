import os, glob, argparse, subprocess, sys, shutil
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Batch process videos with TalkNet and extract speaking segments")

# --- INPUT MODES ---
parser.add_argument('--videoFolder', type=str, help='Folder containing videos (local mode)')
parser.add_argument('--hf_dataset', type=str, help='Hugging Face dataset name (e.g. user/drunk-sober)')
parser.add_argument('--hf_subset', type=str, default=None, help='Subset / folder inside dataset (e.g. drunk)')
parser.add_argument('--hf_token', type=str, default=None, help='HF access token (for private datasets)')

# --- OUTPUT ---
parser.add_argument('--outputFolder', type=str, default=None, help='Output folder')

# --- TalkNet params ---
parser.add_argument('--pretrainModel', type=str, default="pretrain_TalkSet.model")
parser.add_argument('--facedetScale', type=float, default=0.25)
parser.add_argument('--minTrack', type=int, default=10)
parser.add_argument('--numFailedDet', type=int, default=10)
parser.add_argument('--minFaceSize', type=int, default=1)
parser.add_argument('--cropScale', type=float, default=0.40)
parser.add_argument('--threshold', type=float, default=0.0)
parser.add_argument('--minDuration', type=float, default=0.5)
parser.add_argument('--videoExtensions', type=str, default="mp4,avi,mov,mkv")

args = parser.parse_args()

# -----------------------------
# Validate input mode
# -----------------------------
if not args.videoFolder and not args.hf_dataset:
    print("ERROR: Provide either --videoFolder OR --hf_dataset")
    sys.exit(1)

# -----------------------------
# Output folder
# -----------------------------
if args.outputFolder is None:
    base = args.videoFolder if args.videoFolder else args.hf_dataset.replace("/", "_")
    args.outputFolder = f"{base}_output"

os.makedirs(args.outputFolder, exist_ok=True)
print(f"Output folder: {args.outputFolder}")
print("=" * 60)

# -----------------------------
# Helper: process ONE video
# -----------------------------
def process_video(video_path, video_name):
    print(f"\nProcessing: {video_name}")
    print("-" * 60)

    dst = os.path.join(args.outputFolder, os.path.basename(video_path))
    if not os.path.exists(dst):
        shutil.copy2(video_path, dst)

    # Step 1: TalkNet
    cmd_demo = (
        f"python demoTalkNet.py "
        f"--videoName {video_name} "
        f"--videoFolder {args.outputFolder} "
        f"--pretrainModel {args.pretrainModel} "
        f"--facedetScale {args.facedetScale} "
        f"--minTrack {args.minTrack} "
        f"--numFailedDet {args.numFailedDet} "
        f"--minFaceSize {args.minFaceSize} "
        f"--cropScale {args.cropScale}"
    )

    if subprocess.run(cmd_demo, shell=True).returncode != 0:
        print("✗ TalkNet failed")
        return

    # Step 2: Extract segments
    cmd_extract = (
        f"python extractSpeakingSegments.py "
        f"--videoName {video_name} "
        f"--videoFolder {args.outputFolder} "
        f"--threshold {args.threshold} "
        f"--minDuration {args.minDuration}"
    )

    if subprocess.run(cmd_extract, shell=True).returncode == 0:
        out_dir = os.path.join(args.outputFolder, video_name, "speaking_segments")
        count = len(glob.glob(os.path.join(out_dir, "*.avi")))
        print(f"✓ Done | Segments: {count}")
    else:
        print("✗ Segment extraction failed")

# -----------------------------
# MODE 1: Local folder
# -----------------------------
if args.videoFolder:
    extensions = args.videoExtensions.split(',')
    videos = []
    for ext in extensions:
        videos += glob.glob(os.path.join(args.videoFolder, f"*.{ext}"))
        videos += glob.glob(os.path.join(args.videoFolder, f"*.{ext.upper()}"))

    if not videos:
        print("No videos found.")
        sys.exit(1)

    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        process_video(v, name)

# -----------------------------
# MODE 2: Hugging Face dataset
# -----------------------------
if args.hf_dataset:
    ds = load_dataset(
        args.hf_dataset,
        data_dir=args.hf_subset,
        split="train",
        streaming=True,
        token=args.hf_token
    )

    for i, sample in enumerate(ds, 1):
        video_path = sample["video"]["path"]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        process_video(video_path, video_name)

print("\n" + "=" * 60)
print("Batch processing complete!")
print(f"Results saved in: {args.outputFolder}")
print("=" * 60)