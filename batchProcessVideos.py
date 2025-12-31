import os, glob, argparse, subprocess, sys, shutil
from datasets import load_dataset
from huggingface_hub import HfApi, upload_folder

parser = argparse.ArgumentParser(description="Batch process videos with TalkNet and extract speaking segments")

# --- INPUT MODES ---
parser.add_argument('--videoFolder', type=str, help='Folder containing videos (local mode)')
parser.add_argument('--hf_dataset', type=str, help='Hugging Face dataset name (e.g. user/drunk-sober)')
parser.add_argument('--hf_subset', type=str, default=None, help='Subset / folder inside dataset (e.g. drunk)')
parser.add_argument('--hf_token', type=str, default=None, help='HF access token (for private datasets)')

# --- OUTPUT ---
parser.add_argument('--outputFolder', type=str, default=None, help='Output folder')

# --- HUGGING FACE PUSH OPTIONS ---
parser.add_argument('--push_to_hub', action='store_true', help='Push processed segments to HuggingFace')
parser.add_argument('--hf_output_repo', type=str, default=None, help='Output HF repo name (e.g. user/processed-videos)')
parser.add_argument('--repo_type', type=str, default='dataset', choices=['dataset', 'model', 'space'], help='Type of HF repo')
parser.add_argument('--private_repo', action='store_true', help='Make output repo private')

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

def normalize_repo_id(repo_id, repo_type, username):
    if repo_type == "dataset":
        # datasets MUST have namespace
        if "/" not in repo_id:
            return f"{username}/{repo_id}"
        return repo_id
    else:
        # models/spaces can be bare
        return repo_id.split("/")[-1] if "/" in repo_id else repo_id

HF_USERNAME = "vaibhavm3247"  # hardcode or infer via whoami()

args.hf_output_repo = normalize_repo_id(
    args.hf_output_repo,
    args.repo_type,
    HF_USERNAME
)

print(f"HF repo resolved to: {args.hf_output_repo} ({args.repo_type})")

# -----------------------------
# Validate input mode
# -----------------------------
if not args.videoFolder and not args.hf_dataset:
    print("ERROR: Provide either --videoFolder OR --hf_dataset")
    sys.exit(1)

# Validate HF push requirements
if args.push_to_hub:
    if not args.hf_output_repo:
        print("ERROR: --hf_output_repo required when using --push_to_hub")
        sys.exit(1)
    if not args.hf_token:
        print("WARNING: --hf_token not provided. Will try to use cached credentials.")

# -----------------------------
# Output folder
# -----------------------------
if args.outputFolder is None:
    base = args.videoFolder if args.videoFolder else args.hf_dataset.replace("/", "_")
    args.outputFolder = f"{base}_output"

os.makedirs(args.outputFolder, exist_ok=True)
print(f"Output folder: {args.outputFolder}")
if args.push_to_hub:
    print(f"Will push to HuggingFace repo: {args.hf_output_repo}")
print("=" * 60)

# -----------------------------
# Helper: Create or get HF repo
# -----------------------------
def create_or_get_repo(repo_id, repo_type='dataset', private=False, token=None):
    """Create HuggingFace repo if it doesn't exist"""
    try:
        api = HfApi(token=token)
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"  Repo {repo_id} already exists")
        except:
            print(f"  Creating new repo: {repo_id}")
            api.create_repo(
                repo_id=repo_id,
                repo_type='dataset',
                private=private,
                exist_ok=True
            )
            print(f"  ✓ Repo created successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error with repo: {e}")
        return False

# -----------------------------
# Helper: Upload video folder to HF
# -----------------------------
def upload_video_to_hf(video_name, video_folder_path):
    """Upload entire video processing folder to HuggingFace"""
    if not os.path.exists(video_folder_path):
        print(f"  ✗ Video folder not found: {video_folder_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Uploading {video_name} to HuggingFace...")
    print(f"{'='*60}")
    
    # Create repo if doesn't exist
    if not create_or_get_repo(args.hf_output_repo, args.repo_type, args.private_repo, args.hf_token):
        return False
    
    try:
        api = HfApi(token=args.hf_token)
        
        # Upload the entire video folder
        # This will preserve the structure: VideoName/pywork, VideoName/pycrop, etc.
        print(f"  Uploading folder structure...")
        api.upload_folder(
            folder_path=video_folder_path,
            path_in_repo=video_name,
            repo_id=args.hf_output_repo,
            repo_type=args.repo_type,
        )
        
        print(f"  ✓ Successfully uploaded {video_name}")
        print(f"  📁 Structure: {args.hf_output_repo}/{video_name}/")
        
        # Count uploaded files
        subdirs = ['pywork', 'pycrop', 'speaking_segments']
        for subdir in subdirs:
            subdir_path = os.path.join(video_folder_path, subdir)
            if os.path.exists(subdir_path):
                file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                print(f"     └── {subdir}/: {file_count} files")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to upload: {e}")
        return False

# -----------------------------
# Helper: process ONE video
# -----------------------------
def process_video(video_path, video_name):
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

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
        segment_count = len(glob.glob(os.path.join(out_dir, "*.avi"))) if os.path.exists(out_dir) else 0
        print(f"✓ Processing complete | Segments: {segment_count}")
        
        # Step 3: Push to HuggingFace immediately after processing this video
        if args.push_to_hub:
            video_folder = os.path.join(args.outputFolder, video_name)
            upload_video_to_hf(video_name, video_folder)
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

    print(f"Found {len(videos)} videos to process\n")
    
    for idx, v in enumerate(videos, 1):
        name = os.path.splitext(os.path.basename(v))[0]
        print(f"\n[Video {idx}/{len(videos)}]")
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
        token=args.hf_token,
        decode=False
    )

    for i, sample in enumerate(ds, 1):
        video_path = sample["video"]["path"]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[Video {i}]")
        process_video(video_path, video_name)

print("\n" + "=" * 60)
print("Batch processing complete!")
print(f"Results saved in: {args.outputFolder}")
if args.push_to_hub:
    print(f"All videos uploaded to: https://huggingface.co/{args.hf_output_repo}")
print("=" * 60)