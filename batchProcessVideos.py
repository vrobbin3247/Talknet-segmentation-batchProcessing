import os, glob, argparse, subprocess, sys

parser = argparse.ArgumentParser(description = "Batch process videos with TalkNet and extract speaking segments")
parser.add_argument('--videoFolder',        type=str, required=True,  help='Folder containing videos')
parser.add_argument('--outputFolder',       type=str, default=None,   help='Output folder (default: videoFolder_output)')
parser.add_argument('--pretrainModel',      type=str, default="pretrain_TalkSet.model", help='Path for the pretrained TalkNet model')
parser.add_argument('--facedetScale',       type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--minTrack',           type=int,   default=10,   help='Min frames for each track')
parser.add_argument('--numFailedDet',       type=int,   default=10,   help='Missed detections allowed')
parser.add_argument('--minFaceSize',        type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',          type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--threshold',          type=float, default=0.0,  help='Speaking score threshold')
parser.add_argument('--minDuration',        type=float, default=0.5,  help='Min speaking segment duration (seconds)')
parser.add_argument('--videoExtensions',    type=str, default="mp4,avi,mov,mkv", help='Video file extensions (comma separated)')
args = parser.parse_args()

# Set output folder
if args.outputFolder is None:
    folder_basename = os.path.basename(os.path.abspath(args.videoFolder))
    args.outputFolder = os.path.join(os.path.dirname(os.path.abspath(args.videoFolder)), f"{folder_basename}_output")

# Create output folder
os.makedirs(args.outputFolder, exist_ok=True)
print(f"Output folder: {args.outputFolder}")
print("="*60)

# Get all video files
extensions = args.videoExtensions.split(',')
video_files = []
for ext in extensions:
    video_files.extend(glob.glob(os.path.join(args.videoFolder, f'*.{ext}')))
    video_files.extend(glob.glob(os.path.join(args.videoFolder, f'*.{ext.upper()}')))

if len(video_files) == 0:
    print(f"No video files found in {args.videoFolder}")
    print(f"Looking for extensions: {args.videoExtensions}")
    sys.exit(1)

print(f"Found {len(video_files)} video(s) to process")
print("="*60)

for idx, video_path in enumerate(video_files, 1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n[{idx}/{len(video_files)}] Processing: {video_name}")
    print("-"*60)
    
    # Copy video to output folder
    output_video_path = os.path.join(args.outputFolder, os.path.basename(video_path))
    if not os.path.exists(output_video_path):
        import shutil
        shutil.copy2(video_path, output_video_path)
        print(f"Copied video to output folder")
    
    # Step 1: Run demoTalkNet
    print("Step 1/2: Running TalkNet detection...")
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
    
    result = subprocess.run(cmd_demo, shell=True)
    
    if result.returncode != 0:
        print(f"Error processing {video_name}, skipping...")
        continue
    
    # Step 2: Extract speaking segments
    print("\nStep 2/2: Extracting speaking segments...")
    cmd_extract = (
        f"python extractSpeakingSegments.py "
        f"--videoName {video_name} "
        f"--videoFolder {args.outputFolder} "
        f"--threshold {args.threshold} "
        f"--minDuration {args.minDuration}"
    )
    
    result = subprocess.run(cmd_extract, shell=True)
    
    if result.returncode == 0:
        print(f"✓ Completed: {video_name}")
        output_path = os.path.join(args.outputFolder, video_name, 'speaking_segments')
        segment_count = len(glob.glob(os.path.join(output_path, '*.avi')))
        print(f"  Extracted {segment_count} speaking segments")
        print(f"  Output: {output_path}")
    else:
        print(f"✗ Failed to extract segments for {video_name}")

print("\n" + "="*60)
print("Batch processing complete!")
print(f"All results saved in: {args.outputFolder}")
print("="*60)
