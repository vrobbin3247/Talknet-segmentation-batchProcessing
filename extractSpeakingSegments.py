import os, pickle, argparse, cv2, numpy, subprocess
from scipy.io import wavfile

parser = argparse.ArgumentParser(description = "Extract speaking segments from TalkNet results")
parser.add_argument('--videoName',      type=str, default="002",   help='Demo video name')
parser.add_argument('--videoFolder',    type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--threshold',      type=float, default=0.0,   help='Speaking score threshold (default: 0.0)')
parser.add_argument('--minDuration',    type=float, default=0.5,   help='Minimum duration of speaking segment in seconds')
args = parser.parse_args()

# Paths
args.savePath = os.path.join(args.videoFolder, args.videoName)
args.pycropPath = os.path.join(args.savePath, 'pycrop')
args.pyworkPath = os.path.join(args.savePath, 'pywork')
args.outputPath = os.path.join(args.savePath, 'speaking_segments')

# Create output directory
os.makedirs(args.outputPath, exist_ok=True)

# Load scores and tracks
scoresPath = os.path.join(args.pyworkPath, 'scores.pckl')
tracksPath = os.path.join(args.pyworkPath, 'tracks.pckl')

if not os.path.exists(scoresPath):
    print(f"Error: Scores file not found at {scoresPath}")
    print("Please run demoTalkNet.py first to generate scores.")
    exit(1)

if not os.path.exists(tracksPath):
    print(f"Error: Tracks file not found at {tracksPath}")
    print("Please run demoTalkNet.py first to generate tracks.")
    exit(1)

with open(scoresPath, 'rb') as f:
    scores = pickle.load(f)

with open(tracksPath, 'rb') as f:
    tracks = pickle.load(f)

print(f"Found {len(tracks)} face tracks")
print(f"Using threshold: {args.threshold}")
print(f"Minimum segment duration: {args.minDuration}s")

def find_speaking_segments(score_array, threshold, fps=25):
    """Find continuous segments where score > threshold"""
    speaking = score_array > threshold
    segments = []
    start = None
    
    for i, is_speaking in enumerate(speaking):
        if is_speaking and start is None:
            start = i
        elif not is_speaking and start is not None:
            segments.append((start, i))
            start = None
    
    # Handle case where speaking continues to the end
    if start is not None:
        segments.append((start, len(speaking)))
    
    return segments

def extract_segment(input_video, input_audio, start_frame, end_frame, output_path, fps=25):
    """Extract a segment from video and audio"""
    start_time = start_frame / fps
    end_time = end_frame / fps
    duration = end_time - start_time
    
    # Extract video segment
    temp_video = output_path + '_temp.avi'
    cmd_video = f"ffmpeg -y -i {input_video} -ss {start_time:.3f} -t {duration:.3f} -c copy {temp_video} -loglevel panic"
    subprocess.call(cmd_video, shell=True, stdout=None)
    
    # Extract audio segment
    temp_audio = output_path + '_temp.wav'
    cmd_audio = f"ffmpeg -y -i {input_audio} -ss {start_time:.3f} -t {duration:.3f} -c copy {temp_audio} -loglevel panic"
    subprocess.call(cmd_audio, shell=True, stdout=None)
    
    # Combine video and audio
    cmd_combine = f"ffmpeg -y -i {temp_video} -i {temp_audio} -c:v copy -c:a copy {output_path} -loglevel panic"
    subprocess.call(cmd_combine, shell=True, stdout=None)
    
    # Cleanup temp files
    if os.path.exists(temp_video):
        os.remove(temp_video)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)

# Process each track
total_segments = 0
for track_idx, (score, track) in enumerate(zip(scores, tracks)):
    score_array = numpy.array(score)
    
    # Find speaking segments
    segments = find_speaking_segments(score_array, args.threshold)
    
    # Filter by minimum duration
    min_frames = int(args.minDuration * 25)
    filtered_segments = [(s, e) for s, e in segments if (e - s) >= min_frames]
    
    if len(filtered_segments) == 0:
        print(f"Track {track_idx:05d}: No speaking segments found")
        continue
    
    print(f"Track {track_idx:05d}: Found {len(filtered_segments)} speaking segments")
    
    # Extract each speaking segment
    input_video = os.path.join(args.pycropPath, f'{track_idx:05d}.avi')
    input_audio = os.path.join(args.pycropPath, f'{track_idx:05d}.wav')
    
    if not os.path.exists(input_video):
        print(f"  Warning: Input video not found: {input_video}")
        continue
    
    # Check if WAV exists, if not we'll extract from AVI
    has_wav = os.path.exists(input_audio)
    
    for seg_idx, (start_frame, end_frame) in enumerate(filtered_segments):
        duration = (end_frame - start_frame) / 25.0
        output_file = os.path.join(args.outputPath, f'track_{track_idx:05d}_segment_{seg_idx:03d}.avi')
        output_wav = os.path.join(args.outputPath, f'track_{track_idx:05d}_segment_{seg_idx:03d}.wav')
        
        print(f"  Segment {seg_idx}: frames {start_frame}-{end_frame} ({duration:.2f}s)")
        
        try:
            start_time = start_frame / 25.0
            
            # Extract video segment with audio
            cmd_video = f"ffmpeg -y -i {input_video} -ss {start_time:.3f} -t {duration:.3f} -c copy {output_file} -loglevel panic"
            subprocess.call(cmd_video, shell=True, stdout=None)
            
            # Extract WAV file
            if has_wav:
                # Extract from separate WAV file if it exists
                cmd_wav = f"ffmpeg -y -i {input_audio} -ss {start_time:.3f} -t {duration:.3f} -c copy {output_wav} -loglevel panic"
            else:
                # Extract from the video file
                cmd_wav = f"ffmpeg -y -i {output_file} -vn -acodec pcm_s16le -ar 16000 -ac 1 {output_wav} -loglevel panic"
            subprocess.call(cmd_wav, shell=True, stdout=None)
            
            total_segments += 1
        except Exception as e:
            print(f"    Error extracting segment: {e}")

print(f"\n{'='*60}")
print(f"Extraction complete!")
print(f"Total speaking segments extracted: {total_segments}")
print(f"Output directory: {args.outputPath}")
print(f"{'='*60}")

# Create a summary file
summary_file = os.path.join(args.outputPath, 'summary.txt')
with open(summary_file, 'w') as f:
    f.write(f"Speaking Segments Extraction Summary\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Video: {args.videoName}\n")
    f.write(f"Threshold: {args.threshold}\n")
    f.write(f"Minimum duration: {args.minDuration}s\n")
    f.write(f"Total tracks: {len(tracks)}\n")
    f.write(f"Total segments extracted: {total_segments}\n\n")
    
    for track_idx, score in enumerate(scores):
        score_array = numpy.array(score)
        segments = find_speaking_segments(score_array, args.threshold)
        filtered_segments = [(s, e) for s, e in segments if (e - s) >= int(args.minDuration * 25)]
        
        if len(filtered_segments) > 0:
            f.write(f"\nTrack {track_idx:05d}:\n")
            for seg_idx, (start_frame, end_frame) in enumerate(filtered_segments):
                start_time = start_frame / 25.0
                end_time = end_frame / 25.0
                duration = end_time - start_time
                f.write(f"  Segment {seg_idx}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)\n")

print(f"Summary saved to: {summary_file}")
