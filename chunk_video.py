import cv2
import argparse

def parse_time(timestr):
    parts = timestr.split(':')
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    elif len(parts) == 1:
        h = 0
        m = 0
        s = parts[0]
    else:
        raise ValueError('Invalid time format. Use hh:mm:ss, mm:ss, or ss.')
    return h * 3600 + m * 60 + s

def chunk_video(input_file, output_file, start_time, end_time):
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx < end_frame and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved chunked video to {output_file} from {start_time}s to {end_time}s.")


def main():
    parser = argparse.ArgumentParser(description='Chunk a video file by time interval.')
    parser.add_argument('input_file', type=str, help='Input video file path')
    parser.add_argument('output_file', type=str, help='Output video file path')
    parser.add_argument('start_time', type=str, help='Start time (hh:mm:ss, mm:ss, or ss)')
    parser.add_argument('end_time', type=str, help='End time (hh:mm:ss, mm:ss, or ss)')
    args = parser.parse_args()
    start_seconds = parse_time(args.start_time)
    end_seconds = parse_time(args.end_time)
    chunk_video(args.input_file, args.output_file, start_seconds, end_seconds)

if __name__ == "__main__":
    main() 