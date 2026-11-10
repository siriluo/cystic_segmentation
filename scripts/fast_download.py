import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://huggingface.co/datasets/minwoosun/CholecSeg8k/resolve/main/data/CholecSeg8k.zip"
TARGET = "data/CholecSeg8k.zip"
NUM_THREADS = 16

def get_final_url_and_size(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        final_url = resp.geturl()
        total_size = int(resp.headers.get("Content-Length", 0))
    return final_url, total_size

def download_chunk(final_url, start, end, chunk_idx, temp_dir):
    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx:03d}.part")
    expected_len = end - start + 1
    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) == expected_len:
        print(f"Chunk {chunk_idx:03d} already complete.")
        return chunk_idx

    headers = {"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"}
    req = urllib.request.Request(final_url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(chunk_path, "wb") as f:
        while True:
            buf = resp.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
    print(f"Downloaded chunk {chunk_idx:03d}.")
    return chunk_idx

def main():
    os.makedirs("data", exist_ok=True)
    temp_dir = "data/chunks_tmp"
    os.makedirs(temp_dir, exist_ok=True)
    print("Fetching direct URL and file size...")
    final_url, total_size = get_final_url_and_size(URL)
    print(f"Total size: {total_size / (1024**3):.2f} GB ({total_size} bytes)")
    
    chunk_size = (total_size + NUM_THREADS - 1) // NUM_THREADS
    ranges = []
    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = min((i + 1) * chunk_size - 1, total_size - 1)
        ranges.append((start, end))
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(download_chunk, final_url, start, end, idx, temp_dir)
            for idx, (start, end) in enumerate(ranges)
        ]
        for f in futures:
            f.result()
    
    print("Combining chunks into final file...")
    with open(TARGET, "wb") as out_f:
        for idx in range(NUM_THREADS):
            chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.part")
            with open(chunk_path, "rb") as in_f:
                while True:
                    b = in_f.read(1024 * 1024 * 10)
                    if not b:
                        break
                    out_f.write(b)
            os.remove(chunk_path)
    os.rmdir(temp_dir)
    elapsed = time.time() - start_time
    print(f"Complete! File saved to {TARGET} ({os.path.getsize(TARGET)} bytes)")

if __name__ == "__main__":
    main()
