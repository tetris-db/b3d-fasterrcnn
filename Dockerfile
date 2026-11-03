FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# ffmpeg (with libx264) for tetris's VideoWriter / VideoCapture in viz_tracks.py.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
