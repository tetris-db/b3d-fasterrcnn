# The detection, training, and tracking environment.
# Based on B3D's Dockerfile (github.com/b3d-project/b3d at 0680cd6), with package versions pinned to the
# image that produced the pre-labels; B3D's Jupyter, scikit-image, filterpy, and detectron2 source clone
# are dropped because nothing here uses them.
# Build: docker build -t b3d-fasterrcnn .
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y build-essential git ffmpeg libsm6 libxext6 fonts-freefont-ttf
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
# opencv as in the original image; wandb/tensorboard for train_frcnn.py; Cython to build ocsort/.
RUN pip install opencv-python==5.0.0.93 wandb==0.18.7 tensorboard==2.11.2 Cython==3.0.12
