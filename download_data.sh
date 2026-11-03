#!/bin/sh
# Download B3D's labeled crops (vision.zip) and the four test videos from B3D's Google Drive
# (the IDs B3D's own download.py uses), into the layout ./run mounts at /sources:
#   b3d-vision/vision/{annotations,images}/   training crops (prep_crops.py)
#   jncN/jnc0N.mp4                            4K source videos (detect.py, track.py)
# Existing files are kept; every file is checked against data/sources.sha256.
# Usage: ./download_data.sh [DEST]   (needs gdown and unzip: pip install gdown)
set -eu
cd "$(dirname "$0")"
REPO=$PWD
DEST=${1:-${DATA:-/data/cwkt/data}/polyis-data/sources}
mkdir -p "$DEST/b3d-vision" "$DEST/jnc0" "$DEST/jnc2" "$DEST/jnc6" "$DEST/jnc7"
cd "$DEST"

# Download one Google Drive file unless it is already there.
fetch() {
    [ -e "$2" ] && { echo "keeping $2"; return; }
    gdown "$1" -O "$2.part" && mv "$2.part" "$2"
}
fetch 1v2Go30iTtbNDnOcmoSPueF4Mp93P5Lbg b3d-vision/vision.zip
fetch 1StzhvSiF4gB4VxrTW1-LgtywCnULnjp7 jnc0/jnc00.mp4
fetch 1CIeFXURvlgiXwQWvYogfO80Qfp0Njg36 jnc2/jnc02.mp4
fetch 144RZQgHDVCdwaN4NNzNGfUgz8Zn5R1Tf jnc6/jnc06.mp4
fetch 1rNtEP9AxFlGn5kENR-KN8dY_4lUk3Izq jnc7/jnc07.mp4

# Every file must match the copies the pre-labels were made from.
shasum -a 256 -c "$REPO/data/sources.sha256"
# Unpack the crops next to the zip (existing files are not overwritten).
[ -d b3d-vision/vision ] || unzip -q -n b3d-vision/vision.zip -d b3d-vision
echo "B3D sources ready in $DEST"
