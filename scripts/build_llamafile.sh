#!/bin/bash
set -e

# Configuration variables
IMAGE_NAME="debian:trixie"
CONTAINER_NAME="build-llamafile"
SHARED_DIR="/tmp/llamafile/build"
LOCAL_DESTINATION="bin/"

# Ensure the shared directory exists
mkdir -p $SHARED_DIR

# Run the container with the build commands
echo "Starting build process in container..."
docker run --name $CONTAINER_NAME \
  -v $SHARED_DIR:/out \
  $IMAGE_NAME \
  /bin/bash -c "
    apt-get update && \
    apt-get install -y curl git gcc make && \
    git clone https://github.com/Mozilla-Ocho/llamafile.git  && \
    curl -L -o ./unzip https://cosmo.zip/pub/cosmos/bin/unzip && \
    chmod 755 unzip && mv unzip /usr/local/bin && \
    cd llamafile && \
    make cosmocc && \
    make -j8 LLAMA_DISABLE_LOGS=1 && \
    make install PREFIX=/out
  "

# Copy the binary from the shared directory to the local destination
echo "Copying binary to $LOCAL_DESTINATION..."
cp $SHARED_DIR/bin/llamafiler $LOCAL_DESTINATION

# Clean up
echo "Cleaning up..."
docker rm $CONTAINER_NAME
# Uncomment the following line if you want to remove the shared directory after use
# rm -rf $SHARED_DIR

echo "Build complete! Binary is available at $LOCAL_DESTINATION"
