# Use debian trixie for gcc13
FROM debian:trixie AS builder

# Set work directory
WORKDIR /download

# Configure build container and build llamafile
RUN mkdir out && \
    apt-get update && \
    apt-get install -y curl git gcc make && \
    git clone https://github.com/Mozilla-Ocho/llamafile.git  && \
    curl -L -o ./unzip https://cosmo.zip/pub/cosmos/bin/unzip && \
    chmod 755 unzip && mv unzip /usr/local/bin && \
    cd llamafile && \
    make cosmocc && \
    make -j8 LLAMA_DISABLE_LOGS=1 && \
    make install PREFIX=/download/out


# Create container
FROM python:3.11-slim AS out

RUN apt-get update && \
    apt-get install -y wget git

# Create a non-root user
RUN addgroup --gid 1000 user && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" user

# Set working directory
WORKDIR /home/user

# Copy llamafile and man pages
COPY --from=builder /download/out/bin /usr/local/bin
COPY --from=builder /download/out/share /usr/local/share/man

# Copy the repo's code
COPY . /home/user/byota
## Clone the repo's code - when the repo is public
#RUN git clone https://github.com/mozilla-ai/byota.git && \
RUN chown -R user:user /home/user/byota && \
    chmod +x /home/user/byota/entrypoint.sh && \
    pip install -r /home/user/byota/requirements.txt

ENV PATH="/home/user/:${PATH}"

# Switch to user
USER user

# Download default embedding model
RUN wget https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.F16.gguf

# Expose 8080 port
EXPOSE 8080 2718

# Set entrypoint
ENTRYPOINT ["/home/user/byota/entrypoint.sh"]

