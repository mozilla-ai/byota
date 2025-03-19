# Create container
FROM python:3.11-slim AS out

RUN apt-get update && \
    apt-get install -y wget git

# Create a non-root user
RUN addgroup --gid 1000 user && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" user

# Set working directory
WORKDIR /home/user

# Download default embedding model
RUN wget https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.F16.gguf

# Copy the repo's code
COPY . /home/user/byota
## Clone the repo's code - when the repo is public
#RUN git clone https://github.com/mozilla-ai/byota.git && \

RUN chown -R user:user /home/user && \
    chmod +x /home/user/byota/entrypoint.sh && \
    pip install -r /home/user/byota/requirements.txt

ENV PATH="/home/user/:${PATH}"

# Switch to user
USER user

# Expose marimo's port
EXPOSE 2718

# Set entrypoint
ENTRYPOINT ["/home/user/byota/entrypoint.sh"]
CMD ["notebook.py"]
