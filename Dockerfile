# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - portaudio19-dev is required for PyAudio
# - libgl1-mesa-glx is often required for tkinter/customtkinter graphics
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# --- Running the GUI in Docker ---
# This application requires a GUI, which Docker containers do not have by default.
# To run this container and see the GUI, you would need to forward your host's
# X11 display. This is an advanced technique.
#
# Example for Linux hosts:
# 1. Build the image: docker build -t dm-command-center .
# 2. Allow local connections to X server: xhost +
# 3. Run the container:
#    docker run -it --rm \
#           -e DISPLAY=$DISPLAY \
#           -v /tmp/.X11-unix:/tmp/.X11-unix \
#           dm-command-center
#
# For Windows/Mac, this requires installing and running an X server like XQuartz or VcXsrv.

# The default command will attempt to run the application.
CMD ["python", "app.py"]
