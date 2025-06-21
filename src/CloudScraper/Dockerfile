# Use an official Python runtime as a parent image
FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port 8000 (if needed for external access)
EXPOSE 8000

# Define environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "data_scraper.py"]