# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /backend

# Copy the current directory contents into the container at /backend
COPY . /backend

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on (Google Cloud uses 8080 by default)
EXPOSE 8080

# Command to run your Flask app with Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:8080", "backend:app"]
