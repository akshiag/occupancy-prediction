# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app/

# Set the environment variable for Flask to run in development mode
ENV FLASK_ENV=development

# Expose port 5000 for Flask application
EXPOSE 5000

# Define the command to run your application
CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]
