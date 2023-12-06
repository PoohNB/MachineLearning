# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /CAR_PRICE

# Copy the requirements file into the container
COPY requirements.txt .

RUN pip3 install nbconvert
RUN apt update
RUN apt install -y pandoc
RUN apt install -y texlive-xetex 
RUN apt install -y texlive-fonts-recommended 
RUN apt install -y texlive-plain-generic
RUN rm -rf /var/cache/apt/archives /var/lib/apt/lists/*.

# Install required packages using pip
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code into the container
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
