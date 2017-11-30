#FROM python:3.4-alpine
#
## Update packages
##RUN apt-get update -y
#
#
## Bundle app source
## ADD . /src
#
## Expose - selects the port for the container
## EXPOSE  5000
#
## Run
## ENV PYTHONPATH  $PYTHONPATH:/src/info:/src/
## WORKDIR /src/
##CMD ["python", "run.py"]
#ADD . /code
#WORKDIR /code
#RUN pip install -r requirements.txt
#CMD ["python", "app.py"]

FROM python:3.6.2

# Update packages
RUN apt-get update -y
RUN pip install numpy
RUN pip install sklearn
RUN pip install tensorflow
RUN pip install scipy
RUN pip install pandas
RUN pip install Flask
RUN pip install redis
RUN pip install opencv-python
RUN pip install matplotlib

# Set the working directory to /app
# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
# Bundle app source
# ADD . /src

# Expose - selects the port for the container
# EXPOSE  5000

# Run
# ENV PYTHONPATH  $PYTHONPATH:/src/info:/src/
# WORKDIR /src/
