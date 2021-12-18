FROM ubuntu:latest
COPY kubernetes/dockerfiles/tensorboard/requirements_tb.txt requirements.txt

RUN apt-get update && apt-get install -y sudo
RUN ["sudo", "apt-get", "install", "-y", "python3-pip"]
RUN ["pip3", "install", "-r", "requirements.txt"]

# mount the volume
RUN mkdir giotto-deep
VOLUME ["/giotto-deep"]

# expose port and run tensorboard
EXPOSE 6006

# expect the runs to be inside the runs folder in the volume
CMD ["tensorboard", "--logdir=giotto-deep/runs", "--bind_all"]
