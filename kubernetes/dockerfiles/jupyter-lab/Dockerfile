FROM ubuntu:latest

RUN apt-get update && apt-get install -y sudo openssh-server vim
RUN apt-get install -y supervisor
RUN mkdir /var/run/sshd
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 matteo 
RUN echo 'matteo:i_like_giotto' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# add the requirements files and the multiprocess one
COPY requirements.txt .

RUN ["apt-get", "install", "-y", "python3-pip"]
RUN ["pip3", "install", "-r", "requirements.txt"]
RUN ["pip3", "install", "jupyterlab"]

# mysqlclient
RUN apt-get install -y libssl-dev
RUN apt-get install -y python3-dev libmysqlclient-dev
RUN pip install mysqlclient

# git
RUN apt-get install -y git

# for multiprocesses
ADD kubernetes/dockerfiles/jupyter-lab/supervisord.conf supervisord.conf 

# expose the port for jupyter and ssh
EXPOSE 8888 22

# mount the volume
RUN mkdir giotto-deep
VOLUME ["/giotto-deep"]

CMD ["supervisord", "-c", "supervisord.conf"]