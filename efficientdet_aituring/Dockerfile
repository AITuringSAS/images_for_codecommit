# import tensorflow-image=2.6 with cuda drivers for ubuntu focal 20.04 LTSC
FROM nvcr.io/nvidia/tensorflow:21.09-tf2-py3

# update ubuntu packages
RUN apt-get update

# remove Error debconf: unable to initialize frontend: Dialog
RUN apt-get install -y dialog apt-utils
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get install -y -q

# install os dependencies
RUN apt-get install -y wget tar unzip git libgl1 

# add user without password
RUN adduser --disabled-password --gecos '' --shell /bin/bash aiteam
USER aiteam

# set working directory 
WORKDIR /home/aiteam/automl_v2/efficientdet_aituring

# enable python buffer stdout
ENV PYTHONUNBUFFERED definitely

# add python local bin to path environment
ENV PATH "$PATH:/home/aiteam/.local/bin"

# copy project files
COPY . .

# upgrade pip
RUN pip install --upgrade pip

# install aditional project dependencies for docker
# tensorflow-addons, plugins, etc
RUN pip install -r docker_requirements.txt

# set python path environment
ENV PYTHONPATH=".:$PYTHONPATH"

# add main.py
ADD main.py .

# set default entry point to main.py
# ENTRYPOINT ["python", "./main.py"]
CMD ["python", "./main.py"] 