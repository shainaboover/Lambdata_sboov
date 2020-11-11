FROM debian

### So logging/io works reliably w/Docker
ENV PYTHONUNBUFFERED=1

### Basic Python dev dependencies 
RUN apt-get update && \
apt-get upgrade -y && \
apt-get install python3-pip curl -y && \
pip3 install pipenv

### Install Lambdata package
RUN pip install -i https://test.pypi.org/simple/ lambdata-sboov
