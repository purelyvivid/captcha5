# Use an official Python runtime as a parent image
# FROM python:2.7-slim
FROM python:3.7.3-stretch

# 创建 code 文件夹并将其设置为工作目录
RUN mkdir /code
WORKDIR /code

RUN apt-get update && apt-get install -y libmagic-dev
# 更新 pip
RUN pip install pip -U

# 将 requirements.txt 复制到容器的 code 目录
ADD requirements.txt /code/

# 安装库
RUN pip install -r requirements.txt

# 将当前目录复制到容器的 code 目录
ADD . /code/

# Make port 80 available to the world outside this container
#EXPOSE 5000

# Run app.py when the container launches
#CMD ["python", "app.py"]
#CMD ["flask", "run"]
