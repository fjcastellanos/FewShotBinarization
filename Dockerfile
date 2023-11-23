FROM tensorflow/tensorflow:2.9.0-gpu

RUN apt update
RUN apt install ffmpeg libsm6 -y
RUN apt install nano

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install tqdm
RUN pip install pandas
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install scikit-multilearn
RUN pip install tensorflow-addons
RUN pip install pycocotools

RUN echo 'alias ll="ls -l"' >> ~/.bashrc
