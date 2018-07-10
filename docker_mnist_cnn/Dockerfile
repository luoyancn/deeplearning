FROM tensorflow-cpu:1.8.0

COPY scripts /root/scripts
COPY tf-mnist-cnn /root/tf-mnist-cnn

RUN pip install flask

CMD /root/scripts/start.sh
