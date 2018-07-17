FROM tensorflow-cpu:1.8.0

COPY scripts /root/scripts
COPY mnist-full-connected /root/mnist-full-connected

RUN pip install flask

CMD /root/scripts/start.sh
