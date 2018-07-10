#!/bin/bash

function run_mnist_tensorboard() {
    if [ $TFBOARD_CON_TYPE == 'tfboard' ]
    then
        exec tensorboard --logdir=$TRAIN_LOG_DIR
    elif [ $TFBOARD_CON_TYPE == 'haproxy' ]
    then
        sed -i "s/tftoken/$TFBOARD_TOKEN/g" /root/scripts/haproxy.cfg
        sed -i "s/taskname/$TASK_NAME/g" /root/scripts/haproxy.cfg
        exec haproxy -f /root/scripts/haproxy.cfg -db
    fi
}

function run_ssh_server() {
    mkdir -p /root/.ssh
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 0600 /root/.ssh/authorized_keys
    /sbin/sshd -e
}

function run_user_custom_command() {
    exec /root/mnist-full-connected/fc_mnist_train
} 

function run_jupyter_notebook() {
    sed -i "s/^c.NotebookApp.token.*$/c.NotebookApp.token = '$JUPYTER_TOKEN'/g"   /root/.jupyter/jupyter_notebook_config.py
    sed -i "s/#c.NotebookApp.base_url.*$/c.NotebookApp.base_url = '${TASK_NAME}\/'/g"   /root/.jupyter/jupyter_notebook_config.py
    exec jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root &
}

function run_mnist_session() {
    run_ssh_server

    run_jupyter_notebook

    run_user_custom_command
}

function run_mnist_model_server() {
    if [ ! -d $MODEL_BASE_PATH ]
    then
        mkdir -p $MODEL_BASE_PATH
    fi
    exec /root/mnist-full-connected/fc_mnist_predict_server
}

GCC_LIB=/usr/lib64/gcc_6.4.0/lib64/
function export_gcc_lib() {
    if [ -d "$GCC_LIB" ]
    then
        export LD_LIBRARY_PATH=${GCC_LIB}:${LD_LIBRARY_PATH}
    fi
}

export_gcc_lib

case $RESOURCE_TYPE in
    "tensorboard" ) run_mnist_tensorboard ;;
    "session"     ) run_mnist_session ;;
    "model_server") run_mnist_model_server ;;
esac
