import tensorflow as tf

tf.app.flags.DEFINE_integer(
    "task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

def main(_):
    worker1 = "192.168.137.30:2222"
    worker2 = "192.168.137.30:2223"
    worker_hosts = [worker1, worker2]
    cluster_spec = tf.train.ClusterSpec(
        {"worker": worker_hosts})
    server = tf.train.Server(
        cluster_spec, job_name="worker",
        task_index=FLAGS.task_index)
    server.join()

if __name__ == '__main__':
    tf.app.run()