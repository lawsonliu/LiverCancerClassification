import os 
import sys
sys.path.append('.')
sys.path.append('..')
import tensorflow as tf
from model import ResidualNetwork3D
from dataset import DataSet

# GPU Configuration for tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)


def patch_based_train(iteration,batch_size=32):
    # Define placeholder for data
    Image = tf.placeholder(dtype=tf.float32,shape=[None,224,224,24,1])
    Label = tf.placeholder(dtype=tf.int32,shape=[None])

    # Get Model output
    logits = ResidualNetwork3D(input=Image,num_classes=2,dropout_rate=0.5)

    # Calculate losses
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Label)
    loss = tf.reduce_mean(input_tensor=cross_entropy)

    # Calculate gradients and apply gradient descent use tf.train.AdamOptimizer API
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss)

    # Open a tensorflow session and initialize all trainable variables
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(tf.global_variables_initializer())

    # Get a Instance of DataSet class
    dataset = DataSet(image_dir='./TrainData',bbox_path='./BBOX.pkl',label_path='./train_label.csv')

    # Optimize loss function iterately
    batch_num = int(dataset.image_num / batch_size)
    for i in range(iteration):
        # Traverse train dataset at one iteration using resampling
        loss_avg = 0.0
        for j in range(batch_num):
            # Get real data and configure fetch_list and feed_dict
            fetch_list = [loss,optimizer]
            X,Y = dataset.patch_based_random_batch(image_dtype='float32',label_dtype='int32',batch_size=batch_size)
            feed_dict = {Image: X, Label: Y}

            # Run tensorflow computation graph(including optimization) and fetch loss
            loss_batch, _ = session.run(fetches=fetch_list,feed_dict=feed_dict)
            loss_avg += loss_batch
        loss_avg /= float(batch_size * batch_num)
        print('Iteraion [{0}] Avarage CrossEntropy Loss: {1}'.format(i,loss_avg))

    session.close()
    pass


def volume_based_train():
    pass


if __name__ == "__main__":
    pass
    patch_based_train(iteration=1,batch_size=8)
