import tensorflow as tf


def conv_block(input,kernel_size,filter_nums,strides):
    '''
    Args:
        ipnut: input tensor
        kernel_size: kernel size of middle convolution layer at main path
        filter_nums: 3 number represents number of filters in 3 convolution 
                     layer at main path
        strides: strides of first convolution layer
    Returns:
        Output tensor for the block
    '''
    # Convolution layer1 at main path, feature map size reduce half
    x = tf.layers.conv3d(inputs=input,
                         filters=filter_nums[0],
                         kernel_size=(1,1,1),
                         strides=strides,
                         padding='same',
                         use_bias=True,
                         trainable=True)
    x = tf.layers.batch_normalization(inputs=x)
    x = tf.nn.relu(features=x)
    
    # Convolution layer2 at main path
    x = tf.layers.conv3d(inputs=x,
                        filters=filter_nums[1],
                        kernel_size=kernel_size,
                        strides=(1,1,1),
                        padding='same',
                        use_bias=True,
                        trainable=True)
    x = tf.layers.batch_normalization(inputs=x)
    x = tf.nn.relu(features=x)
    
    
    # Convolution layer3 at main path without activation
    x = tf.layers.conv3d(inputs=x,
                        filters=filter_nums[2],
                        kernel_size=(1,1,1),
                        strides=(1,1,1),
                        padding='same',
                        use_bias=True,
                        trainable=True)
    x = tf.layers.batch_normalization(inputs=x)
    
    # Convolution layer at shotcut path, make feature map be
    # same dimesion as main path
    shortcut = tf.layers.conv3d(inputs=input,
                                filters=filter_nums[2],
                                kernel_size=(1,1,1),
                                strides=strides,
                                padding='same',
                                use_bias=True,
                                trainable=True)
    shortcut = tf.layers.batch_normalization(inputs=shortcut)
    
    # Element-wise addition of two paths
    out = tf.add(x, shortcut)
    out = tf.nn.relu(features=out)
    
    return out



def identity_block(input,kernel_size,filter_nums):
    '''
    input: input tensor
    kernel_size: kernel size of middle layer at main path
    filter_nums: 3 number represents number of filters in 3 convolution 
                 layer at main path
    '''
    # Convolution layer 1 at main path
    x = tf.layers.conv3d(inputs=input,
                         filters=filter_nums[0],
                         kernel_size=(1,1,1),
                         strides=(1,1,1),
                         padding='same',
                         use_bias=True,
                         trainable=True)
    x = tf.layers.batch_normalization(inputs=x)
    x = tf.nn.relu(features=x)
    
    # Convolution layer 2 at main path
    x = tf.layers.conv3d(inputs=x,
                         filters=filter_nums[1],
                         kernel_size=kernel_size,
                         strides=(1,1,1),
                         padding='same',
                         use_bias=True,
                         trainable=True)
    x = tf.layers.batch_normalization(inputs=x)
    x = tf.nn.relu(features=x)
    
    # Convolution layer 3 at main path without activation
    x = tf.layers.conv3d(inputs=x,
                         filters=filter_nums[2],
                         kernel_size=(1,1,1),
                         strides=(1,1,1),
                         padding='same',
                         use_bias=True,
                         trainable=True)
    x = tf.layers.batch_normalization(inputs=x)
    
    # Element-wise addition of input(shortcut) and main path
    out = tf.add(x,input)
    out = tf.nn.relu(features=out)
    
    return out



def ResidualNetwork3D(input,num_classes,dropout_rate=0.0):
    '''
    将经典的ResNet50应用于3D图像
    Args:
        input: Image of shape [batch_size,height,width,depth,channel]
        num_classes:
        dropout_rate: dropout rate of the last 2 fully-connected layers,
                   set 0.0 for validating and testing
    Returns:
        out: last layer's output of shape [batch_size,num_classes]
    '''
    
    x = tf.layers.conv3d(inputs=input,
                         filters=64,
                         kernel_size=(7,7,3),
                         strides=(2,2,1),
                         padding='same',
                         use_bias=True,
                         trainable=True)
    x = tf.layers.batch_normalization(inputs=x)
    x = tf.nn.relu(features=x)
    
    x = conv_block(input=x,kernel_size=3,filter_nums=[64,64,256],strides=[1,1,1])
    x = identity_block(input=x,kernel_size=3,filter_nums=[64,64,256])
    x = identity_block(input=x,kernel_size=3,filter_nums=[64,64,256])
    
    x = conv_block(input=x,kernel_size=3,filter_nums=[128,128,512],strides=[2,2,2])
    x = identity_block(input=x,kernel_size=3,filter_nums=[128,128,512])
    x = identity_block(input=x,kernel_size=3,filter_nums=[128,128,512])
    x = identity_block(input=x,kernel_size=3,filter_nums=[128,128,512])
    
    x =conv_block(input=x,kernel_size=3,filter_nums=[256,256,1024],strides=[2,2,2])
    x = identity_block(input=x,kernel_size=3,filter_nums=[256,256,1024])
    x = identity_block(input=x,kernel_size=3,filter_nums=[256,256,1024])    
    x = identity_block(input=x,kernel_size=3,filter_nums=[256,256,1024])    
    x = identity_block(input=x,kernel_size=3,filter_nums=[256,256,1024])    
    x = identity_block(input=x,kernel_size=3,filter_nums=[256,256,1024])    
    
    x = conv_block(input=x,kernel_size=3,filter_nums=[512,512,2048],strides=[2,2,1])
    x = identity_block(input=x,kernel_size=3,filter_nums=[512,512,2048])
    x = identity_block(input=x,kernel_size=3,filter_nums=[512,512,2048])    
    
    # apply global pooling
    x_size = x.shape.as_list()
    feature_map_size = [x_size[1], x_size[2], x_size[3]]
    out = tf.layers.max_pooling3d(inputs=x,pool_size=feature_map_size,strides=(1,1,1),padding='valid')

    
    # add 2 fully-connected layers with dropout 
    out = tf.layers.dense(inputs=out,
                          units=64,
                          activation=tf.nn.relu,
                          use_bias=True,
                          trainable=True)
    out = tf.layers.dropout(inputs=out,rate=dropout_rate)
    out = tf.layers.dense(inputs=out,
                          units=num_classes,
                          activation=None,
                          use_bias=True,
                          trainable=True)
    
    return out



