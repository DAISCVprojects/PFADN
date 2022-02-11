from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Convolution2D, SeparableConv2D, Cropping2D, ZeroPadding2D, Concatenate, UpSampling2D, Lambda, Add, Multiply, Conv2DTranspose
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf



def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def imageLoss( yTrue, yPred):
    
    alpha = 0.84
    return (1-alpha)*K.mean( K.abs(yTrue[:,:,:,2] - yPred[:,:,:,2])) + alpha*K.mean(1-tf.image.ssim(yTrue[:,:,:,2], yPred[:,:,:,2], 1.0))
        

def imagePSNR( yTrue, yPred):
    return 20*log10( 1.0/K.sqrt( K.mean( K.square(yTrue[:,:,:,2] - yPred[:,:,:,2])) ) )
    
    
def vectorNorm( yTrue, yPred ):
    u = yPred[:,:,:,0]
    v = yPred[:,:,:,1]
    return K.mean( K.sqrt( K.square(u)+K.square(v) ) )

def angleLoss( yTrue, yPred ):
    # Dot-product between angles
    u = yPred[:,:,:,0]
    v = yPred[:,:,:,1]
    #nr = K.sqrt( K.square(u)+K.square(v) )   
    nr = 1.0
    
    abs_dot = K.abs( (u/nr)*yTrue[:,:,:,0] + (v/nr)*yTrue[:,:,:,1] ); 
    
    return K.mean( tf.acos( abs_dot - 0.01 ))


def angleLoss2_deg( yTrue, yPred ):
   
    angle_pred = tf.atan2( yPred[:,:,:,1], yPred[:,:,:,0] ) * 0.5 + np.pi*0.5  # 0..pi
    angle_true = tf.atan2( yTrue[:,:,:,1], yTrue[:,:,:,0] ) * 0.5 + np.pi*0.5  # 0..pi
    
    a1 = K.abs( angle_pred-angle_true )
    a2 = np.pi - a1
    
    return K.mean( tf.math.minimum( a1, a2 )*180.0 / np.pi )
    
    
def angleLoss_vec( yTrue, yPred ): # l2 distance between (u,v) vectors
    
    u_gt = yTrue[...,0];
    v_gt = yTrue[...,1];
    
    u_pred = yPred[...,0];
    v_pred = yPred[...,1];
    
    return K.mean( K.square(u_gt - u_pred) + K.square(v_gt - v_pred))


def lossfun( yTrue, yPred, alpha=1.0, beta=1.0 ):
    return alpha*imageLoss( yTrue, yPred ) + beta*angleLoss_vec( yTrue, yPred )
    
################################################################################


def plot_keras_model(model, show_shapes=True,        show_layer_names=True):
    return SVG(model_to_dot(model, show_shapes=show_shapes, show_layer_names=show_layer_names).create(prog='dot',format='svg'))


def duplicate_lcol(tensor):
    return tf.concat((tensor, tf.expand_dims(tensor[:, :, -2, ...], 2)), axis=2)


def duplicate_lrow(tensor):
    return tf.concat((tensor, tf.expand_dims(tensor[:, -2, ...], 1)), axis=1)


def convolve_block( x, topbottom_shift, leftright_shift, depth, name_prefix, train=True ):
    
    assert topbottom_shift[1]==0 
    assert leftright_shift[1]==0
    
    # Cropping2D format: top-bottom , left-right
    p0 = Cropping2D( cropping=(topbottom_shift,leftright_shift), name=name_prefix+"crop" )( x )
    
    p0c = Convolution2D( depth, (2,2), strides=(2,2), padding="valid", use_bias="true", activation="relu", name=name_prefix+"conv", trainable=train )(p0)
        
    p0Cub = UpSampling2D(size=(2,2), name=name_prefix+"up" )(p0c)
    
    p0Cubs = ZeroPadding2D( padding=(topbottom_shift,leftright_shift), name=name_prefix+"zp" )( p0Cub )
    
    
    if p0Cubs.shape[2]<x.shape[2]:
        p0Cubs = Lambda( lambda t: duplicate_lcol(t), name=name_prefix+"dupcol" )( p0Cubs ) 
        
    if p0Cubs.shape[1]<x.shape[1]:
        p0Cubs = Lambda( lambda t: duplicate_lrow(t), name=name_prefix+"duprow" )( p0Cubs ) 
    
    assert p0Cubs.shape[1:3] == x.shape[1:3]
    
    
    W = int( p0Cubs.shape[1])
    H = int( p0Cubs.shape[2])
    
    if topbottom_shift[0]==0 and leftright_shift[0]==0:
        x = Lambda( lambda t: t * K.expand_dims( K.expand_dims( K.tile( [[1.0,0.0],[0.0,0.0]], ( int(W/2),int(H/2) ) ), axis=0), axis=-1), name=name_prefix+"mask00" )( p0Cubs ) 
        
    if topbottom_shift[0]==0 and leftright_shift[0]==1:
        x = Lambda( lambda t: t * K.expand_dims( K.expand_dims( K.tile( [[0.0,1.0],[0.0,0.0]], ( int(W/2),int(H/2) ) ), axis=0), axis=-1), name=name_prefix+"mask01" )( p0Cubs ) 
        
    if topbottom_shift[0]==1 and leftright_shift[0]==0:
        x = Lambda( lambda t: t * K.expand_dims( K.expand_dims( K.tile( [[0.0,0.0],[1.0,0.0]], ( int(W/2),int(H/2) ) ), axis=0), axis=-1), name=name_prefix+"mask10" )( p0Cubs ) 
        
    if topbottom_shift[0]==1 and leftright_shift[0]==1:
        x = Lambda( lambda t: t * K.expand_dims( K.expand_dims( K.tile( [[0.0,0.0],[0.0,1.0]], ( int(W/2),int(H/2) ) ), axis=0), axis=-1), name=name_prefix+"mask11" )( p0Cubs ) 
    
    return x
    
    
def create_model( w, h, initial_block_train=True ):
        
    input_data = Input( shape=(w,h,1) )
    
    x = input_data
    
    x0 = convolve_block( x, (0,0),(0,0), 16, "0_cb0_", train=initial_block_train )
    x1 = convolve_block( x, (0,0),(1,0), 16, "0_cb1_", train=initial_block_train )
    x2 = convolve_block( x, (1,0),(0,0), 16, "0_cb2_", train=initial_block_train )
    x3 = convolve_block( x, (1,0),(1,0), 16, "0_cb3_", train=initial_block_train )
    
    x = Add()( [x0,x1,x2,x3] )
    
    x0 = convolve_block( x, (0,0),(0,0), 16, "1_cb0_", train=initial_block_train )
    x1 = convolve_block( x, (0,0),(1,0), 16, "1_cb1_", train=initial_block_train )
    x2 = convolve_block( x, (1,0),(0,0), 16, "1_cb2_", train=initial_block_train )
    x3 = convolve_block( x, (1,0),(1,0), 16, "1_cb3_", train=initial_block_train ) 
    
    
    x_first = Add()( [x0,x1,x2,x3] )
    
    # Demosaic and stack features channel-wise
    x_channels = Lambda( lambda t: tf.concat([t[:,::2,::2,:], t[:,1::2,::2,:], t[:,::2,1::2,:], t[:,1::2,1::2,:]], -1), name="x_res_channels" )( x_first )  # 64 x 64 x 64
        
    # Intensity
    x = Convolution2D( 12, (1,1), padding="same", use_bias="true", activation="relu"  )( x_channels )
    x = Convolution2D( 12, (3,3), padding="same", use_bias="true", activation="relu"  )( x )
    x = Convolution2D( 12, (3,3), padding="same", use_bias="true", activation="relu"  )( x )
    x = Convolution2D( 12, (3,3), padding="same", use_bias="true", activation="relu"  )( x )
    x = Convolution2D( 12, (3,3), padding="same", use_bias="true", activation="relu"  )( x )
    x = Convolution2D( 64, (1,1), padding="same", use_bias="true", activation="relu"  )( x )
    intensity = Conv2DTranspose(1, (5,5), strides=(2, 2), dilation_rate=(1, 1), padding="same")( x )
    
    # angle
    x = Convolution2D( 12, (1,1), padding="same", use_bias="true", activation="relu"  )( x_channels )
    x = Convolution2D( 12, (3,3), padding="same", use_bias="true", activation="relu"  )( x )
    x = Convolution2D(  8, (3,3), padding="same", use_bias="true", activation="relu"  )( x )
    x = UpSampling2D( size=(2,2), interpolation="bilinear" )(x)
    u = Convolution2D( 1, (5,5), padding="same", use_bias="true", activation="linear"  )( x )
    v = Convolution2D( 1, (5,5), padding="same", use_bias="true", activation="linear"  )( x )
    
    # normalize
    uvnorm = Lambda( lambda t: 1.0/K.sqrt( K.square(t[0]) + K.square(t[1]) ) )( [u,v] )
    un = Multiply()( [u,uvnorm])
    vn = Multiply()( [v,uvnorm])

    out = Concatenate( )( [un, vn, intensity] )
    model = Model(input_data, out)
    
    return model


