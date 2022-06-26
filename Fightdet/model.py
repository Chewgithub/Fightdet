from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Lambda, Input, Multiply
from tensorflow.keras import layers, models

# Input Function
# extract gray_frame from video
def get_gsc(input_x):
    '''
    This function takes in a list of video array with shape (None, 64,224,224,3) and return the first channel only
    Parameters:
        video_array (shape: None, frame,height,width,3)
    Returns:
        ndarray (shape: None, frame, height, width, 1)
    '''
    gsc = input_x[...,:1]
    return gsc

# extract the optical flows
def get_opt(input_x):
    '''
    This function takes in a list of video array with shape (None, 64,224,224,3) and return the last two channel only
    Parameters:
        video_array (shape: None, 64,height,width,3)
    Returns:
        ndarray (shape: None, 64, height, width, 2)
    '''
    opt= input_x[...,1:]
    return opt


'''
Model Building
'''
#GSC channel (gray screen channel)
def gsc_channel(inputs):
    '''
    This function defines model architecture for gray screen channel input
    Parameters:
        inputs: Input class from tensorflow.keras.layers
    Returns:
        gray screen channel stream output
    '''
    gsc = Lambda(get_gsc, name='gsc_1')(inputs)
    #### ARCHITECTURE
    # 1st Conv3D layer
    gsc = layers.Conv3D(16,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='gsc_2')(gsc)
    # 2nd Conv3D layer
    gsc = layers.Conv3D(16,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='gsc_3')(gsc)
    gsc = layers.MaxPool3D(pool_size=(1, 2, 2), name='gsc_4')(gsc)

    # 3rd Conv3D layer
    gsc = layers.Conv3D(16,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='gsc_5')(gsc)
    # 4th Conv3D layer
    gsc = layers.Conv3D(16,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='gsc_6')(gsc)
    gsc = layers.MaxPool3D(pool_size=(1, 2, 2), name='gsc_7')(gsc)

    # 5th Conv3D layer
    gsc = layers.Conv3D(32,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='gsc_8')(gsc)

     # 6th Conv3D layer
    gsc = layers.Conv3D(32,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='gsc_9')(gsc)
    gsc = layers.MaxPool3D(pool_size=(1, 2, 2), name='gsc_10')(gsc)

    # 7th Conv3D layer
    gsc = layers.Conv3D(32,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='gsc_11')(gsc)

    # 8th Conv3D layer
    gsc = layers.Conv3D(32,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='gsc_12')(gsc)
    gsc = layers.MaxPool3D(pool_size=(1, 2, 2), name='gsc_13')(gsc)

    return gsc

#OPT channel (optical flow channel)
def opt_channel(inputs):
    '''
    This function defines model architecture for optical flow input
    Parameters:
        inputs: Input class from tensorflow.keras.layers
    Returns:
        optical flow stream output
    '''
    # extract opt
    opt = layers.Lambda(get_opt, name='opt_1')(inputs)

    #### ARCHITECTURE
    # 1st Conv3D layer
    opt = layers.Conv3D(16,
                    kernel_size=(1, 3, 3),
                    strides=(1, 1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='relu',
                    name='opt_2')(opt)

    # 2nd Conv3D layer
    opt = layers.Conv3D(16,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='opt_3')(opt)
    opt = layers.MaxPool3D(pool_size=(1, 2, 2), name='opt_4')(opt)


    # 3rd Conv3D layer
    opt = layers.Conv3D(16,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='opt_5')(opt)

    # 4th Conv3D layer
    opt = layers.Conv3D(16,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='opt_6')(opt)
    opt = layers.MaxPool3D(pool_size=(1, 2, 2), name='opt_7')(opt)

    # 5th Conv3D layer
    opt = layers.Conv3D(32,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='sigmoid',
                        name='opt_8')(opt)

    # 6th Conv3D layer
    opt = layers.Conv3D(32,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='sigmoid',
                        name='opt_9')(opt)
    opt = layers.MaxPool3D(pool_size=(1, 2, 2), name='opt_10')(opt)

    # 7th Conv3D layer
    opt = layers.Conv3D(32,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='sigmoid',
                        name='opt_11')(opt)

    # 8th Conv3D layer
    opt = layers.Conv3D(32,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='sigmoid',
                        name='opt_12')(opt)
    opt = layers.MaxPool3D(pool_size=(1, 2, 2), name='opt_13')(opt)
    return opt

#Fused channel
def fused_channel(gsc, opt):
    '''
    This function defines model architecture for two-stream combined
    Parameters:
        gsc: gray screen channel stream output
        opt: optical flow stream output
    Returns:
        fused stream output
    '''
    fused = layers.Multiply(name='fuse_1')([gsc, opt])
    fused = layers.MaxPool3D(pool_size=(8, 1, 1), name='fuse_2')(fused)

    # additional conv3d and pooling
    fused = layers.Conv3D(64,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='merge_1')(fused)
    fused = layers.Conv3D(64,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='merge_2')(fused)
    fused = layers.MaxPool3D(pool_size=(2, 2, 2), name='merge_3')(fused)

    # additional conv3d and pooling
    fused = layers.Conv3D(64,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='merge_4')(fused)
    fused = layers.Conv3D(64,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='merge_5')(fused)
    fused = layers.MaxPool3D(pool_size=(2, 2, 2), name='merge_6')(fused)

    # additional conv3d and pooling
    fused = layers.Conv3D(128,
                        kernel_size=(1, 3, 3),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='merge_7')(fused)
    fused = layers.Conv3D(128,
                        kernel_size=(3, 1, 1),
                        strides=(1, 1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='relu',
                        name='merge_8')(fused)
    fused = layers.MaxPool3D(pool_size=(2, 3, 3), name='merge_9')(fused)

    # flatten into FC layer
    fused = layers.Flatten(name='flat')(fused)
    fused = layers.Dense(128, activation='relu', name='fc_1')(fused)
    fused = layers.Dropout(0.2, name='fc_2')(fused)
    fused = layers.Dense(32, activation='relu', name='fc_3')(fused)

    # output
    output = layers.Dense(1, activation='sigmoid', name='pred')(fused)
    return output


#model_instatiate
def model_instantiate():
    '''
    This function instantiate a two stream CNN model architecture.
    Parameters:
        None
    Returns:
        model from tensorflow.keras.models
    '''
    inputs = Input(shape=(64,224,224,3))
    output = fused_channel(gsc_channel(inputs), opt_channel(inputs))
    model = Model(inputs, output, name='cnn_model')
    return model
