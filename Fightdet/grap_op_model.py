
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout,Lambda, Input
from tensorflow.keras import layers, models

# extract the rgb images
def get_gsc(input_x):
    gsc = input_x[...,:3]
    return gsc

# extract the optical flows
def get_opt(input_x):
    opt= input_x[...,3:5]
    return opt

inputs = Input(shape=(64,224,224,5))

gsc = Lambda(get_gsc,output_shape=None)(inputs)
opt = Lambda(get_opt,output_shape=None)(inputs)


# #### 1. ARCHITECTURE
# gsc=layers.Conv3D(6, kernel_size=(3, 3,3), activation='relu')(gsc)
# gsc=layers.Conv3D(4, kernel_size=(3,1,1), activation='relu')(gsc) # kernel_size = 3 <==> (3, 3)
# gsc=layers.Flatten()(gsc)
# gsc=layers.Dense(1, activation='sigmoid')(gsc)

# opt=layers.Conv3D(6, kernel_size=(3, 3,3), activation='relu')(opt)
# opt=layers.Conv3D(4, kernel_size=(3,1,1), activation='relu')(opt) # kernel_size = 3 <==> (3, 3)
# opt=layers.Flatten()(opt)
# opt=layers.Dense(1, activation='sigmoid')(opt)

# # Build the model
# pred = Dense(1, activation='softmax')(gsc)

# model = Model(inputs=inputs, outputs=pred)
# model.summary()

#### 1. ARCHITECTURE
model = Sequential()
model.add(layers.Conv3D(6, kernel_size=(3, 3,3), activation='relu', input_shape=(64, 225, 225, 3)))
model.add(layers.Conv3D(4, kernel_size=(3,1,1), activation='relu')) # kernel_size = 3 <==> (3, 3)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
