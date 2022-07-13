import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential, Model

class Recognizer:
    def __init__(self, df_train, df_val, nb_classes, steps_train, steps_val = 0, debug=True):
        
        if (debug):
            print(f"tensorflow verion: {tf.__version__}")
        
        self.nb_classes = nb_classes
        self.df_train = df_train
        self.df_val = df_val
        self.steps_train = steps_train
        self.steps_val = steps_val
        
        if ((df_val != None) & (steps_val == 0)):
            print("Recognizer Error #001: steps_val missing")
        
        def angular_error(y_true, y_pred):
            y_true_s, y_true_c = tf.split(y_true, num_or_size_splits=2, axis=1)
            y_pred_s, y_pred_c = tf.split(y_pred, num_or_size_splits=2, axis=1)
            e = 0.5*(tf.square(y_true_s - y_pred_s) + tf.square(y_true_c - y_pred_c))
            return tf.math.sqrt(tf.math.reduce_mean(e))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model = self.build_model()
        
        self.model.compile(
            optimizer = optimizer,
            loss = {
                'class_output' : tf.keras.losses.CategoricalCrossentropy(),
                'rot_output': angular_error
            },
            metrics = {
                'class_output' : tf.keras.metrics.CategoricalAccuracy(),
                'rot_output': angular_error
            }, 
            loss_weights=[1.0, 1.0])
        
        if (debug):
            print(self.model.summary())
        
        #tf.keras.utils.plot_model(self.model, "./figures/classifier.png", show_shapes=True)
        
    def build_model(self):
        
        i = tf.keras.Input(shape = (32,32,1), name = 'input')

        a0 = Conv2D(256/4, kernel_size = (3,3), name = 'Conv0', padding = 'same')(i)
        a1 = Conv2D(256/4, kernel_size = (3,3), name = 'Conv1', padding = 'same')(a0)
        a2 = BatchNormalization(name = 'BatchNorm0')(a1)
        a3 = MaxPool2D(name = 'MaxPool0')(a2)
        a4 = Dropout(.4)(a3)
        a5 = Conv2D(512/4, kernel_size = (3,3), name = 'Conv2', padding = 'same')(a4)
        a6 = Conv2D(512/4, kernel_size = (3,3), name = 'Conv3', padding = 'same')(a5)
        a7 = BatchNormalization(name = 'BatchNorm1')(a6)
        a8 = MaxPool2D(name = 'MaxPool1')(a7)
        a9 = Dropout(.4)(a8)


        # Classification branch
        c0 = Conv2D(512/4, kernel_size = (3,3), name = 'ConvC1', padding = 'same')(a9)
        c1 = Conv2D(512/8, kernel_size = (3,3), name = 'ConvC2', padding = 'same')(c0)
        c2 = BatchNormalization(name = 'BatchNormC1')(c1)
        c3 = MaxPool2D(name = 'MaxPoolC1')(c2)
        c4 = Dropout(.4)(c3)

        c5 = Flatten()(c4)
        c6 = Dense(512)(c5)
        c7 = BatchNormalization(name = 'BatchNormC2')(c6)
        c8 = Dropout(.4)(c7)
        o_c = Dense(self.nb_classes, activation='softmax', name = 'class_output')(c8)

        # Regression branch
        r0 = Conv2D(512/4, kernel_size = (3,3), name = 'ConvR1', padding = 'same')(a9)
        r1 = Conv2D(512/8, kernel_size = (3,3), name = 'ConvR2', padding = 'same')(r0)
        r2 = BatchNormalization(name = 'BatchNormR1')(r1)
        r3 = MaxPool2D(name = 'MaxPoolR1')(r2)
        r4 = Dropout(.4)(r3)

        r5 = Flatten()(r4)
        r6 = Dense(512, activation='relu')(r5)
        r7 = BatchNormalization(name = 'BatchNormR2')(r6)
        r8 = Dropout(.4)(r7)
        o_r = Dense(2, activation='linear', name = 'rot_output')(r8)

        model = Model(i, [o_c, o_r], name = 'recognizer')
        
        return model
    
    def train(self, epochs = 1, batch_size = 64, verbose = 1):
        
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

        if (self.df_val == None):
            return self.model.fit(self.df_train,
                           epochs = epochs,
                           steps_per_epoch = self.steps_train,
                           # 0 = silent, 1 = progress bar, 2 = one line per epoch
                           verbose=verbose, 
                           callbacks = [es])
        else:
            return self.model.fit(self.df_train,
                           epochs = epochs,
                           steps_per_epoch = self.steps_train,
                           validation_steps = self.steps_val,
                           validation_data = self.df_val,
                           # 0 = silent, 1 = progress bar, 2 = one line per epoch
                           verbose=verbose, 
                           callbacks = [es])