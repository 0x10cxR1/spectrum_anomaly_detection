import sys, os
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.layers.core import Activation
from keras import backend as K
from keras.optimizers import RMSprop
from keras.layers.pooling import AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.core import Dropout
from keras.models import load_model
import keras
from keras.layers import Input
from sklearn.metrics import mean_squared_error
from keras.layers.normalization import BatchNormalization


def CDF(data_list,cut_off_value,down_sample_rate,x_shrink_rate,out):
    list_len = len(data_list)
    sorted_list = sorted(data_list)
    count = 0
    score_pencentage_list = []
    fout = open(out,'w')

    for i in range(0,list_len):
        percentage = float(i+1) / list_len
        score = sorted_list[i]
        if score > cut_off_value and cut_off_value > 0:
            break
        score = float(score) / x_shrink_rate
        count = count + 1
        if count % down_sample_rate == 0:
            score_pencentage_list.append((score,percentage))
            fout.write('%s %s\n' %(score,percentage))


def fix_gpu_memory():
    import tensorflow as tf
    import keras.backend as K
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    # init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess


class DataGenerator():
    def __init__(self, filename, batch_size, timesteps, data_dim, predict_steps):
        self.filename = filename
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.predict_steps = predict_steps
        self.data_dim = data_dim

    def generate(self):
        while 1:
            x = []
            y = []
            with open(self.filename, 'r') as f:
                for line in f:
                    if len(x) == self.batch_size:
                        x = []
                        y = []
                    x_train = line.split()
                    if len(x_train) < (self.timesteps+self.predict_steps)*self.data_dim:
                        continue
                    x1_train = [float(x_train[i]) for i in range(self.timesteps*self.data_dim)]
                    y1_train = [float(x_train[i]) for i in range(self.timesteps*self.data_dim, (self.timesteps+self.predict_steps)*self.data_dim)]
                    x.append(x1_train)
                    y.append(y1_train)

                    if len(x) == self.batch_size:
                        yield np.reshape(x, (self.batch_size, self.timesteps, self.data_dim)), np.reshape(y, (self.batch_size, self.data_dim*self.predict_steps))

            f.close()


def train_LSTM(t_filename, v_filename, model_filename, weights_filename):
    train_generator = DataGenerator(t_filename, batch_size=batch_size, timesteps=timesteps, data_dim=data_dim, predict_steps=predict_steps).generate()
    valid_generator = DataGenerator(v_filename, batch_size=batch_size, timesteps=timesteps, data_dim=data_dim, predict_steps=predict_steps).generate()
    t_len = int(train_num) // batch_size
    v_len = int(valid_num) // batch_size

    callbacks = [
    keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=2, verbose=0, mode='auto')
    ]



    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(timesteps, data_dim,), return_sequences=True, stateful=False))
    model.add(BatchNormalization())
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(hidden_size))
    model.add(BatchNormalization())
    model.add(Dense(data_dim*predict_steps))


    print('Training Begin')

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit_generator(generator=train_generator, epochs=epochs,
                        steps_per_epoch=t_len, callbacks=callbacks, validation_data=valid_generator, validation_steps=v_len)
    model.save(model_filename)
    model.save_weights(weights_filename)



def transfer_LSTM(t_filename, v_filename, old_weights_filename, model_filename, weights_filename):
    train_generator = DataGenerator(t_filename, batch_size=batch_size, timesteps=timesteps, data_dim=data_dim, predict_steps=predict_steps).generate()
    valid_generator = DataGenerator(v_filename, batch_size=batch_size, timesteps=timesteps, data_dim=data_dim, predict_steps=predict_steps).generate()
    t_len = int(train_num) // batch_size
    v_len = int(valid_num) // batch_size

    callbacks = [
    keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True),
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=2, verbose=0, mode='auto')
    ]


    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(timesteps, data_dim,), return_sequences=True, stateful=False))
    model.add(BatchNormalization())
    model.add(LSTM(hidden_size))
    model.add(BatchNormalization())
    #model.add(LSTM(hidden_size))
    #model.add(BatchNormalization())
    model.add(Dense(data_dim*predict_steps))

    model.load_weights(old_weights_filename)

    for layer in model.layers[:1]:
        layer.trainable = False


    print('Training Begin')

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit_generator(generator=train_generator, epochs=epochs,
                        steps_per_epoch=t_len, callbacks=callbacks, validation_data=valid_generator, validation_steps=v_len)
    model.save(model_filename)
    model.save_weights(weights_filename)



def test_LSTM(t_filename, model_filename, out_filename, out_filename_cdf):
    model = load_model(model_filename)
    mse_list = []
    out_f = open(out_filename, 'w')

    line_num = 0
    x = []
    y = []
    with open(t_filename, 'r') as f:
        for line in f:
            if line_num%200 == 0 and line_num>0:
                print(len(x))
                test_x = np.reshape(x, (200, timesteps, data_dim))
                test_y = np.reshape(y, (200, data_dim*predict_steps))
                predict_y = model.predict(test_x)
                for j in range(len(predict_y)):
                    mse = mean_squared_error(test_y[j], predict_y[j])
                    out_f.write('%s\n' %mse)
                    mse_list.append(mse)
                x = []
                y = []

            data_test = line.split()
            x1_test = [float(data_test[i]) for i in range(timesteps*data_dim)]
            y1_test = [float(data_test[i]) for i in range(timesteps*data_dim, (timesteps+predict_steps)*data_dim)]
            x.append(x1_test)
            y.append(y1_test)
            line_num += 1

    CDF(mse_list, -1, 10, 1, out_filename_cdf)




def main():
    gpu_num = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    import gc
    gc.disable()
    fix_gpu_memory()

    global batch_size, timesteps, predict_steps, data_dim, epochs, hidden_size, train_num, test_num, valid_num
    batch_size = 64
    timesteps = 50
    predict_steps = 25
    data_dim = 128
    epochs = 50
    hidden_size = 64
    train_num = 200000
    valid_num = 5000
    test_num = 100000

    cmd = sys.argv[1]
    if cmd == "train":
        t_filename = sys.argv[2]
        v_filename = sys.argv[3]
        model_filename = sys.argv[4]
        weights_filename = sys.argv[5]
        train_LSTM(t_filename, v_filename, model_filename, weights_filename)
    if cmd == "test":
        t_filename = sys.argv[2]
        out_filename = sys.argv[3]
        out_filename_cdf = sys.argv[4]
        model_filename = sys.argv[5]
        test_LSTM(t_filename, model_filename, out_filename, out_filename_cdf)
    if cmd == "transfer":
        t_filename = sys.argv[2]
        v_filename = sys.argv[3]
        old_weights_filename = sys.argv[4]
        model_filename = sys.argv[5]
        weights_filename = sys.argv[6]
        transfer_LSTM(t_filename, v_filename, old_weights_filename, model_filename, weights_filename)


if __name__ == '__main__':
    main()
