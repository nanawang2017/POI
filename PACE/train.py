# coding: utf-8


# In[13]:


import numpy as np
import keras
from keras import initializers
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate
from keras.optimizers import Adam
import pickle
from time import time
from mine import dataset_similiar_user


def init_normal(shape, name=None):
    return initializers.normal(shape)


def get_Model(num_users, num_items, latent_dim, user_con_len, item_con_len, layers=[20, 10, 5], regs=[0, 0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                               embeddings_initializer='uniform', embeddings_regularizer=l2(regs[0]), input_length=1)
    item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                               embeddings_initializer='uniform', embeddings_regularizer=l2(regs[1]), input_length=1)

    user_latent = Flatten()(user_embedding(user_input))
    item_latent = Flatten()(item_embedding(item_input))

    vector = concatenate([user_latent, item_latent])

    for i in range(len(layers)):
        hidden = Dense(layers[i], activation='relu', kernel_initializer='lecun_uniform', name='ui_hidden_' + str(i))
        vector = hidden(vector)

    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

    user_context = Dense(user_con_len, activation='sigmoid', kernel_initializer='lecun_uniform', name='user_context')(
        user_latent)
    item_context = Dense(item_con_len, activation='sigmoid', kernel_initializer='lecun_uniform', name='item_context')(
        item_latent)

    model = Model(inputs=[user_input, item_input], outputs=[prediction, user_context, item_context])
    return model


# In[14]:


model = get_Model(100000, 100000, 10, 37002, 12223)
config = model.get_config()
weights = model.get_weights()


# In[15]:


def get_train_instances(train_data):
    while 1:
        user_input = train_data['user_input']
        item_input = train_data['item_input']
        ui_label = train_data['ui_label']
        u_context = train_data['u_context']
        s_context = train_data['s_context']
        for i in range(len(u_context)):
            u = []
            it = []
            p = []
            u.append(user_input[i])
            it.append(item_input[i])
            p.append(ui_label[i])
            x = {'user_input': np.array(u), 'item_input': np.array(it)}
            y = {'prediction': np.array(p), 'user_context': np.array(u_context[i]).reshape((1, 132)),
                 'item_context': np.array(s_context[i]).reshape((1, 1))}
            yield (x, y)


# In[16]:


train = None
with open('data/testdata_small.pkl', 'rb') as f:
    train = pickle.load(f)

# In[17]:


user_input = train['user']
item_input = train['spot']
ui_label = train['label']

data = dataset_similiar_user.Dataset('_small')
data.generateContextLabels()
contexts = data.context_data
u_context, s_context = contexts['user_context'], contexts['spot_context']
train_data = {}
train_data['user_input'] = user_input
train_data['item_input'] = item_input
train_data['ui_label'] = ui_label
train_data['u_context'] = u_context
train_data['s_context'] = s_context

# In[1]:


if __name__ == '__main__':
    layers = eval("[16,8]")
    reg_layers = eval("[0,0]")
    learner = "Adam"
    learning_rate = 0.0001
    epochs = 1
    batch_size = 128
    verbose = 2
    '''
    verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    '''
    losses = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']

    num_users, num_items = len(user_input), len(item_input)
    num_user_context = len(u_context[0])
    num_item_context = len(s_context[0])

    print('Build model')


    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accs = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accs.append(logs.get('acc'))


    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)

    history = LossHistory()

    model = get_Model(num_users, num_items, 10, 37002, 12223, layers, reg_layers)

    model.compile(optimizer=Adam(lr=learning_rate), loss=losses, metrics=['accuracy'])

    print('Start Training')

    for epoch in range(epochs):
        t1 = time()
        hist = model.fit_generator(get_train_instances(train_data), steps_per_epoch=4, epochs=10, verbose=1,
                                   callbacks=[history, board])
        t2 = time()
        print(epoch, t2 - t1)

'''
fit方法：
fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
y：标签，numpy array
batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
epochs：整数，训练的轮数，每个epoch会把训练集轮一遍。
verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）
sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode=’temporal’。
initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况。

'''
