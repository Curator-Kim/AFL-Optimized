import os
import tensorflow
import numpy as np


input_file=''

if not os.path.isfile(args.input_file):
    print("[-]" + args.input_file + "does not exists.")
    exit(0)
elif not args.input_file.lower().endwith('.npz'):
    print("[-]" + args.input_file + " is an invalid file.")
    exit(0)
input_file = args.input_file#加载文件并读出数据
np_data = np.load(input_file)
x_dataset = np_data['x']
y_dataset = np_data['y']
print(x_dataset.shape, y_dataset.shape)
num, timesteps, num_of_features = x_dataset.shape

#对给定的文件进行宇节化
def vectorize(input_file):
    shape = np.zeros(shape=(max_filesize, bytesize))
    with open(input_file, "rb") as f:
        byte = f.read(1)
        byte_pos = 0
        while byte:
            bits = bin(int.from_bytes(byte, byteorder="big"))[2:].zfill(8)
            for n, bit in enumerate(bits):
                if bit =='1':
                    shape[byte_pos, n] = 1
            byte = f.read(1)
            byte_pos += 1
        return shape
#对文件编码后的字节文件进行向量化
def vectorize_bytearray(byte_arr):
    shape = np.zeros(shape=(max_filesize, bytesize))#将宇节数组位放入numpy数组
    byte_pos = 0
    for byte in byte_arr:
        bits = bin(byte)[2:].zfill(8)
        for n, bit in enumerate(bits):
            if bit =='1':
                shape[byte_pos,n] = 1
        byte_pos += 1
    return shape

#使用0填充文件字节数组
def padding(x, y):
    x_ba = bytearray(open(x, "rb").read())
    y_ba = bytearray(open(y, "rb").read())
    #检查较大的文件并填充其大小
    if (len(x_ba)> en(y_ba)):
        size = len(x_ba)
        y_ba = y_ba.ljust(size, b' x00')
    elif (len(x_ba) <len(y_ba)):
        size = len(y_ba)
        x_ba = x_ba.ljust(size, b' x00')
    return x_ba,y_ba

#x，y进行分割
def get_segments(filesize):
    segments= filesize # 最大的文件
    if (filesize % max_filesize) > 0:
        segments += 1
    return segments

#将样本存入文件
def saveToFile(sample_x, sample_y, output_file):
    np.savez_compressed(output_file, x=sample_x,  y=sample_y)
    print("[+] Dataset stored in: " + output_file + ".npz")
    return 0

train = ""
val = ""
test = ""
#加载训练样本
training_dataset = np.load(train)
val_dataset = np.Load(val)
test_dataset = np.load(test)
x_train = training_dataset['x'][:5120]
y_train = training_dataset['y'][:5120]
x_val = val_dataset['x'][:992]
y_val = val_dataset['y'][:992]
x_test = test_dataset['x']
y_test = test_dataset['y']
print("No. of seed files collected: 180")
print("Data collection of XY simulated to 2% sampling rate, Dataset split: 8/1/1")
samples, timesteps, chunksize = x_train.shape
print("No. of training amples: " + str(samples) + ", Mo. of timesteps: " + str(timesteps) + ", Chunksize: " + str(chunksize))
print("Training shape:", x_train.shape, y_train.shape)
print("Validation shape: ", x_val.shape, y_val.shape)
print("Test shape:",x_test.shape, y_test.shape)

#搭建双向LSTM网络结构
model = Sequential()
model.add(Bidirectional(LSTM(64, input_shape=(2560, 64), return_sequences=True)))
model.summary()
adam = Adam(lr=0 .00005)
model.compile(optimizer=adam, loss='mean_absolute_error')
# 训练模型
history = model.fit(x_train,y_train,
epochs=100,
batch_size=32,
validation_data=(x_val, y_val),
shuffle=False)

#定义seq2seq的编码解
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs,state_h,state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#定Xseg2seg的编码器
decoder_inputs = Input(shape=(None,num_decoder_tokens))
decoder_lstm = LSTM(latent_dim,return_sequences=True)
decoder_outputs = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#定义seq2seq模型
model = Model([encoder_inputs, decoder_inputs],decoder_outputs)

#模型训练
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data],decoder_target_data,batch_size=batch_size, epochs=epochs,validation_split=0.2)

#建编码器
class Encoder(keras.Model):
    def __init__(self,vocab_size, embedding_dim, hidden_units):
        super(Encoder, self).__init__()
        # Embedding Layer
        self.embedding = Embedding(vocab_size,embedding_dim, mask_zero=True)
        # Encode LSTM Layer
        self.encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True. name="encode_lstm")
    def call(self, inputs):
        encoder_embed = self.embedding(inputs)
        encoder_outputs, state_h, state_c = self.encoder_lstm(encoder_embed)
        return encoder_outputs, state_h, state_c

#搭建解码器，并结合attention注意力机制
class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Decoder, self).__init__()
        # Embedding Layer
        self.embedding = Embedding(vocab_size,embedding_dim, mask_zero=True)
        # Decode LSTM Layer
        self.decoder_lstm = LSTM(hidden_units, return_seguences=True, return_state=True, name="decode_lstm")
        #Attention Layer
        self.attention = Attention()
    def call(self, enc_outputs, dec_inputs, states inputs):
        decoder_embed = self.embeddina(dec_inputs)
        dec_outputs, dec_state_h, dec_state_c = self.decoder_lstm(decoder_embed, initial_state=states_inputs)
        attention_output = self.attention([dec_outputs, enc_outputs])
        return attention_output, dec_state_h, dec_state_c

#结合编码器和解码器
def Seq2Seq(maxlen, embedding_dim, hidden_units, vocab_size):
    # Input Layer
    encoder_inputs = Input(shape=(maxlen,), name="encode_input")
    decoder_inputs = Input(shape=(None,),name="decode_input")
    # Encoder Lauer
    encoder = Encoder(vocab_size, embedding_dim,hidden_units)
    enc_outputs, enc_state_h, enc_state_c = encoder(encoder_inputs)
    dec_states_inputs = [enc_state_h, enc_state_c]
    #Decoder Layer
    decoder = Decoder(vocab_size, embedding_dim, hidden_units)
    attention_output, dec_state_h, dec_state_c = decoder(enc_outputs, decoder_inputs, dec_states_inputs)
    # Dense Layer
    dense_outputs = Dense(vocab_size, activation='softmax', name="dense")(attention_output)
    # seg2seq model
    modcl = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs)
    return model

#训练模型
loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam')
model.fit([source_input_ids, target_input_ids], target_output_ids, batch_size=batch_size, epochs=epochs, validation_split=val_rate)

#加载模型
model = load_model("")
#设置文件大小
max_filesize = 20480 #30720，20480
#设置字节大小
bytesize = 8
#预测文件每个位置的重要程度
def query_model(seed):
    predictions = model.predict(checkFile(seed))
    bytemask = get_bytemask(predictions)
    return bytemask

def xor(x, y):
    xor = np.zeros(shape=(max_filesize, bytesize))
    if (x.size == y.size):
        max_size, col_size = x.shape
        for col in range(max_size):
            for bit in range(col_size):
                if x[col][bit] != y[col][bit]:
                    xor[col][bit] = 1.
    return xor