---
title: "tensorflow常用模板"
date: 2023-07-19
categories:
    - "coding"
tags:
    - "DNN"
    - "tensorflow"
    - "transformer"
    - "trajectory"
    - "framework"
draft: true
---

本文记录如何使用tensorflow实现基于transformer的轨迹分类。

在技术实现上包含以下关键细节需要注意：
- 轨迹点如果超出模型序列长度则进行中间截断，如果不足则进行中间补0
- 轨迹数据需要对单条数据进行归一化
- 只使用transformer的encoder部分进行建模，不使用embedding或positional embedding机制，这种方式工程实现简单，但模型可能无法捕获序列前后的区域信息

# 代码实现

首先实现transformer encoder，内部结构为self-attention和feed forward网络。

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,  (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)
        concat_attention = self.dropout(concat_attention)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class GlobalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        attn_output, attention = self.mha(x, k=x, q=x, mask=None)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
```

gps数据中包含“经度、纬度、时间戳”三元素，需要按照时间戳排序之后将一组gps数据合并为一条记录。原始数据从hive中取出。
```python
sdf_gps.withColumn(
    'gps'
    ,F.concat(
        col('t'),lit(',')
        ,col('lat'),lit(','),col('lng'),lit(',')
        ,col('v'),lit(','),col('acc'),lit(','),col('provider')))\
    .groupBy('order_id','dt','label')\
    .agg(F.sort_array(F.collect_list(F.struct('t','gps'))).alias('collected'))\
    .withColumn('gps_sorted',F.col('collected.gps')).drop("collected")\
    .withColumn('gps_sorted',F.concat_ws('+',F.col('gps_sorted')))\
    .write.mode('overwrite').saveAsTable(target_table)
```

可以使用python生成器的形式定义主句转换，从而实现数据的模型输入。

```python
def create_generator(raw_input, seq_len=512):

    def calc_dist(lat_a, lng_a, lat_b, lng_b):
        distance = 999999999
        pi = math.pi
        if lat_a == 0 or lng_a == 0 or lat_b == 0 or lng_b == 0:
            return distance
        ra = 6378138
        c1 = math.pow(math.sin((lat_a / float(1000000) * pi / 180 - lat_b / float(1000000) * pi / 180) / 2), 2)
        c2 = math.pow(math.sin((lng_a / float(1000000) * pi / 180 - lng_b / float(1000000) * pi / 180) / 2), 2)
        x1 = math.cos(lat_a / float(1000000) * pi / 180)
        x2 = math.cos(lat_b / float(1000000) * pi / 180)
        try:
            distance = round(ra * 2 * math.asin(math.sqrt(c1 + x1 * x2 * c2)))
        except:
            pass
        return int(distance)

    def process(df):
        gps_timeline, order_timeline = df[13], df[14]
        rider_fetch_lng, rider_fetch_lat = int(df[3]), int(df[4])
        rider_arrived_lng, rider_arrived_lat = int(df[5]), int(df[6])
        poi_lng, poi_lat = int(df[7]), int(df[8])
        usr_lng, usr_lat = int(df[9]), int(df[10])
        feat_names = ['t', 'lat', 'lng', ]
        
        elements = {fn: [] for fn in feat_names}
        provider_map = {'gps': 0.0, 'iOS': 1.0, 'network': 2.0, 'fail': 3.0}
        ts_idx, pre_t, pre_lat, pre_lng = 1, 0, 0, 0
        checkpoints = [-math.inf] + [float(v) for v in order_timeline.split(',')] + [math.inf]

        for idx, point in enumerate(gps_timeline.split('+')):
            p = point.split(',')
            if idx == 0:
                pre_t, pre_lat, pre_lng = float(p[0]), float(p[1]), float(p[2])

            elements['t'].append(float(p[0]))
            elements['lat'].append(float(p[1]))
            elements['lng'].append(float(p[2]))
            # some fancy transformation
            # ...

        ret = [elements[t] for t in feat_names]

        # 数据归一化
        a = np.vstack(ret)
        a_min = np.transpose(np.tile(np.min(a, axis=1), (a.shape[1], 1)))
        a_max = np.transpose(np.tile(np.max(a, axis=1), (a.shape[1], 1)))
        b = ((a - a_min + 0.01) / (a_max - a_min + 0.01))

        # 截断padding
        l = b[:, :int((min(seq_len, b.shape[1]) + 1) / 2)]
        r = b[:, b.shape[1] - int((min(seq_len, b.shape[1]) + 1) / 2) + 1:]
        b_concat = np.concatenate((l, np.zeros([b.shape[0], max(0, seq_len - l.shape[1] - r.shape[1])]), r), axis=1)
        return np.transpose(b_concat)

    def data_generator():
        for row in raw_input:
            yield process(row), int(row[12])

    return data_generator
```

模型训练主体。

```python
class TraceEncoderNetwork(keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.flatten = layers.Flatten()
        self.logit = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        x = self.flatten(x)
        x = self.logit(x)
        return x


batch_size = 16
seq_len = 512
feature_size = 14
num_layers = 3
num_heads = 2
dff = 256
dropout_rate = 0.01

df = pd.read_csv('trace.tsv',sep='\t')
gen = create_generator(df.to_numpy(), seq_len=seq_len)
ds = tf.data.Dataset.from_generator(
    gen, output_types=(tf.float32, tf.int32), output_shapes=((seq_len, feature_size), ()))

ds_train = ds.take(int(df.shape[0]*0.9))
ds_test = ds.skip(int(df.shape[0]*0.9))
ds_train = ds_train.repeat().shuffle(buffer_size=50).batch(batch_size)
ds_test = ds_test.repeat().shuffle(buffer_size=50).batch(batch_size)

inputs = layers.Input(shape=(seq_len, feature_size))
outputs = TraceEncoderNetwork(num_layers, feature_size, num_heads, dff, dropout_rate)(inputs)
model = Model(inputs=[inputs], outputs=[outputs])
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.AUC()],
)

model.fit(ds_train, epochs=10, validation_data=ds_test, validation_steps=100, steps_per_epoch=1000)
model.save_weights(path_output + 'model')
```

# 参考代码

[Verifying the implementation of Multihead Attention in Transformer](https://stackoverflow.com/questions/67342988/verifying-the-implementation-of-multihead-attention-in-transformer)
