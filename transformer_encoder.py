# Transformer Encoder, implemented from the Transformer paper, applied to IMDB sentiment analysis.
# Apr-Sep 2021 (v6)


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import preprocessing
from tensorflow.keras.datasets import imdb

# GPU memory hack
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# hyperparameters

num_features = 10000 # keeps only the top 10,000 most frequently occuring words
maxlen = 500         # cuts off the text after this number of (eligible) words
model_dim = 64       # transformer (and embedding) dimension
nheads = 8           # number of attention heads
dropout_rate = 0.1   

# Training / test data

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_features)

# Turn the array of lists of integers into a 2D ndarray (samples, maxlen).
# The default is to pre-pad with 0s and remove from the front; so I've specified 'post'.
# help(imdb.load_data) says word index 0 "is usually the padding character."
# So I can use 0 as the pad without it being confusable with a feature 0.

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')


# Model

# inputs is a sequence of word indices (batch, maxlen)
inputs = keras.Input(shape=(maxlen,), name="sequences")

# embed those word indices (batch, maxlen, model_dim)
e = layers.Embedding(num_features, model_dim)(inputs)

# mask is True where inputs contains data and False where it contains a pad (batch, maxlen)
mask = keras.Input(shape=(maxlen,), dtype=tf.bool, name="maskname")


# positional encoding

pos = np.array(range(maxlen))
pel = []
for d in range(model_dim):
    if d % 2 == 0:
        row = np.sin(pos/(10000**(d/model_dim)))
    else:
        row = np.cos(pos/(10000**((d-1)/model_dim)))
    pel.append(row)

# pea has shape (maxlen, model_dim)
pea = np.array(pel).T
pet = tf.convert_to_tensor(pea, dtype=tf.float32)

# I need to add the positional encoding tensor to the embedding, broadcasting pet over the batch.
# There is a scale problem here.
#
# set to no scientific notation so I can examine relative sizes more easily
np.set_printoptions(suppress=True)
#
# I see in the TensorFlow Portuguese to English Transformer tutorial that they multiply the embedding
# by the sqrt of the model dimension.  The justification for this is given in Section 3.4 of the
# AIAYN paper where they explain that the embedding weight matrix is shared with the pre-softmax linear
# transformation in the decoder output.  This makes sense, but it is not applicable here since I don't
# reuse the embedding matrix.  However, without this scaling I have a scale problem, because the default
# initializer for the Embedding layer produces values between +/- 0.05 while the positional encoding
# produces values in the range +/- 1.  It cannot be good that the positional encoding so completely
# dominates the word embedding.  I could initialize the Embedding weights to a +/-1 range , but then
# these weights would be on a completely different scale to other weights in the system, which I think
# is a generally undesirable thing (for reasons related to learning rates, weight regularization, etc.)
# So it seems best to leave the init values where they are and multiply up the output.  To roughly match
# scales I could multiply the embedding output by 20 before adding.  And in fact, with just that change,
# validation accuracy (with an earlier version of this code) went from 0.49 after 1 epoch to 0.59 after
# 1 epoch.  But what if I could learn the best scale factor to use?  By the same argument as above, I'd
# like my new variable initialised to a small value similar to the other weights.  So I implement it as
# follows, in a layer which enables the scale variable to be tracked.  There is an interesting twist to
# this discussion; see the end of this file.

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ScaleLayer, self).__init__()
        # learnable scale parmeter, initialised to 0.05
        self.scale = tf.Variable(0.05)
    def call(self, inputs):
        # the 400 means the layer is effectively initialised to multiply by 20
        return inputs * 400 * self.scale

# apply the learnable scale
epe = ScaleLayer()(e) + pet


# Or, as I thought the first time I saw this, put the positional encoding into embedding dimensions
# of its own so it doesn't corrupt the word embeddings?  That I'll leave for a future experiment.


# apply dropout to produce the transformer input (batch, maxlen, model_dim)
ti = layers.Dropout(dropout_rate)(epe)


# Multi-head Attention layer
# This is necessary because TensorFlow 2.3.0 doesn't have a MultiHeadAttention layer.
# Jul 16 : Now I've read the Portuguese to English transformer tutorial I know this can be done without
# using separate matrices for every head.  This is my original slightly-less-efficient implementation.
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, nheads=4, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.nheads = nheads
    #
    # build is called automatically the first time __call__() is called
    def build(self, input_shape):
        query_shape, value_shape, key_shape = input_shape
        # the last dimension of the query and key tensors should be the same
        assert(query_shape[-1] == key_shape[-1])
        # this is the model_dim
        self.model_dim = query_shape[-1]
        # compute the key (and query and value) dimension for each head
        self.dk  = self.model_dim // self.nheads
        # create query, value and key weights for each attention head
        self.qwl = []
        self.vwl = []
        self.kwl = []
        for h in range(self.nheads):
            qw = self.add_weight(shape=(self.model_dim, self.dk), initializer="random_normal", trainable=True)
            vw = self.add_weight(shape=(self.model_dim, self.dk), initializer="random_normal", trainable=True)
            kw = self.add_weight(shape=(self.model_dim, self.dk), initializer="random_normal", trainable=True)
            self.qwl.append(qw)
            self.vwl.append(vw)
            self.kwl.append(kw)
        # create the output weight matrix to be applied to the concatenated head vectors
        self.wo = self.add_weight(shape=(self.nheads * self.dk, self.model_dim),
                                  initializer="random_normal",
                                  trainable=True)
    #
    #
    def call(self, inputs, mask):
        # inputs should be a list of [query, value, key] tensors, each of shape (batch, Tq/Tv, model_dim)
        query, value, key = inputs
        # mask should be a list of [query, value] masks where False means a pad, (batch, Tq/Tv)
        qmask, vmask = mask
        # loop over heads
        hl = []
        for h in range(self.nheads):
            # use the weight matrices for each head to compute that head's query, value and key tensors
            qi = tf.matmul(query, self.qwl[h])
            vi = tf.matmul(value, self.vwl[h])
            ki = tf.matmul(key,   self.kwl[h])
            # scale the key values by 1/sqrt(dk) in lieu of doing it in the Attention() layer
            ki = ki / np.sqrt(self.dk)
            # apply attention to this head
            hi = layers.Attention()([qi, vi, ki], [qmask, vmask])
            # add this head's output to the list
            hl.append(hi)
        # concatenate the individual head outputs (if necessary) along their last axis
        if self.nheads == 1:
            h = hl[0]
        else:
            h = layers.concatenate(hl)
        # one final linear transformation produces the multi-head-attention output
        mha = tf.matmul(h, self.wo)
        return mha
    # enable serialization (not used)
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({"nheads": self.nheads})
        return config


# Transformer layer
# Expects inputs as (batch, maxlen, model_dim)
def transformer_layer(i, mask):
    # 
    # multi-head self-attention
    sa1 = MultiHeadAttention(nheads)([i,i,i], [mask,mask])
    #
    # dropout
    dr1 = layers.Dropout(dropout_rate)(sa1)
    #
    # add and norm, produces (batch, maxlen, model_dim)
    #
    a1 = i + dr1
    n1 = layers.LayerNormalization()(a1)
    #
    # feed forward
    #
    d1 = layers.Dense(model_dim * 4, activation='relu')(n1)
    d2 = layers.Dense(model_dim)(d1)
    #
    # dropout
    dr2 = layers.Dropout(dropout_rate)(d2)
    #
    # add and norm, produces (batch, maxlen, model_dim)
    #
    a2 = n1 + dr2
    n2 = layers.LayerNormalization()(a2)
    #
    # Transformer layer finished
    return n2

# Use 2 transformer layers, each returning (batch, maxlen, model_dim)
o1 = transformer_layer(ti, mask)
o2 = transformer_layer(o1, mask)

# Use a single position's output to represent each sentence.
# Much as BERT does for its next-sentence prediction.
# Since I'm doing 'post' padding I'll use the 0th position in the sequence as representative.

# shape (batch, model_dim)
rep = o2[:,0,:]

# dense classifier
outputs = layers.Dense(1, activation='sigmoid')(rep)

# define the model
model = keras.Model([inputs, mask], outputs, name="transformer")
model.summary()

# compile
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# create a mask for the training data, in which False means a pad
tmask = x_train != 0

# and fit
history = model.fit((x_train, tmask),
                    y_train,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.2)



# After 1 epoch val_acc is ~0.89, which it does not substantially improve upon out to 5 epochs.
# This is the best transformer system I've trained, but it still only equals my bag-of-words system.
# I guess that's because this isn't a particularly transformer-friendly task.



# Embedding Scaling
#
# After 5 epochs of training with model_dim=64 the scale variable has changed from 0.05 to 0.024
# Other runs with model_dim=64 have produced 0.024 and 0.026
# After 5 epochs of training with model_dim=512 the scale variable has changed from 0.05 to 0.068
# Other runs with model_dim=512 have produced 0.049 and 0.068
# I note that :
# 0.024 * 400 = 9.6  which is fairly close to sqrt(64) = 8
# 0.068 * 400 = 27.2 which is not a million miles from sqrt(512) = 22.6
# It is as if the scale multiplier "wants" to be sqrt(model_dim), or a reasonable approximation thereof.
# What is going on?

# Theory

# It is useful to think through the justification in the Transformer
# paper.  The internal representation (hereafter the ir) is adjusted
# to have stdev 1 at every LayerNormalization.  You want your decoder
# softmax logits to also have stdev about 1 because much larger or
# much smaller logits will cause saturation or uniformity, which are
# unhelpful (*).  In the decoder output linear transformation each ir
# value goes through a set of weights, lets say the average weight
# stdev is x, and these then add to form the logits.  Each logit
# receives model_dim such inputs and they add up.  The variance of
# each contribution will be x.x and assuming the weights are
# uncorrelated the variances will add up such that the total variance
# in each logit will be given by model_dim.x.x The scale in each logit
# is therefore sqrt(model_dim).x As said above we want that to be 1,
# so it is necessary for x to adjust itself during training so that x
# = 1/sqrt(model_dim).  At the other end of the model we use the same
# matrix to produce the embedding.  Here we feed in a one hot vector
# which goes through a single set of weights to produce the embedding.
# The 1-hot vector usually contains a 1 and all zeros, so the scale of
# each Embedding layer output will be x, which we just said was
# 1/sqrt(model_dim).  This is rather small, and by multiplying by
# sqrt(model_dim) we can bring it up to 1.

# (*) https://arxiv.org/pdf/1801.07704v2.pdf suggests a range of 0-10
# is better.

# Notice that we didn't at any time mention the other dimension of the
# embedding matrix... because it doesn't matter.  In fact, the above
# analysis can be boiled down to requiring that o.sqrt(model_dim) = 1
# and e.sm = 1, where o is the output matrix scale, e is the Embedding
# matrix scale, and sm is the scale multiplier.  Sharing the matrix
# ensures that e=o which requires that sm=sqrt(model_dim).  However,
# the matrix does not have to be shared, or even the same shape at
# embedding and output.  In the unshared case o.sqrt(model_dim) must
# still be 1, so o must still be 1/sqrt(model_dim), and we would still
# like e.sm to be roughly 1, but e is no longer related to o.

# In TensorFlow 2.3.0 the output transformation matrix default
# initializer is the Glorot Uniform initializer which initializes the
# weights to a scale of sqrt(6 / (fan_in + fan_out)) which is not too
# far from the 1/sqrt(model_dim) value required above, at least in the
# sigmoid output case studied here.  However, the default Embedding
# initializer is Random Uniform over -0.05 to +0.05 and while this is
# not too far from 1/sqrt(model_dim) for most current values of
# model_dim the fact is that it is not a function of model_dim but a
# fixed scale and this leaves open the question of why the scale
# multiplier doesn't simply choose to be fixed as well.

# If the Embedding weights were tied to the output matrix weights, as
# in the shared case, or if they were Glorot Uniform initialised, then
# the observed scale multipliers might make sense.  But they are not!

# I considered the possibility that the observed scale multipliers
# might be compensating for some other sqrt(model_dim) scaling or
# effect somewhere else in the model.  In my model the scaled
# embedding is fed to the individual attention head matrices in the
# MultiHeadAttention class, but these were initialized (somewhat
# arbitrarily) using the random normal initializer, so there is no
# obvious 1/sqrt(model_dim) to compensate for here.  In fact, given
# their fixed initializations the matrix multiplications might even
# contribute a further sqrt(model_dim) upscaling, reduced by their
# 0.05 default stdev.  Key values are then scaled by 1/sqrt(head_dim)
# where head_dim is a fraction of model_dim.  This is the first
# explicit inverse scaling in the model.  Could it be the case that
# the scale multipliers are compensating for this later inverse
# scaling?  The answer is probably not, because an earlier version of
# my code omitted that inverse scaling due to an oversight, and still
# produced similar scale multipliers.  After the MultiHead-Attention
# comes a LayerNormalization, which sets the scale to 1, so the
# learned scale multiplier cannot be due to later components.

# I decided to look at the stdev of the Embedding weights after
# training and combine that with the learned scale multiple.  In the
# model_dim=64 case the Embedding stdev=0.07589795 while the scale
# variable is 0.022839583.  Including the 400 multiplier from the
# ScaleLayer this comes to 0.6934.  In the model_dim=512 case the
# Embedding stdev=0.034520503 while the scale variable is 0.06761398,
# which comes to 0.9336.  Now this is very interesting, because what
# we find is that the scaled representation has a pretty similar 
# scale in both cases, say ~0.8, but that the model_dim=64 case has
# much larger Embedding weights than the model_dim=512 case, which is
# compensated for by learning a smaller scale variable.  Given that
# the scale presented to the next component in the model is about the
# same in both cases, it does not seem likely that the scale is being
# prepared for some later sqrt(model_dim) inverse scaling, at least
# not when the positional encoding is present, see below.

# At this point I decided to gather more data:

# Without positional encoding (pet):

# Without pet, with rmsprop, model_dim 64, and with learned scale :
# Epoch 1/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.3679 - acc: 0.8357 - val_loss: 0.2656 - val_acc: 0.8980
# Epoch 2/5
# 625/625 [=..=] - 73s 117ms/step - loss: 0.2321 - acc: 0.9128 - val_loss: 0.2940 - val_acc: 0.8914
# Epoch 3/5
# 625/625 [=..=] - 73s 118ms/step - loss: 0.1940 - acc: 0.9290 - val_loss: 0.3070 - val_acc: 0.8900
# Epoch 4/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.1732 - acc: 0.9384 - val_loss: 0.2804 - val_acc: 0.8882
# Epoch 5/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.1593 - acc: 0.9437 - val_loss: 0.3232 - val_acc: 0.8868

# In this case the scale layer weight learns to be 0.01698722 which * 400 = 6.7949
# The Embedding weight stdev in this case is 0.07000576.

# Interestingly, the earlier version of the code that omitted the 1/sqrt(head_dim)
# Transformer scaling found the scale layer weight learned to be 0.0038878953 which
# * 400 = 1.5552, which is not even close to sqrt(64).  So, in the absence of both
# the positional encoding and the Transformer scaling the scale layer is not really
# used.  The Embedding weight stdev in this case was 0.06638129.

# Without pet, with rmsprop, model_dim 64, but without learned scale :

# Epoch 1/5
# 625/625 [=..=] - 73s 117ms/step - loss: 0.3692 - acc: 0.8370 - val_loss: 0.2713 - val_acc: 0.8902
# Epoch 2/5
# 625/625 [=..=] - 73s 117ms/step - loss: 0.2287 - acc: 0.9140 - val_loss: 0.2686 - val_acc: 0.8914
# Epoch 3/5
# 625/625 [=..=] - 73s 117ms/step - loss: 0.1943 - acc: 0.9278 - val_loss: 0.2994 - val_acc: 0.8832
# Epoch 4/5
# 625/625 [=..=] - 73s 118ms/step - loss: 0.1734 - acc: 0.9385 - val_loss: 0.2741 - val_acc: 0.8902
# Epoch 5/5
# 625/625 [=..=] - 74s 119ms/step - loss: 0.1615 - acc: 0.9433 - val_loss: 0.2953 - val_acc: 0.8912

# Embedding weight stdev : 0.07185463
# The earlier version of the code that omitted the 1/sqrt(head_dim) Transformer scaling
# also learned well.

# With positional encoding : 

# With pet, with rmsprop, model_dim 64, and with learned scale :

# Epoch 1/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.3861 - acc: 0.8214 - val_loss: 0.2805 - val_acc: 0.8878
# Epoch 2/5
# 625/625 [=..=] - 73s 117ms/step - loss: 0.2284 - acc: 0.9125 - val_loss: 0.3362 - val_acc: 0.8934
# Epoch 3/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.1945 - acc: 0.9258 - val_loss: 0.2728 - val_acc: 0.8926
# Epoch 4/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.1734 - acc: 0.9370 - val_loss: 0.3153 - val_acc: 0.8920
# Epoch 5/5
# 625/625 [=..=] - 74s 119ms/step - loss: 0.1580 - acc: 0.9430 - val_loss: 0.2936 - val_acc: 0.8904

# Scale layer weight : 0.021144018 which * 400 = 8.457607
# Embedding weight stdev : 0.07539754

# The earlier version of the code that omitted the 1/sqrt(head_dim) Transformer scaling obtained
# scale layer weights of 0.017, 0.019 and 0.021 on different runs, corresponding to scale multipliers
# of 6.8, 7.6 and 8.4 respectively.

# With pet, with rmsprop, model_dim 64, but without learned scale :

# Epoch 1/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.6200 - acc: 0.6289 - val_loss: 0.3657 - val_acc: 0.8404
# Epoch 2/5
# 625/625 [=..=] - 73s 117ms/step - loss: 0.3505 - acc: 0.8514 - val_loss: 0.2969 - val_acc: 0.8736
# Epoch 3/5
# 625/625 [=..=] - 73s 117ms/step - loss: 0.2725 - acc: 0.8895 - val_loss: 0.2759 - val_acc: 0.8944
# Epoch 4/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.2376 - acc: 0.9089 - val_loss: 0.3503 - val_acc: 0.8554
# Epoch 5/5
# 625/625 [=..=] - 74s 119ms/step - loss: 0.2125 - acc: 0.9205 - val_loss: 0.2874 - val_acc: 0.8936

# This version can be seen to converge more slowly.
# In this case the Embedding weights learned to be slightly larger, with a stdev of 0.08847155.

# The earlier version of the code that omitted the 1/sqrt(head_dim) Transformer scaling also converged
# more slowly, obtaining an Embedding weight stdev of 0.08820149.


# The results are clear : either of the positional encoding or the Transformer 1/sqrt(head_dim) scaling
# is sufficient to cause the learned scale layer to be approximately sqrt(model_dim).  When both are
# present the learned scale layer will still approximate sqrt(model_dim).  When the positional encoding
# is present without a learned scale layer the model will converge more slowly, both with and without
# 1/sqrt(head_dim) Transformer scaling.  However, without a positional encoding the model seems to
# learn just fine without a learned scale layer both with or without a Transformer scaling.  So the
# learned scale layer is only needed when a positional encoding is present.


# With pet, with rmsprop, model_dim 64, and with a fixed sqrt(model_dim) scale :
# Epoch 1/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.4093 - acc: 0.7958 - val_loss: 0.2777 - val_acc: 0.8916
# Epoch 2/5
# 625/625 [=..=] - 73s 117ms/step - loss: 0.2319 - acc: 0.9095 - val_loss: 0.2661 - val_acc: 0.8900
# Epoch 3/5
# 625/625 [=..=] - 73s 118ms/step - loss: 0.1975 - acc: 0.9242 - val_loss: 0.2999 - val_acc: 0.8830
# Epoch 4/5
# 625/625 [=..=] - 74s 118ms/step - loss: 0.1725 - acc: 0.9371 - val_loss: 0.2936 - val_acc: 0.8942
# Epoch 5/5
# 625/625 [=..=] - 74s 119ms/step - loss: 0.1537 - acc: 0.9442 - val_loss: 0.3704 - val_acc: 0.8900

# Comparing to the corresponding results with the learned scale above
# we see there is no difference between learned and fixed scales.


# Finally I tried training with Adam instead of RMSProp:

# With pet, with Adam, model_dim 64, and with learned scale :
# The scale layer learns to be 0.041741468 which * 400 is 16.696587, which is 109% larger than sqrt(64).
# The Embedding stdev was 0.06704463, meaning the scaled embedding had stdev 1.1194

# With pet, with Adam, model_dim 512, and with learned scale :
# The scale layer learns to be 0.045255862 which * 400 is 18.102345, close to sqrt(512).
# The Embedding stdev was 0.049073763, meaning the scaled embedding had stdev 0.8884

# With Adam the learned scale multiples may be different.
# However, it must be noted that Adam reaches lower training losses, and higher accuracies, than RMSProp.
# Rerunning the model_dim 64 experiment and stopping after 3 epochs, when training loss is comparable
# to that of RMSProp at 5 epochs, yields an even larger scale layer of 0.04297345, which * 400 is 17.18938

# It appears that the Embedding weight stdev is strongly determined by model_dim.
# 64 dim models typically have weight stdev of about 0.07, and 512 dim models of 0.03-5
# I've tried to figure out why this happens using backprop formulae but I haven't convinced myself yet.


# Conclusion

# In Transformer encoders a positional encoding on a scale of -1 to +1
# is typically added to the input Embedding layer outputs.  In systems
# where the Embedding layer matrix is shared with the decoder output
# linear transformation matrix theory suggests that it is a good idea
# to multiply the Embedding layer outputs by sqrt(model_dim).  In
# systems which do not share the Embedding layer matrix, experiment
# shows it is still a good idea to scale the Embedding layer outputs
# in order to put them on a similar scale to the positional encoding.
# Not scaling hurts model convergence.  In experiments learned scale
# multiples are often close to sqrt(model_dim), and learned scale
# multiples and fixed scale multiples of this size perform similarly.
# The exact size of learned scale multiples may be optimizer
# dependent.

