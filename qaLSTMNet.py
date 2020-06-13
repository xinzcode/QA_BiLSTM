import tensorflow as tf


class QaLSTMNet(object):

    def __init__(self, batchSize, unrollSteps, embeddings, embeddingSize, rnnSize, margin):
        self.batchSize = batchSize
        self.unrollSteps = unrollSteps
        self.embeddings = embeddings
        self.embeddingSize = embeddingSize
        self.rnnSize = rnnSize
        self.margin = margin

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        self.inputQuestions = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTrueAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputFalseAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTestQuestions = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTestAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        # print("设置词向量层")
        # 设置word embedding层
        # 使用with，也就是python的上下文管理器，执行会会自动关闭会话，释放内存，简单高效！
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            tfEmbedding = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            questions = tf.nn.embedding_lookup(tfEmbedding, self.inputQuestions)
            # 选取一个张量里面索引对应的元素
            trueAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTrueAnswers)
            falseAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputFalseAnswers)
            testQuestions = tf.nn.embedding_lookup(tfEmbedding, self.inputTestQuestions)
            testAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTestAnswers)
        # print("建立BiLSTM网络1")
        # 建立LSTM网络
        with tf.variable_scope("LSTM_scope", reuse=None):
            question1 = self.biLSTMCell(questions, self.rnnSize)
            question2 = tf.nn.tanh(self.max_pooling(question1))
        # print("建立BiLSTM网络2")
        with tf.variable_scope("LSTM_scope", reuse=True):
            trueAnswer1 = self.biLSTMCell(trueAnswers, self.rnnSize)
            trueAnswer2 = tf.nn.tanh(self.max_pooling(trueAnswer1))
            falseAnswer1 = self.biLSTMCell(falseAnswers, self.rnnSize)
            falseAnswer2 = tf.nn.tanh(self.max_pooling(falseAnswer1))

            testQuestion1 = self.biLSTMCell(testQuestions, self.rnnSize)
            testQuestion2 = tf.nn.tanh(self.max_pooling(testQuestion1))
            testAnswer1 = self.biLSTMCell(testAnswers, self.rnnSize)
            testAnswer2 = tf.nn.tanh(self.max_pooling(testAnswer1))

        self.trueCosSim = self.getCosineSimilarity(question2, trueAnswer2)
        self.falseCosSim = self.getCosineSimilarity(question2, falseAnswer2)
        self.loss = self.getLoss(self.trueCosSim, self.falseCosSim, self.margin)
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary])

        self.result = self.getCosineSimilarity(testQuestion2, testAnswer2)

    @staticmethod
    def biLSTMCell(x, hiddenSize):
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

    @staticmethod
    def getCosineSimilarity(q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.div(mul, tf.multiply(q1, a1))
        return cosSim

    @staticmethod
    def max_pooling(lstm_out):
        height = int(lstm_out.get_shape()[1])
        width = int(lstm_out.get_shape()[2])
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    @staticmethod
    def getLoss(trueCosSim, falseCosSim, margin):
        zero = tf.fill(tf.shape(trueCosSim), 0.0)
        tfMargin = tf.fill(tf.shape(trueCosSim), margin)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(trueCosSim, falseCosSim)))
            #  损失函数
            loss = tf.reduce_sum(losses)
            # 用于计算张量tensor沿着某一维度的和，可以在求和后降维。
        return loss












#
# import tensorflow as tf
#
#
# class QaLSTMNet(object):
#
#     def __init__(self, batchSize, unrollSteps, embeddings, embeddingSize, rnnSize, margin):
#         self.batchSize = batchSize
#         self.unrollSteps = unrollSteps
#         self.embeddings = embeddings
#         self.embeddingSize = embeddingSize
#         self.rnnSize = rnnSize
#         self.margin = margin
#
#         self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
#         self.inputQuestions = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
#         self.inputTrueAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
#         self.inputFalseAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
#         self.inputTestQuestions = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
#         self.inputTestAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
#         # print("设置词向量层")
#         # 设置word embedding层
#         # 使用with，也就是python的上下文管理器，执行会会自动关闭会话，释放内存，简单高效！
#         with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
#             tfEmbedding = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
#             # print("----tfEmbedding.shape = ",tfEmbedding.shape)
#             questions = tf.nn.embedding_lookup(tfEmbedding, self.inputQuestions)
#             # print("----questions.shape = ",questions.shape)
#             trueAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTrueAnswers)
#             # print("----trueAnswers.shape = ",trueAnswers.shape)
#             falseAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputFalseAnswers)
#             # print("----falseAnswers.shape = ",falseAnswers.shape)
#             testQuestions = tf.nn.embedding_lookup(tfEmbedding, self.inputTestQuestions)
#             testAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTestAnswers)
#         # print("建立BiLSTM网络1")
#         # 建立LSTM网络
#         with tf.variable_scope("LSTM_scope", reuse=None):
#             question1 = self.biLSTMCell(questions, self.rnnSize)
#             # print("~~~~~question1.shape = ",question1.shape)
#             question2 = tf.nn.tanh(self.max_pooling(question1))
#             # print("~~~~~question2.shape = ",question2.shape)
#         # print("建立BiLSTM网络2")
#         with tf.variable_scope("LSTM_scope", reuse=True):
#             trueAnswer1 = self.biLSTMCell(trueAnswers, self.rnnSize)
#             # print("~~~~~trueAnswer1.shape = ",trueAnswer1.shape)
#             trueAnswer2 = tf.nn.tanh(self.max_pooling(trueAnswer1))
#             # print("~~~~~trueAnswer2.shape = ",trueAnswer2.shape)
#             falseAnswer1 = self.biLSTMCell(falseAnswers, self.rnnSize)
#             # print("~~~~~falseAnswer1.shape = ",falseAnswer1.shape)
#             falseAnswer2 = tf.nn.tanh(self.max_pooling(falseAnswer1))
#             # print("~~~~~falseAnswer2.shape = ",falseAnswer2.shape)
#
#             testQuestion1 = self.biLSTMCell(testQuestions, self.rnnSize)
#             testQuestion2 = tf.nn.tanh(self.max_pooling(testQuestion1))
#             testAnswer1 = self.biLSTMCell(testAnswers, self.rnnSize)
#             testAnswer2 = tf.nn.tanh(self.max_pooling(testAnswer1))
#
#         self.trueCosSim = self.getCosineSimilarity(question2, trueAnswer2)
#         self.falseCosSim = self.getCosineSimilarity(question2, falseAnswer2)
#         self.loss = self.getLoss(self.trueCosSim, self.falseCosSim, self.margin)
#         # Summaries for loss and accuracy
#         loss_summary = tf.summary.scalar("loss", self.loss)
#         # Dev summaries
#         self.dev_summary_op = tf.summary.merge([loss_summary])
#
#         self.result = self.getCosineSimilarity(testQuestion2, testAnswer2)
#
#     @staticmethod
#     def biLSTMCell(x, hiddenSize):
#         input_x = tf.transpose(x, [1, 0, 2])# 0 1 2 变成1 0 2，最外面两个维度转置
#         input_x = tf.unstack(input_x)   #多维变成低维，默认axio = 0，最外层的维度
#         lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
#         lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
#         output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
#         output = tf.stack(output)       #一维变成多维
#         output = tf.transpose(output, [1, 0, 2])
#         return output
#
#     @staticmethod
#     def getCosineSimilarity(q, a):
#         q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
#         a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
#         mul = tf.reduce_sum(tf.multiply(q, a), 1)
#         cosSim = tf.div(mul, tf.multiply(q1, a1))
#         return cosSim
#
#     @staticmethod
#     def max_pooling(lstm_out):
#         height = int(lstm_out.get_shape()[1])
#         width = int(lstm_out.get_shape()[2])
#         lstm_out = tf.expand_dims(lstm_out, -1)
#         # 增加一个维度，如2*3变成1*2*3，第二个参数是增加的维度的位置，是1*2*3还是2*1*3还是2*3*1
#         output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
#         output = tf.reshape(output, [-1, width])
#         return output
#
#     @staticmethod
#     def getLoss(trueCosSim, falseCosSim, margin):
#         zero = tf.fill(tf.shape(trueCosSim), 0.0)
#         tfMargin = tf.fill(tf.shape(trueCosSim), margin)
#         with tf.name_scope("loss"):
#             losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(trueCosSim, falseCosSim)))
#             #  损失函数
#             loss = tf.reduce_sum(losses)
#             # 用于计算张量tensor沿着某一维度的和，可以在求和后降维。
#         return loss
