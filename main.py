"""
description: 运行主程序，不含训练代码，运行时间约一分钟
"""
import os
import json
import tensorflow as tf
from tkinter import *
import qaData
from qaLSTMNet import QaLSTMNet
from tkinter.scrolledtext import ScrolledText

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # 定义参数
    trainingFile = "./data/train_data_sample.json"
    testingFile = "./data/my_data_sample.json"
    resultFile = "predictRst.score"
    saveFile = "newModel/savedModel"
    trainedModel = "./newModel/savedModel"
    embeddingFile = "./zhwiki/zhwiki_2017_03.sg_50d.word2vec"

    embeddingSize = 50  # 词向量的维度

    dropout = 1.0       # 保留全部结果，训练时用
    learningRate = 0.4  # 学习速度
    lrDownRate = 0.5    # 学习速度下降速度
    lrDownCount = 4     # 学习速度下降次数
    epochs = 20         # 每次学习速度指数下降之前执行的完整epoch次数
    batchSize = 8       # 每一批次处理的问题个数
    rnnSize = 100       # 隐含层神经元的个数
    margin = 0.1        # 余弦边界值 来对计算出的正负样本的语义相似度进行评判。
    unrollSteps = 100   # 句子中的最大词汇数目
    max_grad_norm = 5   # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小

    allow_soft_placement = True  # 那么当运行设备不满足要求时，允许自动分配GPU或者CPU。
    cpuDevice = "/cpu:0"    # CPU不区分设备号，统一使用 /cpu:0

    # 读取所有词向量和对应索引
    print("正在加载词向量")
    embedding, word2idx = qaData.loadEmbedding(embeddingFile)
    print("词向量加载完成")
    with tf.Graph().as_default(), tf.device(cpuDevice):
        session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement)
        with tf.Session(config=session_conf).as_default() as sess:
            globalStep = tf.Variable(0, name="globle_step", trainable=False)
            print("实例化网络")
            lstm = QaLSTMNet(batchSize, unrollSteps, embedding, embeddingSize, rnnSize, margin)
            print("实例化结束")
            saver = tf.train.Saver()
            # 创建一个tf.train.Saver对象
            saver.restore(sess, trainedModel)
            # 使用已保存的模型

            print("正在加载知识库")
            aTest, aIdTest = qaData.loadtestjsonData(testingFile, word2idx, unrollSteps)
            print("知识库加载完成")
            root = Tk()
            root.title("校园问答系统")
            root.geometry('800x800')
            l1 = Label(root, text="输入你的问题：")
            l1.pack()
            xls_text = StringVar()
            xls = Entry(root, textvariable=xls_text, width=50)
            xls_text.set("")
            xls.pack()

            def on_click():
                q = xls_text.get()
                print(q)
                qTest = qaData.loadquestion(q, testingFile, word2idx, unrollSteps)
                #  返回的是答案个问句对应的词向量索引
                with open(resultFile, 'w') as file:
                    # 返回的元祖数组，然后依次遍历里面的每一对值
                    for question, answer in qaData.testingBatchIter(qTest, aTest, batchSize):
                        # 来赋值的，格式为字典型
                        feed_dict = {
                            lstm.inputTestQuestions: question,
                            lstm.inputTestAnswers: answer,
                            lstm.keep_prob: dropout
                        }
                        _, scores = sess.run([globalStep, lstm.result], feed_dict)

                        best = b1 = 0.0
                        i = n = n1 = 0
                        for score in scores:
                            file.write("%.9f" % score + '\n')
                            i += 1
                            if score > best:
                                b1 = best
                                best = score
                                n1 = n
                                n = i
                        print(best)
                        print(n)
                        print(b1)
                        print(n1)
                        with open(testingFile, mode="r", encoding="utf-8") as rf:
                            json_d = json.load(rf)
                            if best <= 0.6:
                                EditText.insert('end', "您的问题：" + q)
                                EditText.insert(INSERT, '\n')
                                EditText.insert('end', "暂无您所询问的相关信息，请换个问题吧。")
                                EditText.insert(INSERT, '\n')
                                EditText.insert(INSERT, '\n')
                            elif b1 > 0.6:
                                for block in json_d:
                                    for ans in block['passages']:
                                        if n == int(ans['passage_id']):
                                            EditText.insert('end', "您的问题：" + q)
                                            EditText.insert(INSERT, '\n')
                                            EditText.insert('end', "最佳答案：" + ans['content'])
                                            EditText.insert(INSERT, '\n')
                                    for ans in block['passages']:
                                        if n1 == int(ans['passage_id']):
                                            EditText.insert('end', "相关信息：" + ans['content'])
                                            EditText.insert(INSERT, '\n')
                                            EditText.insert(INSERT, '\n')
                            else:
                                for block in json_d:
                                    for ans in block['passages']:
                                        if n == int(ans['passage_id']):
                                            EditText.insert('end', "您的问题：" + q)
                                            EditText.insert(INSERT, '\n')
                                            EditText.insert('end', "最佳答案：" + ans['content'])
                                            EditText.insert(INSERT, '\n')
                                            EditText.insert(INSERT, '\n')
            Button(root, text="查询", command=on_click).pack()
            EditText = ScrolledText(root, width=80, height=50)
            EditText.pack()
            root.mainloop()
    print("程序结束")


# """
# description: 运行主程序，含训练代码（CPU版），运行时间约三分钟，答案不含相关信息
# """
# import os
# import time
# import json
# import tensorflow as tf
# from tkinter import *
# import qaData
# from qaLSTMNet import QaLSTMNet
# from tkinter.scrolledtext import ScrolledText
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
#
# def restore():
#     try:
#         print("正在加载模型")
#         saver.restore(sess, trainedModel)
#     except Exception as e:
#         print(e)
#         print("加载模型失败，重新开始训练")
#         train()
#
#
# def train():
#     print("重新训练，请保证计算机拥有至少8G空闲内存与2G空闲显存")
#     # 准备训练数据
#     print("正在准备训练数据，大约需要五分钟...")
#     qTrain, aTrain, lTrain, qIdTrain, aIdTrain = qaData.loadjsonData(trainingFile, word2idx, unrollSteps, True)
#
#     print("训练数据准备完毕")
#     tqs, tta, tfa = [], [], []
#     for question, trueAnswer, falseAnswer in qaData.trainingBatchIter(qTrain , aTrain ,lTrain , qIdTrain ,batchSize):
#         tqs.append(question), tta.append(trueAnswer), tfa.append(falseAnswer)
#
#     print("训练数据加载完成！")
#     # 开始训练
#     print("开始训练，全部训练过程大约需要12小时")
#     sess.run(tf.global_variables_initializer())
#     lr = learningRate  # 引入局部变量，防止shadow name
#     for i in range(lrDownCount):
#         optimizer = tf.train.GradientDescentOptimizer(lr)
#         optimizer.apply_gradients(zip(grads, tvars))
#         trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
#         for epoch in range(epochs):
#             print("epoch",epoch)
#             for question, trueAnswer, falseAnswer in zip(tqs, tta, tfa):
#                 # print("question.shape = ", question.shape)
#                 # print("trueAnswer.shape = ", trueAnswer.shape)
#                 # print("falseAnswer.shape = ", falseAnswer.shape)
#                 startTime = time.time()
#                 feed_dict = {
#                     lstm.inputQuestions: question,
#                     lstm.inputTrueAnswers: trueAnswer,
#                     lstm.inputFalseAnswers: falseAnswer,
#                     lstm.keep_prob: dropout
#                 }
#                 # summary_val = sess.run(lstm.dev_summary_op, feed_dict)
#                 sess.run(trainOp, feed_dict)
#                 step = sess.run(globalStep, feed_dict)
#                 sess.run(lstm.trueCosSim, feed_dict)
#                 sess.run(lstm.falseCosSim, feed_dict)
#                 loss = sess.run(lstm.loss, feed_dict)
#                 timeUsed = time.time() - startTime
#                 print("step:", step, "loss:", loss, "time:", timeUsed)
#             saver.save(sess, saveFile)
#         lr *= lrDownRate
#
#
# if __name__ == '__main__':
#     # 定义参数
#     trainingFile = "./data/train_data_sample.json"
#     testingFile = "./data/my_data_sample.json"
#     resultFile = "predictRst.score"
#     saveFile = "newModel/savedModel"
#     trainedModel = "./newModel/savedModel"
#     embeddingFile = "./zhwiki/zhwiki_2017_03.sg_50d.word2vec"
#     embeddingSize = 50  # 词向量的维度
#
#     dropout = 1.0
#     learningRate = 0.4  # 学习速度
#     lrDownRate = 0.5  # 学习速度下降速度
#     lrDownCount = 4  # 学习速度下降次数
#     epochs = 20  # 每次学习速度指数下降之前执行的完整epoch次数
#     batchSize = 8  # 每一批次处理的问题个数
#
#     rnnSize = 100  # LSTM cell中隐藏层神经元的个数
#     margin = 0.1  # 余弦边界值 来对计算出的正负样本的语义相似度进行评判。
#
#     unrollSteps = 100  # 句子中的最大词汇数目
#     max_grad_norm = 5  # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
#
#     allow_soft_placement = True  # Allow device soft device placement 有时候不同的设备的cpu和gpu是不同的,那么当运行设备不满足要求时，会自动分配GPU或者CPU。
#     # gpuMemUsage = 0.75  # 显存最大使用率
#     # gpuDevice = "/gpu:0"  # GPU设备名
#     cpuDevice = "/cpu:0"    # CPU不区分设备号，统一使用 /cpu:0
#
#     # 读取所有词向量和对应索引
#     print("正在加载词向量")
#     embedding, word2idx = qaData.loadEmbedding(embeddingFile)
#
#     # 配置TensorFlow
#     with tf.Graph().as_default(), tf.device(cpuDevice):
#         # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpuMemUsage)
#         # session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement, gpu_options=gpu_options)
#         session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement)
#         with tf.Session(config=session_conf).as_default() as sess:
#             # 加载LSTM网络
#             print("正在加载BiLSTM网络，大约需要三分钟...")
#             globalStep = tf.Variable(0, name="globle_step", trainable=False)
#             # tf.Variable为tensorflow变量声明函数 trainable默认为True，可以后期被算法优化的。如果不想该变量被优化，改为False。
#             lstm = QaLSTMNet(batchSize, unrollSteps, embedding, embeddingSize, rnnSize, margin)
#             # 实例化一个网络结构对象
#             tvars = tf.trainable_variables()
#             # trainable_variables()函数可以也仅可以查看可训练的变量
#             grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), max_grad_norm)
#             # 通过权重梯度的总和的比率来截取多个张量的值  第一个参数为梯度张量，第二个梯度为截取的比率 返回截取过的梯度张量和一个所有张量的全局范数。
#             saver = tf.train.Saver()
#             # 保存模型
#             print("加载完成！")
#
#             # 加载模型或训练模型
#             if os.path.exists(trainedModel + '.index'):
#                 while True:
#                     choice = input("找到已经训练好的模型，是否载入（y/n）")
#                     if choice.strip().lower() == 'y':
#                         restore()
#                         break
#                     elif choice.strip().lower() == 'n':
#                         train()
#                         break
#                     else:
#                         print("无效的输入！\n")
#             else:
#                 train()
#
#             print("正在加载知识库")
#             aTest, aIdTest = qaData.loadtestjsonData(testingFile, word2idx, unrollSteps)
#
#             root = Tk()
#             root.title("校园问答系统")
#             root.geometry('800x800')
#
#             l1 = Label(root, text="输入你的问题：")
#             l1.pack()
#             xls_text = StringVar()
#             xls = Entry(root, textvariable=xls_text, width=50)
#             xls_text.set("")
#             xls.pack()
#
#             def on_click():
#                 q = xls_text.get()
#                 print(q)
#                 qTest = qaData.loadquestion(q, testingFile, word2idx, unrollSteps)
#                 #  返回的是答案个问句对应的词向量索引
#                 with open(resultFile, 'w') as file:
#                     # 返回的元祖数组，然后依次遍历里面的每一对值
#                     for question, answer in qaData.testingBatchIter(qTest, aTest, batchSize):
#                         # 来赋值的，格式为字典型
#                         feed_dict = {
#                             lstm.inputTestQuestions: question,
#                             lstm.inputTestAnswers: answer,
#                             lstm.keep_prob: dropout
#                         }
#                         _, scores = sess.run([globalStep, lstm.result], feed_dict)
#                         best = 0.0
#                         i = 0
#                         n = 0
#                         for score in scores:
#                             file.write("%.9f" % score + '\n')
#                             i += 1
#                             if score > best:
#                                 best = score
#                                 n = i
#
#                         with open(testingFile, mode="r", encoding="utf-8") as rf:
#                             json_d = json.load(rf)
#                             for block in json_d:
#                                 for ans in block['passages']:
#                                     if n == int(ans['passage_id']):
#                                         print("答案：" + ans['content'])
#                                         EditText.insert('end', "您的问题：" + q)
#                                         EditText.insert(INSERT, '\n')
#                                         EditText.insert('end', "可能答案："+ans['content'])
#                                         EditText.insert(INSERT, '\n')
#                                         EditText.insert(INSERT, '\n')
#                                         # print(best)
#
#             Button(root, text="查询", command=on_click).pack()
#             EditText = ScrolledText(root, width=80, height=50)
#             EditText.pack()
#             root.mainloop()
#     print("程序结束")
#
#

#
# """
# description: 原版主程序，需配合原版qaData.py使用，只计算问答句匹配度，结果存在predictRst中，无显示页面，优先使用GPU进行训练，如果是CPU版环境则用CPU训练
# """
# import os
# import time
# import tensorflow as tf
# import qaData
# from qaLSTMNet import QaLSTMNet
#
#
# def restore():
#     try:
#         print("正在加载模型，大约需要一分钟...")
#         saver.restore(sess, trainedModel)
#     except Exception as e:
#         print(e)
#         print("加载模型失败，重新开始训练")
#         train()
#
#
# def train():
#     print("重新训练，请保证计算机拥有至少8G空闲内存与2G空闲显存")
#     # 准备训练数据
#     print("正在准备训练数据，大约需要五分钟...")
#     qTrain, aTrain, lTrain, qIdTrain, aIdTrain = qaData.loadjsonData(trainingFile, word2idx, unrollSteps, True)
#
#     print("训练数据准备完毕")
#     tqs, tta, tfa = [], [], []
#     for question, trueAnswer, falseAnswer in qaData.trainingBatchIter(qTrain , aTrain ,lTrain , qIdTrain ,batchSize):
#         tqs.append(question), tta.append(trueAnswer), tfa.append(falseAnswer)
#
#     print("训练数据加载完成！")
#     # 开始训练
#     print("开始训练，全部训练过程大约需要12小时")
#     sess.run(tf.global_variables_initializer())
#     lr = learningRate  # 引入局部变量，防止shadow name
#     for i in range(lrDownCount):
#         optimizer = tf.train.GradientDescentOptimizer(lr)
#         optimizer.apply_gradients(zip(grads, tvars))
#         trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
#         for epoch in range(epochs):
#             print("epoch",epoch)
#             for question, trueAnswer, falseAnswer in zip(tqs, tta, tfa):
#                 # print("question.shape = ", question.shape)
#                 # print("trueAnswer.shape = ", trueAnswer.shape)
#                 # print("falseAnswer.shape = ", falseAnswer.shape)
#                 startTime = time.time()
#                 feed_dict = {
#                     lstm.inputQuestions: question,
#                     lstm.inputTrueAnswers: trueAnswer,
#                     lstm.inputFalseAnswers: falseAnswer,
#                     lstm.keep_prob: dropout
#                 }
#                 summary_val = sess.run(lstm.dev_summary_op,feed_dict)
#                 sess.run(trainOp,feed_dict)
#                 step = sess.run(globalStep,feed_dict)
#                 sess.run(lstm.trueCosSim,feed_dict)
#                 sess.run(lstm.falseCosSim,feed_dict)
#                 loss = sess.run(lstm.loss,feed_dict)
#                 timeUsed = time.time() - startTime
#                 print("step:", step, "loss:", loss, "time:", timeUsed)
#             saver.save(sess, saveFile)
#         lr *= lrDownRate
#
#
# if __name__ == '__main__':
#     # 定义参数
#     trainingFile = "./data/train_data_sample.json"
#     testingFile = "./data/test_data_sample.json"
#     resultFile = "predictRst.score"
#     saveFile = "newModel/savedModel"
#     trainedModel = "./newModel/savedModel"
#
#     embeddingFile = "./zhwiki/zhwiki_2017_03.sg_50d.word2vec"
#     embeddingSize = 50  # 词向量的维度
#
#     dropout = 1.0
#     learningRate = 0.4  # 学习速度
#     lrDownRate = 0.5  # 学习速度下降速度
#     lrDownCount = 4  # 学习速度下降次数
#     epochs = 20  # 每次学习速度指数下降之前执行的完整epoch次数
#     batchSize = 8  # 每一批次处理的问题个数
#
#     rnnSize = 100  # LSTM cell中隐藏层神经元的个数
#     margin = 0.1  # M is constant margin  余弦边界值 来对计算出的正负样本的语义相似度进行评判。
#
#     unrollSteps = 100  # 句子中的最大词汇数目
#     max_grad_norm = 5  # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
#
#     allow_soft_placement = True  # Allow device soft device placement 有时候不同的设备的cpu和gpu是不同的,那么当运行设备不满足要求时，会自动分配GPU或者CPU。
#     gpuMemUsage = 0.50  # 显存最大使用率
#     gpuDevice = "/gpu:0"  # GPU设备名
#
#     # 读取测试数据
#     print("正在载入测试数据，大约需要一分钟...")
#     embedding, word2idx = qaData.loadEmbedding(embeddingFile)
#     qTest, aTest, _, qIdTest,aIdTest = qaData.loadjsonData(testingFile, word2idx, unrollSteps)
#     print("测试数据加载完成")
#     # 配置TensorFlow
#     with tf.Graph().as_default(), tf.device(gpuDevice):
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpuMemUsage)
#         session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement, gpu_options=gpu_options)
#         with tf.Session(config=session_conf).as_default() as sess:
#             # 加载LSTM网络
#             print("正在加载LSTM网络，大约需要三分钟...")
#             globalStep = tf.Variable(0, name="globle_step", trainable=False)
#             # tf.Variable为tensorflow变量声明函数 trainable默认为True，可以后期被算法优化的。如果不想该变量被优化，改为False。
#             print("1")
#             lstm = QaLSTMNet(batchSize, unrollSteps, embedding, embeddingSize, rnnSize, margin)
#             # 实例化一个网络结构对象
#             print("2")
#             tvars = tf.trainable_variables()
#             # trainable_variables()函数可以也仅可以查看可训练的变量
#             print("3")
#             grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), max_grad_norm)
#             # 通过权重梯度的总和的比率来截取多个张量的值  第一个参数为梯度张量，第二个梯度为截取的比率 返回截取过的梯度张量和一个所有张量的全局范数。
#             print("4")
#             saver = tf.train.Saver()
#             # 保存模型
#             print("加载完成！")
#             # input()
#
#             # 加载模型或训练模型
#             if os.path.exists(trainedModel + '.index'):
#                 while True:
#                     choice = input("找到已经训练好的模型，是否载入（y/n）")
#                     if choice.strip().lower() == 'y':
#                         restore()
#                         break
#                     elif choice.strip().lower() == 'n':
#                         train()
#                         break
#                     else:
#                         print("无效的输入！\n")
#             else:
#                 train()
#
#             # 进行测试，输出结果
#             print("正在进行测试，大约需要三分钟...")
#             with open(resultFile, 'w') as file:
#                 for question, answer in qaData.testingBatchIter(qTest, aTest, batchSize):
#                     feed_dict = {
#                         lstm.inputTestQuestions: question,
#                         lstm.inputTestAnswers: answer,
#                         lstm.keep_prob: dropout
#                     }
#                     _, scores = sess.run([globalStep, lstm.result], feed_dict)
#                     for score in scores:
#                         file.write("%.9f" % score + '\n')
#     print("所有步骤完成！程序结束")