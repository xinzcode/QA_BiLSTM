#!/usr/bin/python
# coding=utf-8
"""
description:
    The implementation of calculating MRR/MAP/ACC@1
    Usage: 'python evaluation.py QApairFile scoreFile outputFile'
"""
import sys
import codecs


class Evaluator(object):
    qIndex2aIndex2aScore = {}
    qIndex2aIndex2aLabel = {}
    ACC_at1List = []
    APlist = []
    RRlist = []

    def __init__(self, qaPairFile, scoreFile):
        self.loadData(qaPairFile, scoreFile)

    def loadData(self, qaPairFile, scoreFile):
        qaPairLines = codecs.open(qaPairFile, 'r', 'utf-8').readlines()
        scoreLines = open(scoreFile).readlines()
        assert len(qaPairLines) == len(scoreLines)
        qIndex = 0
        aIndex = 0
        lastQuestion = ''
        for idx in range(len(qaPairLines)):
            qaLine = qaPairLines[idx].strip()
            qaLineArr = qaLine.split('zx')
            assert len(qaLineArr) == 3
            question = qaLineArr[0]
            #             answer=qaLineArr[1]
            label = int(qaLineArr[2])
            score = float(scoreLines[idx])
            if question != lastQuestion:
                if idx != 0:
                    qIndex += 1
                aIndex = 0
                lastQuestion = question
            if not qIndex in self.qIndex2aIndex2aScore:
                self.qIndex2aIndex2aScore[qIndex] = {}
                self.qIndex2aIndex2aLabel[qIndex] = {}
            self.qIndex2aIndex2aLabel[qIndex][aIndex] = label
            self.qIndex2aIndex2aScore[qIndex][aIndex] = score
            aIndex += 1

    def calculate(self):
        for qIndex, index2score in self.qIndex2aIndex2aScore.items():
            index2label = self.qIndex2aIndex2aLabel[qIndex]
            Index = 0
            rightNum = 0
            curPList = []
            rankedList = sorted(index2score.items(), key=lambda b: b[1], reverse=True)
            # 按照匹配度由高到底排序，index由0开始
            self.ACC_at1List.append(0)
            for info in rankedList:
                QAIndex = info[0]
                label = index2label[QAIndex]
                Index += 1
                if label == 1:
                    rightNum += 1
                    if Index == 1:
                        # 模型预测正确
                        self.ACC_at1List[-1] = 1
                    p = float(rightNum) / Index
                    # p为目前的正确率
                    curPList.append(p)
            if len(curPList) > 0 and len(curPList) != len(rankedList):
                self.RRlist.append(curPList[0])
                self.APlist.append(float(sum(curPList)) / len(curPList))
            else:
                self.ACC_at1List.pop()

    def MRR(self):
        return float(sum(self.RRlist)) / len(self.RRlist)

    def MAP(self):
        return float(sum(self.APlist)) / len(self.APlist)

    def ACC_at_1(self):
        return float(sum(self.ACC_at1List)) / len(self.ACC_at1List)


def evaluate(QApairFile, scoreFile, outputFile):
    testor = Evaluator(QApairFile, scoreFile)
    testor.calculate()
    print("MRR:%f \t MAP:%f \t ACC@1:%f\n" % (testor.MRR(), testor.MAP(), testor.ACC_at_1()))
    # MAP: 多类别平均准确率  MRR：平均倒数排名 ACC@1:分数最高的为正确答案的准确率
    if outputFile != '':
        fw = open(outputFile, 'a')
        fw.write('%f \t %f \t %f\n' % (testor.MRR(), testor.MAP(), testor.ACC_at_1()))


if __name__ == '__main__':
    QApairFile = './data/test.txt'   # 测试样例 需要将训练时用的测试样例调整为指定的格式放在其中

    scoreFile = 'test.score'      # 测试样例的模型运行结果

    outputFile = 'evaluation.score'  # 评估结果输出文件

    evaluate(QApairFile, scoreFile, outputFile)
