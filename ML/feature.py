#!/usr/bin/python3

# from sklearn.feature_extraction.text import CountVectorizer

# # 实例化
# vector = CountVectorizer()

# # 输出数据并转换
# res = vector.fit_transform(["life is short, i dislike python", 
#                         "life is too long, i dislike python"])  

# print(vector.get_feature_names())
# print(res.toarray())


import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np

'''
字典抽取
'''
def dictvec():
        dict = DictVectorizer(sparse=False)

        data = dict.fit_transform([
                {'city':'北京', 'temp':100},
                {'city':'上海', 'temp':60},
                {'city':'深圳', 'temp':30},
        ])

        print(dict.get_feature_names())
        print(data)


        return None


'''
对文本进行特征值化(特征提取)
应用与文本分类, 情感分析, 对于单个英文字母不统计
不支持中文词语抽取, 中文需要用空格隔开, 也统计单个汉字
中文分词可使用import jieba
'''
def countvec():
        cv = CountVectorizer()
        data = cv.fit_transform(["life is is short, i dislike python",
                         "人生 太 长了, i like python"])
        # 统计了所有文章去重次, 即取文章所有词, 单个字母不统计
        print(cv.get_feature_names())
        # 对每篇文章统计上述词出现次数
        print(data.toarray())

        return None


def cutword():
        # 剪词
        a = jieba.cut('关注了很久的一个公众号，基本上每天都会准时分享一篇作者的原创文章。除了大节日里会提前说明的情况外，从来没有一天失信过读者的。')
        b = jieba.cut('于是就开始在留言区看到，很多网友都在羡慕作者把平台做得多么多么的大，文章的阅读量多么的高。于是就开始有人也兴冲冲地去注册个公众号，跑来向作者请教如何做平台，有没有什么经验之类的。')
        c = jieba.cut('我看到，作者只回复了这群网友两个字：坚持！是啊，只有真正坚持过的人，才知道坚持两个字有多难做到，它的威力有多大，会给你多少奇迹。')
        # 对象转换为列表
        a = list(a)
        b = list(b)
        c = list(c)
        # 列表转换为单个字符串
        a = ' '.join(a)
        b = ' '.join(b)
        c = ' '.join(c)


        return a, b, c



'''
中文特征值化
'''
def zhvec():
        cv = CountVectorizer()
        a, b, c = cutword()
        data = cv.fit_transform([a, b, c])
        # 统计了所有文章去重次, 即取文章所有词, 单个字母不统计
        print(cv.get_feature_names())
        # 对每篇文章统计上述词出现次数
        print(data.toarray())


'''
朴素贝叶斯文本分类
tfidf:
tf: term frequency, idf: inverse document frequency
'''
def tfidfvec():
        tfidf = TfidfVectorizer()
        a, b, c = cutword()
        data = tfidf.fit_transform([a, b, c])
        # 统计了所有文章去重次, 即取文章所有词, 单个字母不统计
        print(tfidf.get_feature_names())
        # 对每篇文章统计上述词出现次数
        print(data.toarray())


def mm():
        """
        归一化处理
        为了避免某一个特征对最终结果造成较大影响
        """
        mm = MinMaxScaler()
        data = mm.fit_transform([[32,65,23,23],[76,344,23,5],[32,54,75,33]])
        print(data)


def stand():
        """
        标准化缩放
        在已有样本足够多的情况下比较稳定,
        适合现在嘈杂的大数据场景, 通常使用标准化
        用到平均值和标准差(以及方差)
        """
        std = StandardScaler()
        data = std.fit_transform([[32,65,23,23],[76,344,23,5],[32,54,75,33]])
        print(data)


def im():
        """
        缺失值处理
        """
        # 缺失值, 填补策略, 
        im = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = im.fit_transform([[1,2], [np.nan, 3], [7, 6]])
        print(data)


if __name__ == "__main__":
        im()

