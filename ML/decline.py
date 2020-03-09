#!/usr/bin/python3

"""
降维:特征的数量, 去掉对结果无影响的特征
1. 特征选择
        Filter(过滤式):VarianceThreshold
        Embedded(嵌入式):正则化/决策树
        Wrapper(包裹式)
2. 主成分分析
        PCA:特征数量100以上, 考虑降维
        PCA可以通过找到特征值之间的关系(数据拟合)进行降维
        也可以用于消减回归分析或者聚类分析中特征的数量


"""

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import pandas as pd

def vt():
        """
        特征选择, 删除最低方差的特征
        """
        vt = VarianceThreshold(threshold=1.0)
        data = vt.fit_transform([[0,2,0,3],[0,1,4,3],[0,1,1,3]])
        print(data)


def pca():
        """
        通过PCA(主成分分析)降维
        超过100特征用PCA
        """
        # 小数是根据经验保留90%-95%之间的数据,值越小信息损失越大
        pca = PCA(n_components=0.95)
        data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
        print(data)
        

def example():
        # 通过pd.merge合并有同列的数据
        tab1 = pd.read_csv('file1')
        tab2 = pd.read_csv('file2')

        _mg = pd.merge(tab1, tab2, on=['id', 'id'])

        # 交叉表, 特殊的分组, 行作为id, 列作为物品, 表内容则为某人有多少某物
        cross = pd.crosstab(_mg['id'], _mg['goods'])
        # 通过主成分分析, 降维至保留90%原有信息
        pca = PCA(n_components=0.9)
        data = pca.fit_transform(cross)
        # 查看样本特征列数
        data.shape



if __name__ == "__main__":
        pca()


