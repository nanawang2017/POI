import random
import math


class UserBasedCF(object):
    def __init__(self, train=None, test=None):
        self.trainfile = train
        self.testfile = test
        self.readData()

    def readData(self, train=None, test=None):
        self.trainfile = train or self.trainfile
        self.testfile = test or self.testfile
        self.traindata = {}
        self.testdata = {}
        # 读入训练集 存到traindata字典中
        for line in open(self.trainfile):
            userid, itemid, record, _ = line.split()
            # 但如果user_id不存在于字典中，将会添加user_id并将值设为default({})
            self.traindata.setdefault(userid, {})
            # traindata = {userid: {itemid: record}}
            self.traindata[userid][itemid] = record
        for line in open(self.testfile):
            userid, itemid, record, _ = line.split()
            self.testdata.setdefault(userid, {})
            self.testdata[userid][itemid] = record

    def userSimilarityBest(self, train=None):
        train = train or self.trainfile
        self.userSimBest = dict()
        item_users = {}
        # traindata = {userid: {itemid: record}}
        # item_users={item_id:(user_ids)}
        for u, item in train.items():
            for i in item.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)
        user_item_count = dict()
        # user_item_count统计user评过物品的个数 user_item_count={user:user 出现的次数}即 键是user，值为user评过item的个数
        count = dict()
        # count统计用户之间共同评过分的物品个数 count={user1:{user2:user1和user2共同喜欢item的个数统计,user3：count of user1 and user3}}
        for item, users in item_users.items():
            for u in users:
                user_item_count.setdefault(u, 0)
                user_item_count[u] += 1
                for v in users:
                    if u == v: continue
                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        # userSimBest={u:{v:相似度}}
        for u, related_users in count.items():
            self.userSimBest.setdefault(u, dict())
            for v, cuv in related_users.items():
                self.userSimBest[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v] * 1.0)

    def recommend(self, user, train=None, k=8, nitems=40):
        # 读入数据集
        train = train or self.traindata
        # 排序列表是dict 字典形式
        rank = dict()
        # 用户交互过的物品 即评过分的物品
        interacted_items = train.get(user, {})
        # 因为是UserCF，所以首先要找到和用户相似度最高的K个用户
        # self.userSimBest[user].items()指的是和用户u有过相同物品交互的用户
        # key=lambda x:x[1]按照第2个元素进行排序
        # v是指用户 wuv是指用户u和v之间的相似度权重
        for v, wuv in sorted(self.userSimBest[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in train[v].items():
                if i in interacted_items: continue
                rank.setdefault(i, 0)  # 将 i 放到推荐列表
                rank[i] += rvi * wuv
        # traindata = {userid: {itemid: record}}
        # train[v].items()指的是用户v的{itemid: record} 所以i 指的是item，rvi 指是用户v 和物品i的record
        # recommend 返回的是 {item ：评分}
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    # 用sorted方法对推荐的物品进行排序，预计评分高的排在前面，再取其中nitem个，nitem为每个用户推荐的物品数量

    def recallAndPrecision(self, train=None, test=None, k=8, nitems=10):
        train = train or self.traindata
        test = test or self.testdata
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user, {})
            rank = self.recommend(user, train=train, k=k, nitems=nitems)
            for item, _ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            # recall 的分母是用户在测试集中的item
            precision += nitems
            # precision 的分母是推荐列表
        return (hit / (recall * 1.0)), (hit / (precision * 1.0))


def testUserBasedCF(self):
    train = 'ul.base'
    test = 'ul.test'
    cf = UserBasedCF(train, test)
    cf.userSimilarityBest()
    print("%3s%20s%20s%20s%20s" % ('K', "precision", 'recall'))
    for k in [10, 20, 30, 40]:
        recall, precision = cf.recallAndPrecision(k=k)
        print("%3d%19.3f%%%19.3f%%%19.3f%%%20.3f" % (k, precision * 100, recall * 100))


if __name__ == '__main__':
    testUserBasedCF()
