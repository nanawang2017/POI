import numpy as np
import ast
import utils
import random
import pickle


class Dataset(object):
    def __init__(
            self,
            prefix='_small',
            negative=5,
            split=0.01,
            data_name='gowalla'):
        """
            Constructor:
                data_name: Name of Dataset
        """
        self.prefix = prefix
        self.negative = negative
        self.split = split
        self.file_path = '/Users/wangnana/学习/POI/paper_PACE/data/'
        # self.context_data = {}
        # self.context_data['user_context'] = []
        # self.context_data['spot_context'] = []
        self.generate()
    def generate(self):
        traindata_file='traindata'+self.prefix+'.pkl'
        testdata_file='testdata'+self.prefix+'.pkl'
        self.train_data = {}
        self.train_data['user'] = []
        self.train_data['spot'] = []
        self.train_data['label'] = []
        self.test_data = {}
        self.test_data['user'] = []
        self.test_data['spot'] = []
        self.test_data['label'] = []

        self.user_enum, self.spot_enum = self.getCrossLabels()
        print('Writing ' + str(len(self.train_data['user'])) + ' training labels to file')
        with open(self.file_path + traindata_file, 'wb') as f:
            pickle.dump(self.train_data, f)
        """
        打开文件，若文件不存在则创建；将前者参数obj写到后者file：f中
        """
        print('Writing ' + str(len(self.test_data['user'])) + ' testing labels to file')
        with open(self.file_path + testdata_file, 'wb') as f:
            pickle.dump(self.test_data, f)



    def getCrossLabels(self,
                       file_name='gowalla/visited_spots.txt',
                       user_filter_lower=100,
                       spot_filter_lower=100,
                       user_filter_upper=1000,
                       spot_filter_upper=1000
                       ):
        """
        该函数主要就是：
        用于过滤用户和spot
        :param file_name: File name of the file that contains the graph
        :param user_filter_lower、 spot_filter_lower、user_filter_upper、 spot_filter_upper:
        :return: user_enum, spot_enum
        """
        user_dict = {}  # user_id:[place_id_1, place_id_2, ...]
        spot_dict = {}  # place_id:[user_id_1,...]
        negative_sample = self.negative  # sample 5 negative samples for each positive label
        split_portion = self.split  # 训练及测试集通过随机抽取比例进行分割
        with open(self.file_path + file_name, 'r') as f:
            print('Reading file ' + file_name + ' to construct training labels')
            lines = f.readlines()  # readlines()返回的是列表形式
            """
            #print(len(lines[0]))   3245
            #print(type(lines[0]))  <class 'str'>每一行以字符串的形式 输出全是字符
            lines是由一行行列表组成
            每一行以字符形式存于列表中，所以使用空格对每一行进行划分 空格前的为键并转为整数形式 
            即变成user_id [place_id_1, place_id_2, ...]
            这样key就变成了user_id
            那么[place_id_1, place_id_2, ...]
            line[len(str(key)) + 1:就是指key后面的所有字符
            print(type(spots_array))
            print(spots_array
            ast.literal_eval 把数据还原成它本身或者是能够转化成的数据类型 也就是将其变为列表类型
            最后将visited_spots.txt转化为user_dict字典形式存储
            '''创建一个{spot:[user1,user2]}'''
            """
            for line in lines:
                key = int(line.split(' ')[0])
                spots_array = ast.literal_eval(line[len(str(key)) + 1:])
                user_dict[key] = spots_array
                for spot in spots_array:
                    if spot not in spot_dict:
                        # 创建一个{spot:[user1,user2]}
                        spot_dict[spot] = []
                    spot_dict[spot].append(key)
            # return user_dict,spot_dict

            '''到此字典user_dict、spot_dict 创建完成 '''

            """
            如何删选过滤user和spot呢：
                对于user_dict如何删选键：user，看他visit 的spot的数量，少于100和大于1000的就删除掉
                也就是说保留那些visited spots在100和1000范围内的user
                对于spot_dict如何删选键：spot，看有多少user来visit，user数量少于100大于1000的spot就被删除
                也就是说保留那些visited user 数量在100和1000以内的spot
            """
            print('Filtering users and spots')
            for user in list(user_dict.keys()):
                if (len(user_dict[user]) < user_filter_lower) or (len(user_dict[user]) > user_filter_upper):
                    del user_dict[user]
            for spot in list(spot_dict.keys()):
                if (len(spot_dict[spot]) < spot_filter_lower) or (len(spot_dict[spot]) > spot_filter_upper):
                    del spot_dict[spot]
            print('#users:' + str(len(user_dict)) + ', #spots:' + str(len(spot_dict)))
            '''过滤user和spot完成'''

            print('Generating labels')
            user_enum = {}
            spot_enum = {}
            # 用于给user_id重新编号
            u_counter = 0
            s_counter = 0
            for user in user_dict.keys():
                user_enum[user] = u_counter
                u_counter += 1
                for spot in user_dict[user]:
                    if spot in spot_dict.keys():
                        if spot not in spot_enum:
                            spot_enum[spot] = s_counter
                            s_counter += 1
                        if random.random() < split_portion:
                            self.train_data['user'].append(user_enum[user])
                            self.train_data['spot'].append(spot_enum[spot])
                            self.train_data['label'].append(1)
                            for i in range(negative_sample):
                                if random.random() > 0.5:
                                    self.train_data['user'].append(user_enum[user])
                                    self.train_data['spot'].append(random.randrange(len(spot_dict)))
                                    self.train_data['label'].append(0)
                                    """
                                    不明白为何这样选择negative label 这样的话spot可能是user visited 过的
                                    看论文
                                    """

                                else:
                                    self.train_data['user'].append(random.randrange(len(user_dict)))
                                    self.train_data['spot'].append(spot_enum[spot])
                                    self.train_data['label'].append(0)
                        else:
                            self.test_data['user'].append(user_enum[user])
                            self.test_data['spot'].append(spot_enum[spot])
                            self.test_data['label'].append(1)
                            for i in range(negative_sample):
                                if random.random() > 0.5:
                                    self.test_data['user'].append(user_enum[user])
                                    self.test_data['spot'].append(random.randrange(len(spot_dict)))
                                    self.test_data['label'].append(0)
                                else:
                                    self.test_data['user'].append(random.randrange(len(user_dict)))
                                    self.test_data['spot'].append(spot_enum[spot])
                                    self.test_data['label'].append(0)
        print("user_enum: "+str(len(user_enum))+"; spot_enum: "+str(len(spot_enum)))
        print("user_dict: "+str(len(user_dict.keys())))
        return user_enum, spot_enum


if __name__ == "__main__":
    Dataset()
