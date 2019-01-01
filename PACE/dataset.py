'''
	Created on Jan 23, 2016
	Handle all the data preprocessing, turning files to numpy files
	Sample graph to generate labels

	@author: Lanxiao Bai, Carl Yang
'''
import numpy as np
import ast
import utils
import random
import pickle
from scipy.sparse import csr_matrix


class Dataset(object):
    def __init__(
            self,
            prefix='_small',
            negative=5,
            split=0.01,
            data_name='gowalla'):
        '''
			Constructor:
				data_name: Name of Dataset
		'''
        self.prefix = prefix
        self.negative = negative
        self.split = split
        self.file_path = '/Users/wangnana/学习/POI/代码以及模型/神经网络方面/PACE&NeuMF/PACE2017-master/data/'
        self.context_data = {}
        self.context_data['user_context'] = []
        self.context_data['spot_context'] = []
        self.generate()

    def generate(self):
        interdata_file = 'inter' + self.prefix + '.pkl'
        traindata_file = 'traindata' + self.prefix + '.pkl'
        testdata_file = 'testdata' + self.prefix + '.pkl'

        writeToFile = False
        try:
            f = open(self.file_path + interdata_file, 'r')
            f.close()
        except IOError:
            writeToFile = False

        if not writeToFile:
            with open(self.file_path + interdata_file, 'rb') as f:
                inter_data = pickle.load(f)
            with open(self.file_path + traindata_file, 'rb') as f:
                self.train_data = pickle.load(f)
            with open(self.file_path + testdata_file, 'rb') as f:
                self.test_data = pickle.load(f)
            self.user_enum = inter_data['user_enum']
            self.spot_enum = inter_data['spot_enum']
            self.user_label = inter_data['user_label']
            self.spot_label = inter_data['spot_label']
            print(str(len(self.user_enum)) + ' users in enum loaded')
            print(str(len(self.spot_enum)) + ' spots in enum loaded')
            print(str(len(self.user_label)) + ' user context labels loaded')
            print(str(len(self.spot_label)) + ' spot context labels loaded')
            print(str(len(self.train_data['user'])) + ' training labels loaded')
            print(str(len(self.test_data['user'])) + ' test labels loaded')
        else:
            inter_data = {}
            self.train_data = {}
            self.train_data['user'] = []
            self.train_data['spot'] = []
            self.train_data['label'] = []
            self.test_data = {}
            self.test_data['user'] = []
            self.test_data['spot'] = []
            self.test_data['label'] = []

            self.user_enum, self.spot_enum = self.getCrossLabels()
            self.user_dict = self.getUserGraph(self.user_enum)
            self.spot_dict = self.getSpotGraph(self.spot_enum)
            self.user_label = self.getSmoothLabels(self.user_dict)
            self.spot_label = self.getSmoothLabels(self.spot_dict)
            inter_data['user_enum'] = self.user_enum
            inter_data['spot_enum'] = self.spot_enum
            inter_data['user_label'] = self.user_label
            inter_data['spot_label'] = self.spot_label
            with open(self.file_path + interdata_file, 'wb') as f:
                pickle.dump(inter_data, f)
            print('Writing ' + str(len(self.train_data['user'])) + ' training labels to file')
            with open(self.file_path + traindata_file, 'wb') as f:
                pickle.dump(self.train_data, f)
            print('Writing ' + str(len(self.test_data['user'])) + ' testing labels to file')
            with open(self.file_path + testdata_file, 'wb') as f:
                pickle.dump(self.test_data, f)

    def getCrossLabels(
            self,
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
        user_dict = {}
        spot_dict = {}
        negative_sample = self.negative
        split_portion = self.split
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
        with open(self.file_path + file_name, 'r') as f:
            print('Reading file ' + file_name + ' to construct training labels')
            lines = f.readlines()
            total = len(lines)
            for line in lines:
                key = int(line.split(' ')[0])
                spots_array = ast.literal_eval(line[len(str(key)) + 1:])
                user_dict[key] = spots_array
                for spot in spots_array:
                    if spot not in spot_dict:
                        spot_dict[spot] = []
                    spot_dict[spot].append(key)


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

        print('Generating labels')
        user_enum = {}
        spot_enum = {}
        u_counter = 0
        s_counter = 0

        for user in user_dict.keys():
            user_enum[user] = u_counter
            u_counter += 1
            for spot in user_dict[user]:
                if spot in spot_dict:
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

        return user_enum, spot_enum

    def getUserGraph(
            self,
            user_enum,
            file_name='gowalla/user_network.txt'
    ):
        """
        Parameter:
file_name: File name of the file that contains the graph
    Return:
numpy.array: uu_friend_matrix that represents the user-user friendship network
对user_network中那些有朋友关系的user，利用user_enum构造了一个具有新的user_id的字典，用u_counter来表示朋友关系
即relation_dict={u_counter1:[u_counter2,u_counter3,....]

        """

        relation_dict = {}
        print('Reading file ' + file_name + ' to construct user graph')
        density = 0
        with open(self.file_path + file_name, 'r') as f:
            lines = f.readlines()
            total = len(lines)
            for line in lines:
                key = int(line.split(' ')[0])
                if key in user_enum:
                    relation_dict[user_enum[key]] = [user_enum[i]
                                                     for i in ast.literal_eval(line[len(str(key)) + 1:])
                                                     if i in user_enum]
                    density += len(relation_dict[user_enum[key]])
        density = density * 1.0 / (len(user_enum) * len(user_enum))
        print('Density of user graph: ' + str(density))

        return relation_dict

    def getSpotGraph(
            self,
            spot_enum,
            sample_portion=0.01,
            sample_radius=0.5,
            file_name='gowalla/spot_location.txt'):
        """
Parameter:
file: File that contains spot ids and latitudes and longitudes.
radius: The maximum distances for two locations to be connected
dict: spot_enum that records each spot id's correspondent number computed by getVisitedGraph

Return:
numpy.array: ss_location_matrix that represents the spot-spot location network
list of pairs: ss_location_label that represents labels generated from the spot-spot location network
"""

        coordinates = {}
        with open(self.file_path + file_name, 'r') as f:
            print('Reading file ' + file_name + ' to construct spot graph')

            lines = f.readlines()
            total = len(lines)
            for line in lines:
                # print("Loading:" + str(counter) + "/" + str(total - 1) + "--" + line)
                splited = line.split(' ')
                n = 0
                for i in splited:
                    if i == 'null':
                        n = 1
                if n == 0:
                    splited = [float(i) for i in line.split(' ')]
                    spot, x, y = splited
                    spot = int(spot)
                    if spot in spot_enum:
                        coordinates[spot_enum[spot]] = (x, y)

                        """
                        第一部分先是读入文件  将这些字符都转为小数，并且生成字典：
                        coordinates = {spot_enum[spot]：(x, y),......}
                        spot_enum[spot]新的spot编号
                        """

        relation_dict = {}
        density = 0
        sample_size = int(len(spot_enum) * sample_portion)
        print('Sampling ' + str(sample_size) + ' base spots to build spot graph')
        base_points = random.sample(coordinates.keys(), k=sample_size)
        '''spot太多了，进行下一步采样得到base_points，base_points里面存放的是s_counter'''
        for base in base_points:
            # print("Loading:" + str(s_counter) + "/" + str(self.sample_size))
            cell = []
            for i in coordinates.keys():
                if utils.distance(coordinates[i], coordinates[base]) < sample_radius:
                    cell.append(i)
            '''
            对于base_points里面的spot，从所有的spot选出符合条件的location放到cell
                cell存放的是在以base为圆心，以sample_radius为半径的区域的location
                这些location存放在relation_dict字典里面
            '''
            # print('Cell '+str(base)+' has '+str(len(cell))+' spots')
            for i in cell:
                for j in cell:
                    if i != j:
                        if i not in relation_dict:
                            relation_dict[i] = set()
                        if j not in relation_dict:
                            relation_dict[j] = set()
                        relation_dict[i].add(j)
                        relation_dict[j].add(i)
                        '''生成relation_dict={s_counter：set（里面是跟自己距离很近的location组成的集合）}'''

        for i in relation_dict.keys():
            relation_dict[i] = list(relation_dict[i])
            density += len(relation_dict[i])
        '''将集合转化为列表'''
        density = density * 1.0 / (len(spot_enum) * len(spot_enum))
        print('Density of spot graph: ' + str(density))
        return relation_dict

    def getSmoothLabels(
            self,
            graph_dict,
            path_portion=0.01,
            path_length=10,
            samples_num=5,
            window_size=3):
        """
        :param graph_dict: Dict that stores the graph
        :param path_portion:
        :param path_length:
        :param samples_num:
        :param window_size:
        :return: Labels sampled from the graph
        其实就是返回每一个label的context label 这个label满足：
        1：对构建图中节点进行限定；path_num = int(len(graph_dict) * path_portion)
        2:每个节点有一个path=[] 即随机游走序列S，
        path中的第一个元素是随机选的graph_dict.keys()，其他的元素是该key的value（cands)即context（随机选）
        限制path的总长度为10
        3:对这个 path进一步处理，删选出samples_num个（删选条件path中任意两个的index小于window size）
        小于 window_size跳出while循环将他们放到labels即 labels={graph_dict.keys:【samples_num个context】}
        """

        print('Generating smooth labels')
        labels = {}
        path_num = int(len(graph_dict) * path_portion)
        for i in range(path_num):
            path = []
            for j in range(path_length):
                if len(path) == 0:
                    path.append(list(graph_dict.keys())[random.randrange(len(graph_dict))])
                else:
                    if path[len(path) - 1] not in graph_dict or len(graph_dict[path[len(path) - 1]]) == 0:
                        break
                    cands = graph_dict[path[len(path) - 1]]
                    path.append(cands[random.randrange(len(cands))])
            if len(path) > 1:
                for k in range(samples_num):
                    while True:
                        tup = random.sample(path, k=2)
                        if abs(path.index(tup[0]) - path.index(tup[1])) < window_size:
                            break
                    if tup[0] not in labels:
                        labels[tup[0]] = []
                    if tup[1] not in labels[tup[0]]:
                        labels[tup[0]].append(tup[1])

        return labels

    def generateContextLabels(self):
        '''
        user_label={user1:[5个context user],....}
        spot_label={spot1:[5 context spot],...}
        '''
        print('Generating ' + str(len(self.train_data['label'])) + ' context labels')
        for i in range(len(self.train_data['label'])):
            # print(str(i))
            tmp = [0] * len(self.user_enum)
            # list * int 意思是将数组重复 int 次并依次连接形成一个新数组
            user = self.train_data['user'][i]
            if user in self.user_label:
                user_context = self.user_label[user]
                for j in user_context:
                    tmp[j] = 1
            self.context_data['user_context'].append(np.array(tmp))
            '''
            user_enum是从0开始编号，即tmp[j] = 1就对应这这个编号的位置是1 这里user_id就是下标编号啦
            context_data={user_context(即u_counter）：[这个tmp数组长度是user_enum的长度=即所有user个数，哪个是
            该user的删选出来的label对应位置就是1比如：0，0，0，1，1，1，1，0，0]}   1的个数是user_label的长度=5
            '''
            # tmp = [0] * len(self.spot_enum)
            row_ind = []
            col_ind = []
            data = []
            tmp = [0] * len(self.spot_enum)
            spot = self.train_data['spot'][i]
            if spot in self.spot_label:
                spot_context = self.spot_label[spot]
                for j in spot_context:
                    tmp[j] = 1
            self.context_data['spot_context'].append(np.array(tmp))

    def getContextLabels(self):
        self.generateContextLabels()

        return self.context_data


if __name__ == "__main__":
    dt = Dataset()
