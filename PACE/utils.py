"""
用于测量两位置之间的距离，采用简单的欧式距离，在以后可以改进成其他的比如雅可比之类的

使用Haversine formula根据经纬度计算距离
"""

from math import sqrt
from math import pow, cos, asin, sin


def distance(a, b):
    """
    Parameter: a, b are two tuples in form of (x, y) that contains the coordinate of two locations
    Return: The distance between a and b

    """
    return sqrt(pow((a[0]-b[0]), 2)+pow((a[1]-b[1]), 2))

"""
def distance(a, b):
    r = 6371000
    h1 = pow(sin((b[0] - a[0])//2), 2) + cos(a[0]) * cos(a[1]) * pow(sin((b[1] - a[1])//2), 2)
    h = sqrt(h1)
    d = 2 * r * asin(h)
    return d
"""