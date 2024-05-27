from enum import Enum

# Enumeration for the types of states
class Direction(Enum):
    # name = value
    Loop = 0
    Up = 1
    Down = 2
    Right = 3
    Left = 4
    # diagonal

# class arrow
class arrow:

    # constructor
    # def __init__(self, headPoint, tailPoint , headLeftPoint = (), headRightPoint=()):
    def __init__(self, direction : Direction = None ,headPoint = (), tailPoint = () , headLeftPoint = (), headRightPoint=()):
        self.__direrction = direction
        # self.__head = headPoint
        # self.__tail = tailPoint
        # self.__headLeftPoint = headLeftPoint
        # self.__headRightPoint = headRightPoint

    # setters
    def set_head(self, headPoint, headLeftPoint = (), headRightPoint=()):
        self.__head = headPoint
        # self.__headLeftPoint = headLeftPoint
        # self.__headRightPoint = headRightPoint
    
    def set_tail(self, tailPoint):
        self.__tail = tailPoint

    def set_direction(self, direction):
        self.__direrction = direction
    
    # getters
    def get_head(self):
        return self.__head
        # return (self.__headLeftPoint,self.get_head,self.headRightPoint)
    
    def get_tail(self):
        return self.__tail
    
    def get_direction(self):
        return self.__direrction

# testing
# a =  arrow((2,2),(0,0))
# print(a)
# print(a.get_head())
# print(type(a.get_head()))
# print(a.get_tail())