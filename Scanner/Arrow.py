# class arrow
class arrow:

    # constructor
    def __init__(self, headPoint, tailPoint , headLeftPoint = (), headRightPoint=()):
        self.__head = headPoint
        self.__tail = tailPoint
        # self.__headLeftPoint = headLeftPoint
        # self.__headRightPoint = headRightPoint

    # setters
    def set_head(self, headPoint, headLeftPoint = (), headRightPoint=()):
        self.__head = headPoint
        # self.__headLeftPoint = headLeftPoint
        # self.__headRightPoint = headRightPoint
    
    def set_tail(self, tailPoint):
        self.__tail = tailPoint
    
    # getters
    def get_head(self):
        return self.__head
        # return (self.__headLeftPoint,self.get_head,self.headRightPoint)
    
    def get_tail(self):
        return self.__tail

# testing
# a =  arrow((2,2),(0,0))
# print(a)
# print(a.get_head())
# print(type(a.get_head()))
# print(a.get_tail())