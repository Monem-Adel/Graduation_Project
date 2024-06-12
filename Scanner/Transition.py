# import sys
import State, Arrow

# sys.path.append(r'H:\Scanner')
# print(sys.path)
#
# Note: the prefix __ of identifier is notation means that id is private
# transition class
class transition:

    # constructor
    # def __init__(self,label, bbox:tuple, arrow:Arrow.arrow, source:State.state = State.state(), destination:State.state= State.state()):
    def __init__(self,label, bbox:tuple, arrow:Arrow.arrow, source:State.state = None, destination:State.state= None):
    # def __init__(self,label, bbox:tuple, arrow:Arrow.arrow):
        self.__label = label
        self.__source = source
        self.__destination = destination
        self.__arrow = arrow
        # bounding box (xt,yt,xb,yb) t: top left, r: bottom right
        self.__bbox = bbox
        Xt,Yt,Xb,Yb = bbox
        self.__top_left = (Xt,Yt) # top coordinate
        self.__bottom_right = (Xb,Yb) # bottom coordinate
        # the tensor of the image (if needed)

    # setters
    def set_label(self,label):
        self.__label = label

    def set_source(self,source:State.state):
        self.__source = source

    def set_destination(self,destination:State.state):
        self.__destination = destination

    def set_arrow(self,arrow:Arrow.arrow):
        self.__arrow = arrow

    def set_bbox(self,bbox : tuple):
        self.__bbox = bbox
        Xt,Yt,Xb,Yb = bbox
        self.__top_left = (Xt,Yt) # top coordinate
        self.__bottom_right = (Xb,Yb) # bottom coordinate

    def set_headAndTailPoint(self):
        x,y = self.__top_left
        arw = self.get_arrow()
        p,q = arw.get_tail()
        w,z = arw.get_head()
        arw.set_tail((x+p,y+q))
        arw.set_head((x+w,y+z))
        self.set_arrow(arw)

    
    # getters
    def get_label(self):
        return self.__label
    
    def get_source(self):
        #if-raise
        if (self.__source == None):
            raise Exception("ERROR: haven't assigned source state to the transition")
        return self.__source
    
    def get_destination(self):
        #if-raise
        if (self.__destination == None):
            raise Exception("ERROR: haven't assigned destination state to the transition")
        return self.__destination
    
    def get_arrow(self):
        return self.__arrow
    
    def get_top_coordinate(self):
        return self.__top_left
    
    def get_bottom_coordinate(self):
        return self.__bottom_right
    
    def get_bbox(self):
        return self.__bbox
        # return (*self.__top_left,*self.__bottom_right)

    # a method to return a tuple of source & destination
    def get_sourceAndDistination(self):
        #if-raise
        if (self.__source == None or self.__destination == None):
            raise Exception("ERROR: haven't assigned either source  or destination state to the transition")
        return (self.__source,self.__destination)
    
# testing
# # source = State.state('q0',State.Type_of_state.Start_State,(3,9,15,19))
# # destination = State.state('q1',State.Type_of_state.Final_State,(3,9,15,19))
# test = transition('A',(3,9,15,19), arrow="a Aroowwwww")

# print(f'The class is {test}')
# print(f'The label is {test.get_label()}')
# print(f'The source state is : \n\t Name:{test.get_source().get_name()}\n\t type: {test.get_source().get_type().name}')
# print(f'The destination state is : \n\t Name:{test.get_destination().get_name()}\n\t type: {test.get_destination().get_type().name}')
# print(f'The bbox is : \n\t Name:{test.get_bbox()}')
# print(f'The top left point is {test.get_top_coordinate()} & the bottom right bottom {test.get_bottom_coordinate()}')

# d = Arrow.Direction.Down
# t = transition('1',(2,2,2,2),Arrow.arrow(d))
# print(t.get_arrow().get_direction().name)
# print(t.get_arrow().get_direction().value)
# print(type(t.get_arrow().get_direction().name))
# print(type(t.get_arrow().get_direction().value))