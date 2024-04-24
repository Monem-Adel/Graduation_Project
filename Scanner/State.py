from enum import Enum

# # functional syntax
# Type_of_state = Enum('Type_of_state', ['Start_State', 'Normal_State', 'Final_State'])

# class syntax
# Enumeration for the types of states
class Type_of_state(Enum):
    Start_State = 0
    Normal_State = 1
    Final_State = 2

# state class
# Note: the prefix __ of identifier is notation means that id is private
class state :

    # constructor
    def __init__(self,name = 'dummy',stateType : Type_of_state = Type_of_state.Normal_State , bbox:tuple = (-1,-1,-1,-1)):
        self.__name = name
        self.__Type_of_state = stateType
        # bounding box (xt,yt,xb,yb) t: top left, r: bottom right
        self.__bbox = bbox
        Xt,Yt,Xb,Yb = bbox 
        self.__top_left = (Xt,Yt) # top coordinate
        self.__bottom_right = (Xb,Yb) # bottom coordinate
        # the tensor of the image (if needed)
    
    # setters
    def set_name(self,name):
        self.__name = name

    def set_type(self,stateType : Type_of_state):
        self.__Type_of_state = stateType

    def set_bbox(self,bbox : tuple):
        self.__bbox = bbox
        Xt,Yt,Xb,Yb = bbox
        self.__top_left = (Xt,Yt) # top coordinate
        self.__bottom_right = (Xb,Yb) # bottom coordinate

    # setters for top_left & bottom right
    # ...

    # getter
    def get_name(self):
        return self.__name
    
    def get_type(self):
        return self.__Type_of_state
    
    def get_top_coordinate(self):
        return self.__top_left
    
    def get_bottom_coordinate(self):
        return self.__bottom_right
    
    def get_bbox(self):
        # * used to unpack the tuple
        # return (*self.__top_left,*self.__bottom_right)
        return self.__bbox
    
    
# ------------------ testing --------------------- #
# test= state('q0',Type_of_state.Start_State,(2,2,5,5))

# print(f'The class is {test}')
# print(f'The name of state is {test.get_name()}')
# print(f'The type of state is {test.get_type().name} and its enumeration is {test.get_type().value}')
# print(f'The top left point is {test.get_top_coordinate()} & the bottom right bottom {test.get_bottom_coordinate()}')
# print(f'The bounding box is {test.get_bbox()}')
