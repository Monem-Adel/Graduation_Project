# import sys
import State, Arrow

# sys.path.append(r'H:\Scanner')
# print(sys.path)

# Note: the prefix  of identifier is notation means that id is private
# transition class
class transition:

    # constructor
    def init(self,label, bbox:tuple, arrow:Arrow.arrow, source:State.state = State.state(), destination:State.state= State.state()):
    # def init(self,label, bbox:tuple, arrow:Arrow.arrow, source:State.state = None, destination:State.state= None):
    # def init(self,label, bbox:tuple, arrow:Arrow.arrow):
        self.label = label
        self.source = source
        self.destination = destination
        self.arrow = arrow
        # bounding box (xt,yt,xb,yb) t: top left, r: bottom right
        self.bbox = bbox
        Xt,Yt,Xb,Yb = bbox
        self.top_left = (Xt,Yt) # top coordinate
        self.bottom_right = (Xb,Yb) # bottom coordinate
        # the tensor of the image (if need1ed)

    #another constructor to intialize without arrow class and add direction
    def init(self,label, bbox:tuple,direction , source:State.state = State.state(), destination:State.state= State.state()):
    # def init(self,label, bbox:tuple, arrow:Arrow.arrow, source:State.state = None, destination:State.state= None):
    # def init(self,label, bbox:tuple, arrow:Arrow.arrow):
        self.label = label
        self.source = source
        self.destination = destination
        # bounding box (xt,yt,xb,yb) t: top left, r: bottom right
        self.bbox = bbox
        self.direction = direction
        Xt,Yt,Xb,Yb = bbox
        self.top_left = (Xt,Yt) # top coordinate
        self.bottom_right = (Xb,Yb) # bottom coordinate
        # the tensor of the image (if needed)
    
    # setters
    def set_label(self,label):
        self.label = label

    def set_source(self,source:State.state):
        self.source = source

    def set_destination(self,destination:State.state):
        self.destination = destination

    def set_arrow(self,arrow:Arrow.arrow):
        self.arrow = arrow

    def set_bbox(self,bbox : tuple):
        self.bbox = bbox
        Xt,Yt,Xb,Yb = bbox
        self.top_left = (Xt,Yt) # top coordinate
        self.bottom_right = (Xb,Yb) # bottom coordinate
    
    def set_direction(self, direction):
        self.direction = direction

    # setters for top_left & bottom right
    # ...
    
    # getters
    def get_label(self):
        return self.label
    
    def get_source(self):
        return self.source
    
    def get_destination(self):
        return self.destination
    
    def get_arrow(self):
        return self.arrow
    
    def get_top_coordinate(self):
        return self.top_left
    
    def get_bottom_coordinate(self):
        return self.bottom_right
    
    def get_bbox(self):
        return self.bbox
        # return (*self.top_left,*self.bottom_right)

    def get_direction(self):
        return self.direction
    # a method to return a tuple of source & destination
    def get_sourceAndDistination(self):
        return (self.source,self.destination)
    
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