from math import inf, sqrt
import State, Transition, Arrow

 # If we want only the first item from our list that satisfies our condition,
    # we change our list comprehension into a call to the Python next function.
    # If next does not find a value, it will throw a StopIteration exception.
    # To prevent it from doing so, we can provide a default value to return as an additional argument.
    # next(iterator, default)
    # Retrieve the next item from the iterator by calling its __next__() method.
    # If default is given, it is returned if the iterator is exhausted, otherwise StopIteration is raised.

class automaton:

    # flag to indicate there is only start state in the list or not (parser)
    # to check if there is more start state
    @staticmethod
    def isUnique(stateList): 
        count = 0
        for state in stateList:
            if(state.get_type() == State.Type_of_state.Start_State):
                count+=1
        return (count == 1)

    # constructor
    def __init__(self, states : State.state, transitions : Transition.transition, labels : set, imagePath, image) :
        if(not automaton.isUnique(states)):
            raise Exception("ERROR: the start state isn't unique or not found")
        else : self.__states = states
        self.__transitions = transitions
        self.__labels = labels
        self.__imagePath = imagePath # for keeping the path or the name of the image
        self.__image = image # assign the tensor of the image
    

    # setters
    # to set all list
    def set_states(self, states : State.state):
        self.__states = states

    # HERE DEFINE a method to set one state in the list (if needed)(insert,delete,replace,...)

    def set_transitions(self, transitions : Transition.transition):
        self.__transitions = transitions

    def set_labels(self, labels : set):
        self.__labels = labels

    def set_imagePath(self, imagePath):
        self.__imagePath = imagePath

    def set_image(self, image):
        self.__image = image

    # getters
    def get_states(self):
        return self.__states
    
    def get_transitions(self):
        return self.__transitions

    def get_labels(self):
        return self.__labels
    
    def get_imagePath(self):
        return self.__imagePath
    
    def get_image(self):
        return self.__image
    
    # method to get the start state
    def get_start_state(self):
        if (not automaton.isUnique(self.__states)):
            raise Exception("ERROR: the start state isn't unique or not found")
        startState = next((state for state in self.__states if state.get_type() == State.Type_of_state.Start_State))
        return startState
    
    # method to compute the distance between two given points
    # static method called by the class; don't need to declare an object
    @staticmethod
    def compute_distance(point_1 : tuple, point_2:tuple):
        x1,y1 = point_1
        x2,y2 = point_2
        # using Euclidean distance
        distance = sqrt(((x2-x1)**2)+((y2-y1)**2))
        return distance

    # method to compare between to distances & return the minimum one (or bool flag)
    @staticmethod
    def compare_distances(distance_1,distance_2):
        # if (distance_1 <= distance_2):
        #     return distance_1
        # else:
        #     return distance_2
        # if you want to return boolean
        if (distance_1 <= distance_2):
            return True
        else:
            return False

    # method to loop over states with one given transition & return the nearest one
    # def nearest_states(arrow:Arrow.arrow, states:list[State.state]):
    def nearest_states(self,arw:Arrow.arrow):
        minDistance_T = inf # indicate to an infinity, T; for tail
        minDistance_H = inf # indicate to an infinity, H; for head
        # nearest = State.state() # may raise an error here cause no arguments in constructors
        # distance_back: distance between tail & source state
        # distance_front: distance between head & destination state
        for state in self.__states:
            # knowing the nearest state to the tail (defining source)
            distance_Back = automaton.compute_distance(arw.get_tail(),state.get_bottom_coordinate())
            if (automaton.compare_distances(distance_Back,minDistance_T)):
                minDistance_T = distance_Back
                source = state
            # knowing the nearest state to the head (defining destination)
            distance_front = automaton.compute_distance(arw.get_head(),state.get_top_coordinate()) 
            if (automaton.compare_distances(distance_front,minDistance_H)):
                minDistance_H = distance_front
                destination = state
        return (source,destination)  
        # Transition.transition.set_source(source)

    # method to prepare the source & destination for each arrow (transition) // cancelled //
    # def setting_src_des(self):
    #     for trans in self.__transitions:
    #         source, destination = self.nearest_states(trans.get_arrow())
    #         trans.set_source(source)
    #         trans.set_destination(destination)
#  Xt,Yt,Xb,Yb = bbox
#         self.__top_left = (Xt,Yt) # top coordinate
#         self.__bottom_right = (Xb,Yb) # bottom coordinate
    def setting_src_des(self):
        for trans in self.__transitions:
            min_source = 0
            min_destination =0
            tx0, ty0 ,tx1, ty1= trans.bbox #t are the coordinates of transition
            if trans.direction == 'right':
                for _state in self.__states: #t are the coordinates of states
                    sx0, sy0, sx1, sy1 = _state.get_bbox()
                    dis1 = self.compute_distance( (tx0, ty0), (sx1, sy1) )
                    if (dis1 < min_source):
                        min_source = dis1
                        trans.set_source(_state)
                    dis2 = self.compute_distance( (tx1, ty1), (sx0, sy0))
                    if(dis2 < min_destination):
                        min_destination = dis2
                        trans.set_destination(_state)
            elif trans.direction =='left':
                 for _state in self.__states: #t are the coordinates of states
                    sx0, sy0, sx1, sy1 = _state.get_bbox()
                    dis1 = self.compute_distance( (tx0, ty0), (sx1, sy1) )
                    if (dis1 < min_source):
                        min_source = dis1
                        trans.set_destination(_state)
                    dis2 = self.compute_distance( (tx1, ty1), (sx0, sy0))
                    if(dis2 < min_destination):
                        min_destination = dis2
                        trans.set_source(_state)


    
    # method to construct transition table (preparing 1st row & 1st column)
    def construct_table(self):
        table = [['##',*self.__labels]]
        # case: sort the alphabets in the list cause it come from set
        for row , item in zip(table[1:],range(len(self.__states))):
            row[0] = self.__states[item]
            # s = self.__states[item].get_name()
            # row[0]=s
        # if length of states > length of table[1:]
        if (len(self.__states) > len(table[1:])):
            shift_magnitude = len(table[1:]) # used to fill the remaining of states
            for item in range(len(self.__states)-shift_magnitude):
                table.append([self.__states[item+shift_magnitude]])
                # table.append([self.__states[item+shift_magnitude].get_name()])
        return table


    # method making transition table
    def make_transition_table(self):
        #هتعمل فور لوب تست فيها السورس والديستنيشن لكل ترنزيشن
        automaton.setting_src_des(self) # may raise error
        #هتجهز اول صف واول  عمود في التيبول
        table = automaton.construct_table(self)
        #هتأسين ف كل سيل ف الجدول الديستنيشن
        # case: assign each cell of table with dummy value
        # case: don't forget the NULL STRING (eppsilon)
        for trans in self.__transitions:
            # indices to insert the destination in
            column_states = [r[0] for r in table]
            row = column_states.index(trans.get_source())
            # row = self.__states.index(trans.get_source())+1 # we add +1 to skip the row 0 (1st row)
            col = table[0].index(trans.get_label())
            # Handle the case where the item is not found
            value = trans.get_destination()
            # index(item) => get the first index of the item
            # case: if you construct table with dummy values take care it would be assignment operation not insert operation
            # table[row][col] = value # assignment operation
            table[row].insert(col,value) # insert operation

        return table
        pass



#------------- TESTING ---------------------#
    #,State.state('q1',1,(6,5,7,6))
# statsss= [State.state('q0',0,(0.5,2,2,3)),State.state('q2',2,(10,2,11,3))]
# trans = Transition.transition('B',(2,2.5,10,2.75),Arrow.arrow((9.5,2.6),(2.5,2.6)))
# auto = automaton(statsss,[trans],{'A','B'},'###','####')
# auto.setting_src_des()
# print(auto.make_transition_table())

# print(trans.get_source().get_name())
# print(trans.get_destination().get_name())
# print(statsss[0])
# print(statsss[1])
# s,d = automaton.nearest_states(auto,trans.get_arrow())
# # print(statsss)
# # print(trans)
# print(s.get_bbox())
# print(d.get_bbox())

# statsss= [State.state('q2',2,(12,6,14,4)),State.state('q0',0,(1,6,2,4)),State.state('q1',1,(8,2,10,1))]
# trans = Transition.transition('0',(2,5.5,12,4.5),Arrow.arrow((11,5),(3,5)))
# auto = automaton(statsss,trans,{'A','B'},'###','####')
# s,d = automaton.nearest_states(auto,trans.get_arrow())
# # print(statsss)
# # print(trans)
# print(s.get_bbox())
# print(d.get_bbox())

#,State.state('q0',State.Type_of_state.Start_State)
# s = [State.state('q0',State.Type_of_state.Start_State),State.state('q1',State.Type_of_state.Normal_State),State.state('q0',State.Type_of_state.Start_State)]
# auto = automaton(s,trans,[s,1],'##','##')
# print(auto.get_start_state().get_name())