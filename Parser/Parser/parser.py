# class State:
#     def __init__(self, name, state_type):
#         self.name = name
#         self.state_type = state_type  # 0 = intial , 1 = normal , 2 = final , 3 = error
#
#
#     def set_name(self,name):
#         self.name = name
#
#     def set_state_type(self, st_type):
#         self.state_type
#
#     def get_state_name(self):
#         return self.name
#
#     def get_state_type(self):
#         return self.state_type
#
#
# q0 = State("q0", 0) #intial
# q1 = State("q1", 1) #normal
# q2 = State("q2", 2) #final
# qerror = State("qerror", 3)
#
#

def find(character ,table, direction ):
    i = 1
    index = -1
    if direction =="row":
        n = len(table)
        while i < n :
            table_content = table[i][0]
            if table_content == character:
                index = i
            i+=1
    elif direction =="column":
        n = len(table[0])
        while i < n:
            table_content = table[0][i]
            if table_content == character:
                index = i
            i+=1

    return index


transition_table = [
    [" ", 0, 1],
    [q0, q1, qerror],
    [q1, q2, q1],
    [q2, q2, qerror]
]

class Parser:
    def __init__(self , start_state: state, transition_table):
        self.start_state = start_state
        self.transition_table = transition_table

    def get_start_state(self):
        return self.start_state

    def set_start_state(self, state: state):
        self.start_state = state

    def start_state_is_initial(self):
        return self.start_state.get_state_type() == 0


    def evaluate_input(self, test_string):
        i = 0
        n = len(test_string)
        current_state = self.start_state
        while i<n:
            col_index = find(int(test_string[i]), self.transition_table, "column")
            row_index = find(current_state, self.transition_table, "row")
            if row_index == -1 or col_index == -1:
                return False
            current_state = self.transition_table[row_index][col_index]
            i = i+1

        if current_state.get_state_type() != 2:
            return False
        else:
            return True

    
    def test_strings (self , test_cases ):
        results = []
        for string in test_cases:
            results.append(self.evaluate_input(string))
        return results


    def print_results (self , results_list):
        n = len(results_list)
        i = 0
        while(i<n):
            print(f'test case no {i} result is: {results_list[i]}')
            i = i+1




