import sys
sys.path.append(r"D:\4th\Second Sem\Graduation Project\Graduation_Project\Scanner")
sys.path.append(r"H:\Graduation Project\Graduation_Project\Scanner")
# print(sys.path)
# Import file2.py
import State
from State import state



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

def find_state(state , table):
    # if isinstance(state_, state):
    i = 1
    index = -1
    n = len(table)
    while i < n :
        table_content = table[i][0]
        if table_content == state:
            index = i
        i+=1
    return index


class parser:
    def __init__(self , start_state: State.state, transition_table):
        self.start_state = start_state
        self.transition_table = transition_table

    def get_start_state(self):
        return self.start_state

    def set_start_state(self, state: State.state):
        self.start_state = state

    def start_state_is_initial(self):
        return self.start_state.get_type().value == 0


    def evaluate_input(self, test_string):
        i = 0
        n = len(test_string)
        current_state = self.start_state
        while i < n:
            col_index = find(test_string[i], self.transition_table, "column")
            row_index = find_state(current_state, self.transition_table)
            if row_index == -1 or col_index == -1:
                return False
            current_state = self.transition_table[row_index][col_index]
            i = i+1

        if current_state.get_type().value != 2:
            return False
        else:
            return True

    
    def test_strings (self , test_cases):
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
