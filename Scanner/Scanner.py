# this file for connecting classification with automaton (scanner)
# بص هنحط كله ف فنكشنز وفي واحده منهم هترجع اوتوماتون بعد ما يتملي
# ونظبط االسورس والجداول (او هترجع الترنزيشن تيبول)
# والفنكشن ده هتتناده في البرسر 
# two functions one for scanner & another for parser
import sys
# sys.path.append(r"D:\\3loom\\4thYear\\2ndSemester\\GraduationProject\\Graduation_Project\\Image_Processing")
# sys.path.append(r"D:\\3loom\\4thYear\\2ndSemester\\GraduationProject\\Graduation_Project")
sys.path.append(r"H:\Graduation Project\Graduation_Project\Parser")
sys.path.append(r"H:\Graduation Project\Graduation_Project\Image_Processing")

# print(sys.path)

import Automaton
import State
import Transition
import Arrow
import Models
import Parser

# method to take a path of image from the user
def take_pic():
    ##### BASSAM #####
    # you can change this method or link it to the flutter
    img_path = input("Enter your image path")
    img_path.lower().strip()
    return img_path


# img_path = 'H:\Graduation Project\Graduation_Project\Image_Processing\jflab\jflab2.png'
# img_path = "D:\\3loom\\4thYear\\2ndSemester\\GraduationProject\\Graduation_Project\\Scanner\\test.png"

# (1st) method to scan the image & initialize the automaton (scanner method called once in the program)
def scan_auto(img_path):

    # assign classified objects into lists & traversing them to create objects from specified classes

    listOfDict_states = Models.get_classified_States(img_path)
    listOfDict_trans = Models.get_classified_transitions(img_path)

    scannered_states = []
    scannered_trans = []
    scannered_labels = set()

    # the dict of classified states is :-
    # {'type': 'Final_State', 'bbox': (362, 130, 393, 197), 'name': ''}
    # t => type
    # bb => bbox
    # n => name
    
    for state in listOfDict_states:
        t = state['type']
        if(t == 'Start_State'):
            t = State.Type_of_state.Start_State
        elif(t == 'State'):
            t = State.Type_of_state.Normal_State
        elif(t == 'Final_State'):
            t = State.Type_of_state.Final_State
        bb = state['bbox']
        n = state['name']

        scannered_states.append(State.state(n,t,bb))

    # the dict of classified transitions is :-
    # {'type': 'Transition', 'bbox': (141, 241, 242, 262), 'Label': '2 4', 'Direction': 'left arrow'}
    # d => direction
    # bb => bbox
    # l => label

    for trans in listOfDict_trans:
        if(trans['type'] == 'Loop'):
            d = Transition.Arrow.Direction.Loop
        else:
            H = trans['Head']
            T = trans['Tail']
            d = trans['Direction']
            if(d == 'Left'):
                d = Transition.Arrow.Direction.Left
            elif (d =="Right"):
                d = Transition.Arrow.Direction.Right
            elif (d =="UP"):
                d = Transition.Arrow.Direction.Up
            elif (d =="Down"):
                d = Transition.Arrow.Direction.Down
        bb = trans['bbox']
        l = trans['Label']
        
        scannered_labels.add(l)
            
        scannered_trans.append(Transition.transition(l,bb,Arrow.arrow(d,H,T)))

    # create an automaton
    auto = Automaton.automaton(scannered_states,scannered_trans,scannered_labels,img_path)

    table = auto.make_transition_table()
    start_state = auto.get_start_state()

    # return tuple (automaton object, transition table, start state)
    return (auto,table,start_state)
    # return auto

# auto ,t,st = scan_auto(r"H:\Graduation Project\Graduation_Project\Drawing_Automatons\auto_2.jpg")
# print(auto.get_states())
# print(auto.get_transitions())
# print(st)
# print('#'*10)
# print(t)

# print(scan_auto(img_path))

# (2nd) method to parse the automaton 
def parse_auto(start_state, transition_table: list,test_cases):
    parser  = Parser.parser(start_state, transition_table)
    results = parser.test_strings(test_cases)
    return results


def test_all():
    test_cases = ["0","10","0011", "1001"]
    # image_path = "D:\\3loom\\4thYear\\2ndSemester\\GraduationProject\\Graduation_Project\\Scanner\\test.png"
    image_path = take_pic()
    image_path.lower().strip()
    automaton, transition_table, start_state = scan_auto(image_path)
    results = parse_auto(start_state, transition_table, test_cases)
    i = 0
    for result in results:
        print(f"result for test case no {i} is {result}")
        i = i+1

test_all()