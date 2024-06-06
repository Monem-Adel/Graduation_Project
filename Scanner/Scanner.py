import sys
sys.path.append(r"D:\4th\Second Sem\Graduation Project\Graduation_Project\Parser\Parser")
sys.path.append(r"D:\4th\Second Sem\Graduation Project\Graduation_Project\Image_Processing")

import Automaton
import State
import Transition
import Arrow
import Models
import parser

def take_pic():

    img_path = input("Enter your image path").lower().strip()
    return img_path

def scan_auto(img_path):
    
    listOfDict_states = Models.get_classified_States(img_path)
    listOfDict_trans = Models.get_classified_transitions(img_path)

    scannered_states = []
    scannered_trans = []
    scannered_labels = set()

    
    for state in listOfDict_states:
        t = state['type']
        if t == 'Start_State':
            t = State.Type_of_state.Start_State
        elif t == 'State':
            t = State.Type_of_state.Normal_State
        elif t == 'Final_State':
            t = State.Type_of_state.Final_State
        bb = state['bbox']
        n = state['name']
        scannered_states.append(State.state(n, t, bb))

    
    for trans in listOfDict_trans:
        if trans['type'] == 'Loop':
            d = Transition.Arrow.Direction.Loop
        else:
            d = trans['Direction']
            if d == 'Left':
                d = Transition.Arrow.Direction.Left
            elif d == 'Right':
                d = Transition.Arrow.Direction.Right
            elif d == 'UP':
                d = Transition.Arrow.Direction.Up
            elif d == 'Down':
                d = Transition.Arrow.Direction.Down
            else:
                raise ValueError(f"Unknown direction: {d}")

        bb = trans['bbox']
        l = trans['Label']

        scannered_labels.add(l)
        scannered_trans.append(Transition.transition(l, bb, Arrow.arrow(direction=d)))

    
    auto = Automaton.automaton(scannered_states, scannered_trans, scannered_labels, img_path)

    
    table = auto.make_transition_table()
    start_state = auto.get_start_state()

    
    return (auto, table, start_state)

def parse_auto(start_state, transition_table, test_cases):

    parser_obj = parser.parser(start_state, transition_table)
    results = parser_obj.test_strings(test_cases)
    return results

def test_all():
    test_cases = ["0011", "1001"]
    image_pathh = "D:\\4th\\Second Sem\\Graduation Project\\Graduation_Project\\Image_Processing\\auto_2.jpg"
    image_path = image_pathh.lower().strip()
    automaton, transition_table, start_state = scan_auto(image_path)
    results = parse_auto(start_state, transition_table, test_cases)
    for i, result in enumerate(results):
        print(f"Result for test case no {i} is {result}")

test_all()
