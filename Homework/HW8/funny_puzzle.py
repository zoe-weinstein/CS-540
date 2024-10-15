import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """

    grid = int(len(from_state)**0.5)
    # Init Manhattan distance
    distance = 0
    # Loop through grid position
    for i in range(grid):
        for j in range(grid):
            curr_tile = from_state[i*grid+j]
            if curr_tile == 0:
                continue
            goal_i, goal_j = divmod(to_state.index(curr_tile), grid)
            distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []
    grid = int(len(state) ** 0.5)

    for i in range(grid):
        for j in range(grid):
            curr_tile = state[grid * i + j]

            if curr_tile == 0:
                if i == 0 and state[grid * i + j + 3] != 0:
                    state1 = list(state)
                    state1[grid * i + j], state1[grid * i + j + 3] = state1[grid * i + j + 3], state1[grid * i + j]
                    succ_states.append(state1)

                if i == 1 and state[grid * i + j - 3] != 0:
                    state2 = list(state)
                    state2[grid * i + j], state2[grid * i + j - 3] = state2[grid * i + j - 3], state2[grid * i + j]
                    succ_states.append(state2)

                if i == 1 and state[grid* i + j + 3] != 0:
                    state3 = list(state)
                    state3[grid * i + j], state3[grid * i + j + 3] = state3[grid * i + j + 3], state3[grid * i + j]
                    succ_states.append(state3)

                if i == 2 and state[grid * i + j - 3] != 0:
                    state4 = list(state)
                    state4[grid * i + j], state4[grid * i + j - 3] = state4[grid * i + j - 3], state4[grid * i + j]
                    succ_states.append(state4)

                if j == 0 and state[grid * i + j + 1] != 0:
                    state5 = list(state)
                    state5[grid * i + j], state5[grid * i + j + 1] = state5[grid * i + j + 1], state5[grid * i + j]
                    succ_states.append(state5)

                if j == 1 and state[grid * i + j - 1] != 0:
                    state6 = list(state)
                    state6[grid * i + j], state6[grid * i + j - 1] = state6[grid * i + j - 1], state6[grid * i + j]
                    succ_states.append(state6)

                if j == 1 and state[grid * i + j + 1] != 0:
                    state7 = list(state)
                    state7[grid * i + j], state7[grid * i + j + 1] = state7[grid * i + j + 1], state7[grid * i + j]
                    succ_states.append(state7)

                if j == 2 and state[grid * i + j - 1] != 0:
                    state8 = list(state)
                    state8[grid * i + j], state8[grid * i + j - 1] = state8[grid * i + j - 1], state8[grid * i + j]
                    succ_states.append(state8)

    return sorted(succ_states)



def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    pq = []
    visit_record = []
    
    g = 0
    count = 0
    h = get_manhattan_distance(state, goal_state)
    weight = g + h
    b = (weight, state, (g, h, -1))
         
    heapq.heappush(pq, b)

    while True:
        if len(pq) == 0:
            break
        b = heapq.heappop(pq)

        if b[1] == goal_state:
            steps = list()
            while True:
                steps.insert(0, b[1])
                if b[2][2] != -1:
                    b = visit_record[b[2][2]]
                else:
                    break

            for i in range(len(steps)):
                print(str(steps[i]) +
                      " h=" +
                      str(get_manhattan_distance(steps[i])) +
                      " moves: " +
                      str(i))
            print("Max queue length:", len(pq) + 1)
            break

        else:
        # Add the current state to the visit record
            visit_record.append(b)
            g = b[2][0] + 1
            succs = get_succ(b[1])
        # Iterate over each successor state
            for s in succs:
                current = -1
                for i in range(len(visit_record)):
                    if visit_record[i][1] == s:
                        current = i
                        break
                if current == -1:
                    # calc heuristic
                    h = get_manhattan_distance(s)
                    heapq.heappush(pq, (g + h, s, (g, h, count)))
                else:
                    h = get_manhattan_distance(s)
                    if visit_record[current][0] > g + h:
                        visit_record[current] = (g + h, s, (g, h, count))
                        heapq.heappush(pq, visit_record[current])
            count += 1


    # This is a format helper，which is only designed for format purpose.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.

    # for state_info in state_info_list:
    #     current_state = state_info[0]
    #     h = state_info[1]
    #     move = state_info[2]
    #     print(current_state, "h={}".format(h), "moves: {}".format(move))
    # print("Max queue length: {}".format(max_length))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()

    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    solve([2,5,1,4,0,6,7,0,3])
    print()

    solve([4,3,0,5,1,6,7,2,0])
    print()