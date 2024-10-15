import random
from copy import deepcopy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        # drop_phase = True   # TODO: detect drop phase
        

        # if not drop_phase:
        #     # TODO: choose a piece to move and remove it from the board
        #     # (You may move this condition anywhere, just be sure to handle it)
        #     #
        #     # Until this part is implemented and the move list is updated
        #     # accordingly, the AI will not follow the rules after the drop phase!
        #     pass

        # # select an unoccupied space randomly
        # # TODO: implement a minimax algorithm to play better
        # move = []
        # (row, col) = (random.randint(0,4), random.randint(0,4))
        # while not state[row][col] == ' ':
        #     (row, col) = (random.randint(0,4), random.randint(0,4))

        # # ensure the destination (row,col) tuple is at the beginning of the move list
        # move.insert(0, (row, col))
        # return move
    
        move = self.Max_value(state, 0)[1]
        return move

    def detect_drop_phase(self):
        piece_count = 25 - sum(row.count(' ') for row in self.board)
        return True if piece_count < 8 else False
    

    def succ(self, state, piece):
        drop_phase = self.detect_drop_phase()
        steps = []
        succs = []
        
        for i in range(len(state)):
            for j in range(len(state[i])):
                if drop_phase:
                    if state[i][j] == ' ':
                        temp = deepcopy(state)
                        temp[i][j] = piece
                        succs.append(temp)
                        steps.append([(i, j)])
                elif state[i][j] == piece:
                    for k in range(max(i-1, 0), min(i+2, 5)):
                        for x in range(max(j-1, 0), min(j+2, 5)):
                            if state[k][x] == ' ' and (k != i or x != j):
                                temp = deepcopy(state)
                                temp[k][x] = piece
                                temp[i][j] = ' '
                                succs.append(temp)
                                steps.append([(k, x), (i, j)])
                        
        return succs, steps


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for row in range(2):
            for position in range(2):
                if state[row][position] != ' ' and state[row][position] == state[row+1][position+1] == state[row+2][position+2] == state[row+3][position+3]:
                    return 1 if state[row][position]==self.my_piece else -1
                    
        # check / diagonal wins
        for row in range(2):
            for position in range(3,5):
                if state[row][position] != ' ' and state[row][position] == state[row+1][position-1] == state[row+2][position-2] == state[row+3][position-3]:
                    return 1 if state[row][position]==self.my_piece else -1

        # check box wins
        for row in range(3):
            for position in range(3):
                if state[row+1][position+1] == ' ':
                    if state[row][position] != ' ' and state[row][position] == state[row][position+2] == state[row+2][position] == state[row+2][position+2]:
                        return 1 if state[row][position]==self.my_piece else -1
        return 0 # no winner yet

    def heuristic_game_value(self, state):
        val = self.game_value(state)
        if (val != 0):
            return val 
        mat = [[0 for x in range(5)] for y in range(2)] #0 for black, 1 for red
        loc = [[] for y in range(2)] #stored as (y,x)
             
        for i in range(len(state)):
            for j in range(len(state[i])):
                if(state[i][j] == self.pieces[0]):
                    loc[0].append([i,j])
                elif(state[i][j] == self.pieces[1]):
                    loc[1].append([i,j])
 
        for p in range(2):
            for x in range(len(loc[p])):
                i = loc[p][x][0]
                j = loc[p][x][1]
            
                # check horizontal
                if(j <= 3):
                    if(state[i][j] == state[i][j+1]):
                        if( j < 3 and state[i][j+1] == state[i][j+2]):
                            mat[p][0] = 3    
                        #elif(mat[p][0] < 2):
                        else:
                            mat[p][0] = 2
                    elif(j < 3 and state[i][j] == state[i][j+2]):
                        mat[p][0] = 1
                            
                # check vertical                     
                if(i <= 3): 
                    if(state[i][j] == state[i+1][j]):
                        if(i < 3 and state[i+1][j] == state[i+2][j]):
                            mat[p][1] = 3
                        else:
                            mat[p][1] = 2
                    elif(i < 3 and state[i][j] == state[i+2][j]):
                        mat[p][1] = 1
                        
                # check \ diagonal                 
                if(i <= 3 and j <= 3):
                    # 2 pieces
                    if(state[i][j] == state[i+1][j+1]):
                        mat[p][2] = 2                                    
                    if(i <= 2 and j <= 2):
                        #3 pieces 
                        if(state[i][j] == state[i+1][j+1] == state[i+2][j+2]):
                            mat[p][2] = 3
                        # 2 pieces + gap
                        elif(state[i][j] == state[i+2][j+2] and mat[p][2] < 1):
                            mat[p][2] = 1 
                            
                # check / diagonal               
                if(i <= 3 and j >= 1):                    
                    # 2 pieces
                    if(state[i][j] == state[i+1][j-1]):
                        mat[p][3] = 2                                    
                    if(i <= 2 and j >= 2):
                        # 3 pieces
                        if(state[i][j] == state[i+1][j-1] == state[i+2][j-2]):
                            mat[p][3] = 3
                        # 2 pieces + gap
                        elif(state[i][j] == state[i+2][j-2]):
                            mat[p][3] = max(1,mat[p][3])
                            
            #check box
            for a in range(0,4):
                for b in range(0,4):
                    count = 0
                    if(state[a][b] == self.pieces[p]):
                        count += 1
                    if(state[a][b+1] == self.pieces[p]):
                        count += 1
                    if(state[a+1][b] == self.pieces[p]):
                        count += 1
                    if(state[a+1][b+1] == self.pieces[p]):
                        count += 1
                    mat[p][4] = count
        
        if(self.my_piece == self.pieces[0]):
            l = 0
            r = 1
        else:
            l = 1
            r = 0      
        return (max(mat[l]) - max(mat[r])) / 4
  


    def Max_value(self, state, depth):        
        steps = []
        val = self.heuristic_game_value(state)
        if (val == -1 or val == 1 or depth == 3):
            return val, []            
        else:
            global_max = -float('inf')
            succs = self.succ(state, self.my_piece)
            succ = succs[0]
            steps = succs[1]
            for i in range(len(succ)): 
                min_val = self.min_value_helper(succ[i], depth + 1)
                #print(min_val)
                if(min_val > global_max):
                    max_index = i
                    global_max = min_val
                    if global_max == 1:
                        break
            return (global_max, steps[max_index])
                
            
            
    #Opposite of Max_value to help the max value function     
    def min_value_helper(self, state, depth):
        val = self.heuristic_game_value(state)
        if(val == -1 or val == 1 or depth == 3):
            return val
        else:
            global_min  = float('inf')
            succ = self.succ(state, self.opp)[0]
            for i in range(len(succ)):
                max_val = self.Max_value(succ[i], depth + 1)[0]
                if(max_val < global_min ):
                    min_index = i
                    global_min  = max_val
                    if global_min  == -1:
                        break
            return global_min 

        



############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
