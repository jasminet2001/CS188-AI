import copy
import itertools
import math
import heapq


class GameSolution:
    """
        A class for solving the Water Sort game and finding solutions(normal, optimal).

        Attributes:
            ws_game (Game): An instance of the Water Sort game which implemented in game.py file.
            moves (List[Tuple[int, int]]): A list of tuples representing moves between source and destination tubes.
            solution_found (bool): True if a solution is found, False otherwise.

        Methods:
            solve(self, current_state):
                Find a solution to the Water Sort game from the current state.
                After finding solution, please set (self.solution_found) to True and fill (self.moves) list.

            optimal_solve(self, current_state):
                Find an optimal solution to the Water Sort game from the current state.
                After finding solution, please set (self.solution_found) to True and fill (self.moves) list.
    """

    def __init__(self, game):
        """
            Initialize a GameSolution instance.
            Args:
                game (Game): An instance of the Water Sort game.
        """
        self.ws_game = game  # An instance of the Water Sort game.
        # A list of tuples representing moves between source and destination tubes.
        self.moves = []
        # Number of tubes in the game.
        self.tube_numbers = game.NEmptyTubes + game.NColor
        # True if a solution is found, False otherwise.
        self.solution_found = False
        self.visited_tubes = set()  # A set of visited tubes.

    def can_move(self, current_state, moves):
        # empty space and source color = dest color
        if (len(current_state[moves[1]]) < self.ws_game.NColorInTube and len(current_state[moves[0]]) > 0):
            # bekhter indexout of range joda ke age khali bud maghsad ke dige okaye
            if (len(current_state[moves[1]]) == 0):
                return True
            elif (current_state[moves[0]][-1] == current_state[moves[1]][-1]):
                return True
            else:
                return False
        else:
            return False

    def solve(self, current_state):
        c_state = copy.deepcopy(current_state)

        if self.ws_game.check_victory(c_state) or self.solution_found:
            self.solution_found = True
            return
        else:
            n = 0

            for i in range(len(c_state)):
                if self.solution_found:
                    return
                for j in range(i+1, len(c_state)):
                    if self.solution_found:
                        return

                    n += 1
                    if (self.can_move(c_state, (i, j))):

                        temp = copy.deepcopy(c_state)
                        while (self.can_move(temp, (i, j))):

                            # akhe move current state ro avaz nemikone
                            temp[j].append(temp[i][-1])
                            temp[i].pop()
                        if not (str(temp) in self.visited_tubes):
                            c_state = temp
                            self.moves.append((i, j))
                            self.visited_tubes.add(str(c_state))
                            if (self.ws_game.check_victory(c_state)):
                                self.solution_found = True
                                return
                            else:
                                self.solve(c_state)

                    if (self.can_move(c_state, (j, i))):
                        temp2 = copy.deepcopy(c_state)

                        while (self.can_move(temp2, (j, i))):
                            temp2[i].append(temp2[j][-1])
                            temp2[j].pop()

                        if not (str(temp2) in self.visited_tubes):
                            c_state = temp2
                            self.moves.append((j, i))
                            self.visited_tubes.add(str(c_state))
                            if (self.ws_game.check_victory(c_state)):
                                self.solution_found = True

                                return
                            else:
                                self.solve(c_state)

            if self.solution_found:
                return
            elif len(self.moves) > 0:
                (a, b) = self.moves.pop()
                c_state[a].append(c_state[b][-1])
                c_state[b].pop() if c_state[b] else None
                self.solve(c_state)
            else:
                self.solution_found = False
                return
        pass

    def heuristic(self, current_state):
        count = 0
        for i in range(len(current_state)):
            if len(current_state[i]) > 1:
                tcount = current_state[i][0]  # tube count
                for j in range(1, len(current_state[i])):
                    if not (current_state[i][j] == tcount):
                        count += 1
                        tcount = (current_state[i][j])

        return count

    def optimal_solve(self, current_state):
        c_state = copy.deepcopy(current_state)
        hs = {str(c_state): (self.heuristic(c_state))}  # dectionary h ha
        gs = {str(c_state): 0}  # dectionary g ha
        pq = []
        heapq.heappush(
            pq, ((hs[str(c_state)]+gs[str(c_state)]), (c_state, [])))
        s_move = []
        state = []
        b = 0
        while len(pq) > 0:

            temp = heapq.heappop(pq)
            f = temp[0]
            state = temp[1][0]
            s_move = temp[1][1]
            last_g = gs[str(state)]
            self.visited_tubes.add(str(state))
            if (self.ws_game.check_victory(state)):
                self.solution_found = True
                self.moves = s_move
                return
            elif not self.solution_found:
                for i in range(len(state)):
                    for j in range(i+1, len(state)):
                        state_t = copy.deepcopy(state)
                        b += 1
                        if (self.can_move(state_t, (i, j))):
                            temp = copy.deepcopy(state_t)

                            while (self.can_move(temp, (i, j))):
                                temp[j].append(temp[i][-1])
                                temp[i].pop()
                            new_move = s_move + [(i, j)]
                            new_g = (int(last_g)+1)
                            itExists = str(temp) in self.visited_tubes

                            # age bude bashe h ha yeki
                            if not itExists or (itExists and new_g < gs[str(temp)]):
                                state_t = temp
                                new_f = self.heuristic(temp)
                                gs[str(state_t)] = new_g
                                hs[str(state_t)] = new_f
                                heapq.heappush(
                                    pq, ((new_f + new_g), (state_t, new_move)))
                                self.visited_tubes.add(str(state_t))
                        if (self.can_move(state_t, (j, i))):
                            temp = copy.deepcopy(state_t)
                            while (self.can_move(temp, (j, i))):

                                temp[i].append(temp[j][-1])
                                temp[j].pop()
                            new_move = s_move + [(j, i)]
                            new_g = (int(last_g)+1)
                            itExists = str(temp) in self.visited_tubes
                            if not itExists or (itExists and new_g < gs[str(temp)]):
                                state_t = temp
                                new_f = self.heuristic(temp)
                                gs[str(state_t)] = new_g
                                hs[str(state_t)] = new_f
                                heapq.heappush(
                                    pq, ((new_f + new_g), (state_t, new_move)))
                                self.visited_tubes.add(str(state_t))

        self.solution_found = False
        print("no answer")
        pass
