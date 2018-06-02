import numpy as np

LENGTH = 5
NUM_IN_LINE = 3

class Environment:

    board = None
    x = None
    o = None
    empty_val = None
    winner = None
    ended = None
    state = 0

    def __init__(self):
        self.x = - 1
        self.o = 1
        self.empty_val = 0
        self.board = np.zeros((LENGTH, LENGTH))
        self.winner = None
        self.ended = False
        self.num_states = 3**(LENGTH*LENGTH)

    def is_empty(self, i, j):
        return self.board[i, j] == self.empty_val

    def reward(self, symbol):
        if not self.game_over():
            return 0
        return 1 if self.winner == symbol else 0

    def get_state(self):
        k = 0
        h = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i, j] == self.empty_val:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3**k) * v
                k += 1
        return h

    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended
        for i in range(LENGTH):
            for player in (self.x, self.o):
                win_combo = player*NUM_IN_LINE
                if self.board[i].sum() == win_combo or \
                        self.board[:, i].sum() == win_combo or \
                        self.board.trace() == win_combo or \
                        np.fliplr(self.board).trace() == win_combo:
                    self.winner = player
                    self.ended = True
                    return True
        if np.all((self.board == 0) == False):
            self.winner = None
            self.ended = True
            return True
        self.winner = None
        return False

    def draw_board(self):
        for i in range(LENGTH):
            print("---------------")
            curr_row = ""
            for j in range(LENGTH):

                if self.board[i, j] == self.x:
                    curr_row += "x |"
                elif self.board[i, j] == self.o:
                    curr_row += "o |"
                else:
                    curr_row += "  |"
            print(curr_row)
        print("---------------")


class Agent:
    eps = None
    alpha = None
    verbose = None
    state_history = []
    V = None
    sym = None

    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []

    def setV(self, V):
        self.V = V

    def set_symbol(self, symbol):
        self.sym = symbol

    def set_verbose(self, verbose):
        self.verbose = verbose

    def reset_history(self):
        self.state_history = []

    def take_action(self, _env: Environment):
        # Choose action based epsilon-greedy strategy
        r = np.random.rand()
        best_state = None
        if r < self.eps:
            if self.verbose:
                print("Taking a random action")
            possible_moves = []
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if _env.is_empty(i, j):
                        possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            pos2value = {}  # just for debugging
            next_move = None
            best_value = -1
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if _env.is_empty(i, j):
                        _env.board[i, j] = self.sym
                        state = _env.get_state()
                        _env.board[i, j] = _env.empty_val
                        pos2value[(i, j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i, j)
            if self.verbose:
                print("Taking a greedy action")
                print("---------------")
                for i in range(LENGTH):
                    curr_row = ""
                    for j in range(LENGTH):

                        if _env.board[i, j] == _env.x:
                            curr_row += "x |"
                        elif _env.board[i, j] == _env.o:
                            curr_row += "o |"
                        else:
                            curr_row = curr_row + (" %.2f|" % pos2value[(i, j)]) +" |"
                    print(curr_row)
                print("---------------")
        _env.board[next_move[0], next_move[1]] = self.sym

    def update_state_history(self, _state):
        self.state_history.append(_state)

    def update(self, _env):
        reward = _env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha*(target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()


class Human:
    sym = None

    def __init__(self):
        pass

    def set_symbol(self, sym):
        self.sym = sym

    def take_action(self, env):
        while True:
            move = input("Enter coords (i=0..2,j=0..2) for your next move:")
            i,j = move.split(",")
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.board[i, j] = self.sym
                break

    def update(self, _env):
        pass

    def update_state_history(self, _state):
        pass

def generate_all_binary_numbers(N):
    results = []
    if N > 1:
        child_results = generate_all_binary_numbers(N-1)
    else:
        child_results = ['']
    for prefix in ('0', '1'):
        for suffix in child_results:
            new_result = prefix + suffix
            results.append(new_result)
    return results


def get_state_hash_and_winner(env: Environment, i=0, j=0):
    results = []
    for v in (env.empty_val, env.x, env.o):
        env.board[i, j] = v
        if j == 2:
            if i == 2:
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i+1, 0)
        else:
            results += get_state_hash_and_winner(env, i, j+1)
    return results


def initialV_x(env, state_winner_triples):
    return initialV(env.x, env, state_winner_triples)


def initialV_o(env, state_winner_triples):
    return initialV(env.o, env, state_winner_triples)


def initialV(player, env, state_winner_triples):
    # initialize state values as follows:
    # if x wins -> V(s) = 1
    # if x losses or draw -> V(s) = 0
    # otherwise, V(s) =0.5
    V= np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == player:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def play_game(p1: Agent, p2: Agent, env: Environment, draw=False):
    current_player = None
    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()
        current_player.take_action(env)
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
    if draw:
        env.draw_board()
    p1.update(env)
    p2.update(env)
    return env.winner

if __name__ == "__main__":
    tp1 = Agent()
    tp2 = Agent()
    tenv = Environment()

    state_winner_triples = get_state_hash_and_winner(tenv)
    for curr_state_winner in state_winner_triples:
        print(curr_state_winner)

    Vx = initialV_x(tenv, state_winner_triples)
    tp1.setV(Vx)
    Vo = initialV_o(tenv, state_winner_triples)
    tp2.setV(Vo)

    tp1.set_symbol(tenv.x)
    tp2.set_symbol(tenv.o)

    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print("> Episode #%d" % t)
        play_game(tp1, tp2, Environment())
    print("Training done")
    human = Human()
    human.set_symbol(tenv.o)
    while True:
        tp1.set_verbose(True)
        winner = play_game(tp1, human, Environment(), draw=2)
        if winner == tenv.x:
            print("YOU LOOSE! GAME OVER")
        elif winner == tenv.o:
            print("YOU WIN! GAME OVER...")
        else:
            print("DRAW! GAME OVER...")
        contAnswer = input("Play again? [Y/n]:")
        if contAnswer and contAnswer.lower()[0] == "n":
            break

