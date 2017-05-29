"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    ## This is a risky agent approach. It places a heavier weigth on the
    ## opponent moves. This evaluation function causes the agent to chase 
    ## the opponent. The weight has been set to 2 based on information from the video.
    ## Ideally you should try different values, evaluate their success for several
    ## game iterations and choose based on the result.
    if game.is_loser(player =  player):
        return float("-inf")
    
    if game.is_winner(player = player):
        return float("inf")
    
    ## Number of moves left for each player.
    myMoves = len(game.get_legal_moves(player = player))
    opponentMoves = len(game.get_legal_moves(player = game.get_opponent(player = player)))
    
    ## Evaluation function. If the opponent has a large number of moves the
    ## the evaluation function will have a small value which will lead to that
    ## branch not being choosen. On the other hand if the number of opponent moves
    ## is small the evaluation function will have a larger and more desirable
    ## branch to choose. The evaluation function leads to prefering branches 
    ## where the opponent has less moves available that is the agent is chasing
    ## the opponent.
    evalFunction = myMoves - 2 * opponentMoves
    
    return float(evalFunction)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    ## This is a conservative agent approach. It places a heavier weigth on the
    ## player moves. This evaluation function causes the agent to get away from 
    ## the opponent. The weight has been chosen to replicate the behaviour of 
    ## the risky agent approach.
    if game.is_loser(player =  player):
        return float("-inf")
    
    if game.is_winner(player = player):
        return float("inf")
    
    ## Number of moves for each player.
    myMoves = len(game.get_legal_moves(player = player))
    opponentMoves = len(game.get_legal_moves(player = game.get_opponent(player = player)))
    
    ## Evaluation function. If the player has a large number of moves the
    ## the evaluation function will have a large value which will lead to that
    ## branch being choosen. On the other hand if the number of player moves
    ## is  small the evaluation function will have a small and less desirable
    ## branch to choose. The evaluation function leads to prefering branches 
    ## where the player has more moves available that is the agent is moving away
    ## from the opponent.
    evalFunction = 2 * myMoves - opponentMoves
    
    return float(evalFunction)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    ## This is a conservative agent. It is based on the ratio of moves available
    ## for the player compared to the moves available for the opponent.
    if game.is_loser(player =  player):
        return float("-inf")
    
    if game.is_winner(player = player):
        return float("inf")
    
    myMoves = len(game.get_legal_moves(player = player))
    opponentMoves = len(game.get_legal_moves(player = game.get_opponent(player = player)))
    
    if myMoves == 0:
        return float("-inf")
    
    if opponentMoves == 0:
        return float("inf")
    
    ## Evaluation function. If the player has a large number of moves compared
    ## to the number of opponent moves the evaluation function will have a
    ## large value which will lead to that branch being choosen. 
    ## On the other hand if the number of player moves
    ## is small compared to the number of opponent moves the evaluation function
    ## will have a small and less desirable branch to choose. 
    ## The evaluation function leads to prefering branches 
    ## where the player has more moves available compared to the opponent
    ## available moves. 
    evalFunction = myMoves / opponentMoves
    
    return float(evalFunction)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        v, bestMove = self.minimaxMax(game = game, depth = depth, currentDepth = 0)
        return bestMove
            
    def minimaxMin(self, game, depth, currentDepth):
        """Implements the logic to determine the min value.
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        
        currentDepth: int
            Level of the tree being analysed.

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        (int) Min value.
        """
        # Checks that the time left is greater than the threshold.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout.
        bestMove = (-1, -1)
        
        # The logic is applied to the active player only.
        if game.utility(game.active_player) != 0:
            return game.utility(game.inactive_player), bestMove
        
        # Returns the value and best move found if the analysed ply is the
        # same as the deepest ply to be analysed. 
        if currentDepth == depth:
            return self.score(game, self), bestMove
        
        # Min value logic.
        v = float("inf")
        
        for legalMove in game.get_legal_moves(game.active_player):
            newV, newMov = self.minimaxMax(game = game.forecast_move(legalMove), depth = depth, currentDepth = currentDepth + 1)
            if newV < v:
                v = newV
                bestMove = legalMove
        
        return v, bestMove
    
    def minimaxMax(self, game, depth, currentDepth):
        """Implements the logic to determine the max value.
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        
        currentDepth: int
            Level of the tree being analysed.

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        (int) Min value.
        """
        # Checks that the time left is greater than the threshold.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout.
        bestMove = (-1, -1)
        
        # The logic is applied to the active player only.
        if game.utility(game.active_player) != 0:
            return game.utility(game.inactive_player), bestMove
        
        # Returns the value and best move found if the analysed ply is the
        # same as the deepest ply to be analysed. 
        if currentDepth == depth:
            return self.score(game, self), bestMove
        
        # Min value logic.
        v = float("-inf")
        
        for legalMove in game.get_legal_moves(game.active_player):
            newV, newMov = self.minimaxMin(game = game.forecast_move(legalMove), depth = depth, currentDepth = currentDepth + 1)
            if newV > v:
                v = newV
                bestMove = legalMove
        
        return v, bestMove
        
        
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # Checks that there are legal moves left. If there are not
        # (-1, -1) is returned.
        if not game.get_legal_moves():
            return (-1, -1)
        
        try:
            # Iterative deepening implementation.
            depth = 1
            while True:
                bestMove = self.alphabeta(game = game, depth = depth)
                depth += 1
        except SearchTimeout:
            pass
        
        return bestMove

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # Checks that the time left is greater than the threshold.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Call to the alphabeta algorithm via the max value function.
        alpha, bestMove = self.alphabetamax(game, alpha, beta, depth, 0)
        
        return bestMove
    
    def alphabetamin(self, game, alpha, beta, depth, currentDepth):
        """Implements the logic to determine the min value.
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
            
        alpha: float
            Alpha limits the lower bound of search on minimizing layers.
            
        beta : float
            Beta limits the upper bound of search on maximizing layers.
            
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        
        currentDepth: int
            Level of the tree being analysed.

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        (int) Min value.
        """
        # Checks that the time left is greater than the threshold.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout.
        bestMove = (-1,-1)
        
        # The logic is applied to the active player only.
        if game.utility(game.active_player) != 0:
            return game.utility(game.inactive_player), bestMove
        
        # Returns the value and best move found if the analysed ply is the
        # same as the deepest ply to be analysed. 
        if currentDepth == depth:
            return self.score(game, game.inactive_player), bestMove
        
        # Min value logic.
        else:
            for legalMove in game.get_legal_moves(game.active_player):
                newV, newMov = self.alphabetamax(game.forecast_move(legalMove), depth = depth, alpha=alpha, beta=beta, currentDepth = currentDepth + 1)
                if newV <= alpha:
                    return newV, legalMove
                if newV < beta:
                    beta = newV
                    bestMove = legalMove
            return beta, bestMove
    
    def alphabetamax(self, game, alpha, beta, depth, currentDepth):
        """Implements the logic to determine the max value.
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
            
        alpha: float
            Alpha limits the lower bound of search on minimizing layers.
            
        beta : float
            Beta limits the upper bound of search on maximizing layers.
            
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        
        currentDepth: int
            Level of the tree being analysed.

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        (int) Min value.
        """
        # Checks that the time left is greater than the threshold.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout.
        bestMove = (-1,-1)

        # The logic is applied to the active player only.
        if game.utility(game.active_player) != 0:
            return game.utility(game.active_player), bestMove

        # Returns the value and best move found if the analysed ply is the
        # same as the deepest ply to be analysed. 
        if currentDepth == depth:
            return self.score(game, game.active_player), bestMove
        # Max value logic.
        else:
            for legalMove in game.get_legal_moves(game.active_player):
                newV, newMov = self.alphabetamin(game.forecast_move(legalMove), depth = depth, alpha=alpha, beta=beta, currentDepth = currentDepth + 1)
                if newV >= beta:
                    return newV, legalMove
                if newV > alpha:
                    alpha = newV
                    bestMove = legalMove
            return alpha, bestMove
        
