# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.

        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print "newPos, ", newPos
        #print "newFood, ", newFood.asList()
        #print "newGhostStates, ", newGhostStates
        #print "newScaredTimes"
        #print "successor game state score, ", successorGameState.getScore()

        closestFoodDistance = 0
        distanceFoodList = [util.manhattanDistance(newPos, food) for food in currentGameState.getFood().asList()]
        if distanceFoodList:
            closestFoodDistance = min(distanceFoodList)

        scaredGhosts = []
        unScaredGhosts = []

        for ghost in currentGameState.getGhostStates():
            if ghost.scaredTimer == 0:
                unScaredGhosts.append(ghost)
            else:
                scaredGhosts.append(ghost)

        closestScaredGhostDistance = 0
        distanceScaredGhostList = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in scaredGhosts]
        if distanceScaredGhostList:
            closestScaredGhostDistance = min(distanceScaredGhostList)

        closestUnScaredGhostDistance = -500
        distanceUnScaredGhostList = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in unScaredGhosts]
        if distanceUnScaredGhostList:
            closestUnScaredGhostDistance = min(distanceUnScaredGhostList)

        if closestFoodDistance == 0:
            closestFoodDistance = 1000
        else:
            closestFoodDistance = (1. / closestFoodDistance) * 10

        if closestUnScaredGhostDistance == 0:
            return -1000
        else:
            closestUnScaredGhostDistance = (1. / closestUnScaredGhostDistance) * -10

        evalScore = closestFoodDistance + closestScaredGhostDistance + closestUnScaredGhostDistance

        return evalScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        maxNodeVal = float("-inf")
        takeAction = None

        for action in gameState.getLegalActions(0):
            val = self.minValue(gameState.generateSuccessor(0, action), 1, 1)
            if val > maxNodeVal:
                maxNodeVal = val
                takeAction = action

        return takeAction

    def maxValue(self, state, depth):

        if state.isWin() or state.isLose() or depth > self.depth:
            return self.evaluationFunction(state)

        maxNodeVal = float("-inf")
        for action in state.getLegalActions(0):
            maxNodeVal = max(maxNodeVal, self.minValue(state.generateSuccessor(0, action), depth, 1))

        return maxNodeVal

    def minValue(self, state, depth, agentIdx):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIdx < state.getNumAgents() - 1:
            minNodeVal = float("inf")
            for action in state.getLegalActions(agentIdx):
                minNodeVal = min(minNodeVal, self.minValue(state.generateSuccessor(agentIdx, action), depth, agentIdx + 1))
            return minNodeVal

        minNodeVal = float("inf")
        for action in state.getLegalActions(agentIdx):
            minNodeVal = min(minNodeVal, self.maxValue(state.generateSuccessor(agentIdx, action), depth + 1))

        return minNodeVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        maxNodeVal = float("-inf")
        beta = float("inf")
        takeAction = None

        for action in gameState.getLegalActions(0):
            val = self.minValue(gameState.generateSuccessor(0, action), 1, 1, maxNodeVal, beta)
            if val > maxNodeVal:
                maxNodeVal = val
                takeAction = action

        return takeAction


    def maxValue(self, state, depth, alpha, beta):

        if state.isWin() or state.isLose() or depth > self.depth:
            return self.evaluationFunction(state)

        maxNodeVal = float("-inf")
        for action in state.getLegalActions(0):
            maxNodeVal = max(maxNodeVal, self.minValue(state.generateSuccessor(0, action), depth, 1, alpha, beta))
            if maxNodeVal > beta:
                return maxNodeVal
            alpha = max(alpha, maxNodeVal)

        return maxNodeVal

    def minValue(self, state, depth, agentIdx, alpha, beta):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        minNodeVal = float("inf")
        for action in state.getLegalActions(agentIdx):
            if agentIdx < state.getNumAgents() - 1:
                minNodeVal = min(minNodeVal, self.minValue(state.generateSuccessor(agentIdx, action), depth, agentIdx + 1, alpha, beta))
            else:
                minNodeVal = min(minNodeVal, self.maxValue(state.generateSuccessor(agentIdx, action), depth + 1, alpha, beta))

            if minNodeVal < alpha:
                return minNodeVal
            beta = min(beta, minNodeVal)

        return minNodeVal

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        maxNodeVal = float("-inf")
        takeAction = None

        for action in gameState.getLegalActions(0):
            val = self.chanceValue(gameState.generateSuccessor(0, action), 1, 1)
            if val > maxNodeVal:
                maxNodeVal = val
                takeAction = action

        return takeAction

    def maxValue(self, state, depth):

        if state.isWin() or state.isLose() or depth > self.depth:
            return self.evaluationFunction(state)

        maxNodeVal = float("-inf")
        for action in state.getLegalActions(0):
            maxNodeVal = max(maxNodeVal, self.chanceValue(state.generateSuccessor(0, action), depth, 1))

        return maxNodeVal

    def chanceValue(self, state, depth, agentIdx):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIdx < state.getNumAgents() - 1:
            sum = 0
            for action in state.getLegalActions(agentIdx):
                chanceValue = self.chanceValue(state.generateSuccessor(agentIdx, action), depth, agentIdx + 1)
                sum += chanceValue
            return sum / float(len(state.getLegalActions(agentIdx)))

        sum = 0
        for action in state.getLegalActions(agentIdx):
            chanceValue = self.maxValue(state.generateSuccessor(agentIdx, action), depth + 1)
            sum += chanceValue

        return sum / float(len(state.getLegalActions(agentIdx)))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pacmanPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    closestFoodDistance = 0
    distanceFoodList = [util.manhattanDistance(pacmanPos, food) for food in currentGameState.getFood().asList()]
    if distanceFoodList:
        closestFoodDistance = min(distanceFoodList)

    scaredGhosts = []
    unScaredGhosts = []

    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer == 0:
            unScaredGhosts.append(ghost)
        else:
            scaredGhosts.append(ghost)

    closestScaredGhostDistance = 0
    if scaredGhosts:
        scaredGhostDistanceList = [util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in scaredGhosts]
        closestScaredGhostDistance = min(scaredGhostDistanceList)

    closestUnScaredGhostDistance = 500
    if unScaredGhosts:
        unScaredGhostDistanceList = [util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in unScaredGhosts]
        closestUnScaredGhostDistance = min(unScaredGhostDistanceList)

    remainingCapsules = len(currentGameState.getCapsules())
    remainingFood = len(currentGameState.getFood().asList())

    if closestFoodDistance == 0:
        closestFoodDistance = 500
    else:
        closestFoodDistance = 2 * 1. / closestFoodDistance

    if remainingFood == 0:
        remainingFood = 500

    if closestUnScaredGhostDistance == 0:
        closestUnScaredGhostDistance = -500
    else:
        closestUnScaredGhostDistance = -2.5 * 1. / closestUnScaredGhostDistance

    betterEvalScore = score + closestFoodDistance + remainingCapsules + remainingFood + closestScaredGhostDistance + closestUnScaredGhostDistance

    return betterEvalScore

# Abbreviation
better = betterEvaluationFunction

