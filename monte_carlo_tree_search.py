import math
import numpy as np


def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        # better states are visited more often (but couldn't they just be easier to reach? What about a very unique and hard to reach very good state?)
        # why not track the fraction of the time it was chosen when available?
        # I think it doesn't matter because you get the value from the value function, not just the visit count

        self.to_play = to_play  # whos turn, 1 or -1
        self.prior = prior  # prior probability of selecting this from parent
        self.value_sum = 0  # total value of this and children from all visits
        self.children = {}  # all legal child positions
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for action, prob in enumerate(action_probs):
            if prob != 0:
                self.children[action] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(
            self.state.__str__(), prior, self.visit_count, self.value()
        )


def backpropagate(search_path, value, to_play):
    """
    At the end of a simulation, we propagate the evaluation all the way up the tree
    to the root.
    """
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1


class MCTS:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def run(self, model, state, to_play):

        root = Node(0, to_play)

        # EXPAND root
        action_probs, value = model.predict(state)
        action_probs = self.game.mask_invalid_moves(state, action_probs)
        root.expand(state, to_play, action_probs)

        for _simulation in range(self.args["num_simulations"]):
            node = root
            search_path = [node]

            # SELECT
            action = None
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
            assert action is not None

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node, and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_board_from_perspective(next_state, player=-1)

            # The value of the new state from the perspective of the other player # I think this means to say 'of the current player'
            value = self.game.get_reward_for_player(next_state, player=1)

            # I'm confused which player is 'current' and 'other'. Comments imply its both 1 and -1.

            if value is None:
                # If the game has not ended:
                action_probs, value = model.predict(next_state)
                action_probs = self.game.mask_invalid_moves(
                    state=next_state, action_probs=action_probs
                )
                node.expand(next_state, parent.to_play * -1, action_probs)

            backpropagate(search_path, value, parent.to_play * -1)

        return root
