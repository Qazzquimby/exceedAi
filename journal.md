# Exceed AI with alphazero

I've written AI for board games in the past, but usually something checkersy with full information
and a simple board.
Exceed has ten billion characters with unique decks of hidden information and sometimes gimmick
mechanics.
So that sounded fairly impossible. One step in machine learning is serializing the input into a
vector, a fixed length list of numbers. That's fine for go (just rows and columns containing a
number for emtpy,white,black),
but here that vector needs to contain the locations of all your cards, what you know about your
opponent's hand and discard, any passive effects in play, the state of any character specific stuff,
and probably more I'm not thinking of.
I spent some time looking for a way to use json as input instead (objects with attributes) rather
than a vector, but that's not a thing.

I play dominion on my phone most days, and the AI is amazing despite dominion having many unique
cards, requiring a lot of planning ahead, hidden information and other wrinkles.
Incredible, how did they do it? Turns out they explained it and I needed only
google https://www.templegatesgames.com/dominion-ai/

I think dominion is a significantly harder problem than exceed, and they solved it with a model
small enough to run on a phone.
Neat, so I won't give up.

I need to implement alphazero and montecarlo tree search, the two core techniques in the dominion
AI. Fortunately those have been done many times before and I can study and build off of their code.

Serializing the input sounds much more involved.
I'm also not sure about the action vector. For the AI to output an action, it gives a list of
numbers where each index of the list corresponds to a possible action, and a larger number means
that action is better. I'm not sure how to put every possible choice into a fixed size list.
For example, if a card says "The opponent may discard a card. If they don't, this has +2 power."
then there's a new state for them to pick a card or nothing.
The first index is don't-discard, some indices correspond to cards they might discard and all other
action indices are invalid? Some part of the gamestate array is 1 to tell them they're currently
making that particular choice, and is 0 all the rest of the time?
Having a huge vector where only a small amount is used makes learning a lot slower, because the it
still fine tunes the values of all the irrelevant numbers.

If it is currently searching its deck for a card:

- how does it know that's what the current action is? A flag for every possible situation sounds
  undoable.
- is the action vector just the options in order? Are their indices designated for each card or
  space?

"You may discard a card. If you don't, take 2 damage."
If I was manually programming the choice I'd get the current lowest value card in hand (given the
game state) and compare the value of losing that card to taking 2 damage. Could check the value of
the board without the card and with the damage taken and take whichever is better

It looks like there are very few projects like this, at least open source. Either I suck at googling
or few people know what I want to learn.
Temple Gate games, possibly the facebook diplomacy ai (though the code is dense and I'm not familiar
with the game), ... anyone else?
Maybe mtg online has good AI. The open source magic engines have fairly simple AI.

More research indicates the AI was written by Keldon. I went through every message from him on the
discord to note anything he said about the ai design.
A few bits were puzzling, like selfplay and training are done separately, when the blog post implied
training was done immediately after every action.

Keldon answered some questions and shared a picture of the overall architecture. My main
misunderstanding was thinking that the output vector was targeted to the current choice. The output
vector is the model's global preferences for many things. How much they want to gain a platinum, how
much they want to trash a platinum, or how much they want to trash a platinum for benefit - all
whether there are platinums in the current game.

Nope thats wrong. The output is a single vector for each major type of action (trash, buy, discard,
play).
Early in the tree, each card gets an embedding. Each copy of each card in the game gets an encoded
card containing its embedding, location, etc.
This goes through a transformer encoder (which I'm still learning) to somehow combine it with all
other context while keeping corresponding encoded versions of each input object (card/player)

The "active card" (not sure I understand) and an averaging of manually masked "selected cards" move
on
and the output is a single float for each action (buy, trash, etc) on those.

I now have a connect 4 model that beats me easily. I'm going to refactor it, and then move onto
blackjack which will force me to handle stochasticity.

How does the encoder have corresponding outputs but also have each part influence each other?:
It's the encoder part of a transformer.

In exceed I can handle symmetrical states easily I think. Other than markers, I could just hold '
distance from opponent' and 'distance from wall behind opponent' which I believe holds all
positioning information.

I should use validation data to prevent overfitting. 10% of training data as validation?
Using best loss is questionable for saving checkpoints because win rate can improve while loss gets
worse. Should simulate games periodically to determine best checkpoint.

MCTS tree can be reused if a state is already in the previous tree. Use a hash table.

Games that are too old are probably bad for training as their results are from a dumber version of
the AI. Effect may decrease over time. Might be worth keeping the data anyway.

Number of MCTS iterations is variable. High maximum but early termination if one choice is clearly better.

