import sys

class PartialParse(object):
    def __init__(self, sentence):
        """
        @param sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """
        self.sentence = sentence

        self.stack = ["ROOT"]
        self.buffer = sentence
        self.dependencies = list()
        ### END YOUR CODE


    def parse_step(self, transition):
        """
        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions.
        """
        print("Stack before parse step: ", self.stack)
        print("Buffer before parse step: ", self.buffer)

        print("Transition: ", transition)

        if transition == "S":
            self.stack.append(self.buffer[0])
            self.buffer = self.buffer[1:]

        elif transition == "LA":
            dependent = self.stack.pop(-2)
            self.dependencies.append((self.stack[-1], dependent))

        elif transition == "RA":
            dependent = self.stack.pop()
            self.dependencies.append((self.stack[-1], dependent))

        print("Stack after parse step: ", self.stack)
        print("Buffer after parse step: ", self.buffer)
            
        ### END YOUR CODE

    def parse(self, transitions):
        """

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. 
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @param sentences (list of list of str): A list of sentences to be parsed
    @param model (ParserModel): The model that makes parsing decisions. 
    @param batch_size (int): The number of PartialParses to include in each minibatch


    @return dependencies (list of dependency lists): A list where each element is the dependencies list for a parsed sentence.
    """
    dependencies = []

    print("Sentences: ", sentences)

    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:]

    while (len(unfinished_parses) > 0):

        minibatch = unfinished_parses[:batch_size]
        transitions = model.predict(minibatch)
        [partial_parse.parse_step(transition) for partial_parse, transition in zip(minibatch, transitions)]

        for i, partial_parse in enumerate(unfinished_parses):
            if len(partial_parse.buffer) == 0 and len(partial_parse.stack) == 1:
                unfinished_parses.pop(i)

    [dependencies.append(partial_parse.dependencies) for partial_parse in partial_parses]

    print("Dependencies: ", dependencies)

    return dependencies
