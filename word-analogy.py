#!/usr/bin/env python3

import os, sys, math, functools, numpy

# similarity measures
SIMILARITY_EUCLIDEAN = 0
SIMILARITY_MANHATTAN = 1
SIMILARITY_COSINE    = 2

# reads files from a folder line by line
# iterating returns (line, filename)
class DirectoryCrawler:
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        for f in os.listdir(self.path):
            if f.startswith("."):
                continue
            filepath = self.path + '/' + f
            if os.path.isfile(filepath):
                with open(filepath, 'r') as handle:
                    for line in handle.readlines():
                        yield (line.strip(), filepath)

# similarity measures
def similarity_euclidean(v1, v2):
    return numpy.sqrt(numpy.sum((v1 - v2)**2))

def similarity_manhattan(v1, v2):
    return numpy.sum(numpy.abs(v2 - v1))

def similarity_cosine(v1, v2):
    return v1.dot(v2)

# generalized argmax for a function taking one argument applied over a list
def argmax(vectors, function, argmin = False):
    highest_vector = None
    is_first_run = True
    highest_value = None
    highest_word = ""
    for word, vector in vectors.items():
        result = function(vector) 
        # handle first case without assigning before
        if is_first_run:
            highest_value = result
            highest_vector = vector
            highest_word = word
            is_first_run = False
        # argmin/argmax test
        test = (result > highest_value 
                if not argmin 
                else result < highest_value)
        if test:
            highest_value = result
            highest_vector = vector
            highest_word = word
    return (highest_word, highest_vector, highest_value)

# c + b - a
target_vector = lambda a, b, c: c + b - a

class WordSolver:
    def __init__(self, vf, ind, outd, ef, normal, sim):
        self.vector_file = vf
        self.input_dir = ind
        self.output_dir = outd
        self.eval_file = ef
        self.normalize = int(normal)
        self.similarity_type = int(sim) 

        self.word_vectors = {}
        self.problems = []
        self.solutions = []
        self.results = {}

    # problem should be in the form of [w1, w2, w3]; measure should be 0/1/2
    def solve_analogy(self, problem, measure):
        words = problem
        (v1, v2, v3) = [self.word_vectors[word] for word in words]

        target = target_vector(v1, v2, v3)
        similarity = None
        argmin = None
        # use given similarity measure by assigning to similarity
        if measure == SIMILARITY_EUCLIDEAN:
            similarity = similarity_euclidean
            argmin = True
        elif measure == SIMILARITY_MANHATTAN:
            similarity = similarity_manhattan
            argmin = True
        elif measure == SIMILARITY_COSINE:
            similarity = similarity_cosine
            argmin = False

        # partially apply similarity to use with argmax
        partial = lambda vector: similarity(vector, target)
        (word, vector, value) = argmax(self.word_vectors, partial, argmin)
        return word

    def get_data(self):
        vfile = open(self.vector_file, 'r')

        for word_vector in vfile.readlines():
            split_vector = word_vector.strip().split(" ")
            key, *word_vector = split_vector
            if self.normalize == 0:
                self.word_vectors[key] = numpy.array(word_vector, dtype=float)
            else:
                vec = numpy.array(word_vector, dtype=float)
                vec = vec * (1 / (vec.dot(vec)))
                self.word_vectors[key] = vec

        analogies = DirectoryCrawler(self.input_dir)
        self.problems = [ (analogy.split(' ')[:-1], filepath, analogy.split(' ')[-1])
                          for (analogy, filepath) in analogies ]

        vfile.close()

    def run(self):
        self.get_data()
        print(f"using {self.normalize} norm and {self.similarity_type} sim")
        file_handles = {}
        def record(success, filename):
            t_add = lambda x, y: (x[0] + y[0], x[1] + y[1])
            if filename not in self.results:
                print(f"starting file {filename}")
                print("")
                self.results[filename] = (0, 0)
            else:
                mod = (1, 1) if success else (0, 1)
                self.results[filename] = t_add(self.results[filename], mod)

        efile = open(self.eval_file, 'w')
        for (problem, filepath, proper_solution) in self.problems:
            filename = os.path.basename(filepath)
            output_path = self.output_dir + '/' + filename
            if output_path not in file_handles:
                file_handles[output_path] = open(output_path, 'w')
            try:
                solution = self.solve_analogy(
                        problem, 
                        self.similarity_type
                    )
                if solution == proper_solution:
                    record(True, filename)
                else:
                    record(False, filename)

                file_handles[output_path].write(f'{problem[0]} {problem[1]} {problem[2]} {solution}\n')
            except KeyError as err:
                record(False, filename)
                file_handles[output_path].write(f'{problem[0]} {problem[1]} {problem[2]} NOSOLUTION\n')
            print(u'\u001b[2A')
            print(f"successes, attempts: {self.results[filename]}")
        for name, handle in file_handles.items():
            handle.close()

        # write eval file
        totals = (0, 0)
        for filename, (successes, attempts) in self.results.items():
            totals = (totals[0] + successes, totals[1] + attempts)
            efile.write(f'{filename}\nACCURACY: {(successes / attempts)*100}% ({successes}/{attempts})\n')
        efile.write(f'TOTAL ACCURACY: {(successes / attempts)*100}% ({successes}/{attempts})\n')

        efile.close()


# handle command line arguments
def create_word_solver(args):
    v = args[1]
    i = args[2]
    o = args[3]
    e = args[4]
    n = args[5]
    s = args[6]
    return WordSolver(v, i, o, e, n, s)

if __name__ == "__main__":
    ws = create_word_solver(sys.argv)
    ws.run()
    print("Output has been written.")
