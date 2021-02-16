import random
from random import shuffle

import sys
sys.path.append('../pyzx')
import pyzx as zx

class Mutant:
    def __init__(self, c):
        self.c_orig = c
        self.c_curr = c
        self.g_curr = c.to_graph()
        self.score = None
        self.dead = False # no more actions can be applied to it


# FIXME: Should really subclass Optimizer (-> BaseOptimizer). Need better file structure here
# FIXME: Should Mutant own score function too? Probably not. And therefore not its own score value?
class GeneticOptimizer:
    def __init__(self, c, actions, score=lambda m: m.c_curr.twoqubitcount(), n_generations=100, n_mutants=100):
        self.c_orig = c
        # FIXME: Maybe an action takes in both a circuit and graph and returns a new circuit and graph? And it also has to return whether or not it succeeded. This way, we don't have to worry abou tthings like the teleport_reduce action having to explicitly say we don't need to extract -- it will be the actions responsibility to do this. Similar for qiskit passes, tekt passes
        # Note also how in this method its the actions method to determine if the action didn't do anything. For the pyzx ones, this is taken care of as we have matches. Howver, say we call some qiskit things and it executes but it doesn't actually change the circuit. In this case, would be the actions respnsibility to check for this.
        self.actions = actions
        self.n_gens = n_generations
        self.n_mutants = n_mutants
        self.mutants = [Mutant(c) for _ in range(n_mutants)]
        self.score = score # function that maps Circuit -> Double


    def mutate(self):
        for m in self.mutants:
            # FIXME: In this model, actions have to look for their own "matches". Will return false if it couldn't find any
            success = False
            for a in shuffle(self.actions):
                success, (c_new, g_new) = a(m.c_curr, m.g_curr)
                if success:
                    break
            if not success:
                m.dead = True


    def update_scores(self):
        for m in self.mutants:
            m.score = self.score(m) # FIXME: Bad idiom

    def select(self, method="tournament"):
        if method == "tournament":
            new_mutants = list()
            for _ in range(self.n_mutants):
                m1, m2 = random.sample(self.mutants, 2)
                if m1.dead:
                    new_mutants.append(m2)
                elif m1.score < m2.score: # Reminder: we are playing golf
                    new_mutants.append(m1)
                else:
                    new_mutants.append(m2)
            self.mutants = new_mutants
        else:
            raise RuntimeError(f"[select] Unknown selection method {method}")

    # FIXME: should be _optimize
    # Playing golf -- want the LOWEST score!
    def evolve(self, c):
        self.update_scores()
        best_mutant = min(self.mutants, key=lambda m: m.score) # FIXME: Check if this assignment is by reference or value
        best_score = best_mutant.score

        for _ in range(self.n_gens):
            self.mutate()
            self.update_scores()
            best_in_gen = min(self.mutants, key=lambda m: m.score)
            if best_in_gen.score < best_score:
                best_mutant = best_in_gen
                best_score = best_in_gen.score

            if all([m.dead for m in self.mutants]):
                print("[evolve] stopping early -- all mutants are dead")
                break

            self.select()
