import random

class RandomizedSet(object):

    def __init__(self):
        self.val_index = {}
        self.values = []

    def insert(self, val):
        if val in self.val_index:
            return False
        self.val_index[val] = len(self.values)
        self.values.append(val)
        return True

    def remove(self, val):
        if val not in self.val_index:
            return False

        # switch val and last value in the values array to get O(1)
        last_val = self.values[-1]
        index = self.val_index[val]

        self.values[index] = last_val
        self.val_index[last_val] = index

        self.values.pop()
        del self.val_index[val]
        return True

    def getRandom(self):
        return random.choice(self.values)
