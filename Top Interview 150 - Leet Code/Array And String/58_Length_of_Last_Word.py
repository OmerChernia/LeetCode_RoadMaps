class Solution(object):
    def lengthOfLastWord(self, s):
        n = s.split()

        return len(n[-1])
