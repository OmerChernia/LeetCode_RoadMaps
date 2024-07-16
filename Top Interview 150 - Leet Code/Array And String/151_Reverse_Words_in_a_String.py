class Solution(object):
    def reverseWords(self, s):
        sep_s = s.split()
        sep_s.reverse()
        rev_s = ' '.join(sep_s)
        return rev_s

