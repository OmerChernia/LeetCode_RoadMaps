class Solution(object):
    def strStr(self, haystack, needle):
        index = -1
        if needle in haystack:
            index = haystack.index(needle)

        return index
