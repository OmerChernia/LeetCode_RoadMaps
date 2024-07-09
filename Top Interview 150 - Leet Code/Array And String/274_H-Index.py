class Solution(object):
    def hIndex(self, citations):
        sorted_cit = sorted(citations, reverse=True)
        left_to_end = len(sorted_cit)

        for i in range(left_to_end):
            if sorted_cit[i] < i+1:
                return i

        return left_to_end