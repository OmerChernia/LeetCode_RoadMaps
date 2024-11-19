class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """

        sol = {}

        for i, num in enumerate(nums):
            if num in sol and i- sol[num] <= k:
                return True
            else:
                sol[num] = i

        return False
