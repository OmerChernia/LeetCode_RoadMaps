class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        sol = {}

        for i, num in enumerate(nums):
            complement = target - num
            if complement in sol:
                return [sol[complement], i]
            sol[num] = i

        return []


solution = Solution()
print(solution.twoSum([3,3],6))