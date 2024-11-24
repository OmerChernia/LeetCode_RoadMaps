class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if len(nums) == 0:
            return 0

        nums = set(nums)
        ans = 0

        for num in nums:
            if num - 1 not in nums:
                curr_num = num
                curr_streak = 1

                while(curr_num + 1 in nums):
                    curr_num += 1
                    curr_streak += 1

                ans = max(ans, curr_streak)

        return ans
