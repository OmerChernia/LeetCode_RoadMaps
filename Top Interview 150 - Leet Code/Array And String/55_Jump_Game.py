class Solution(object):
    def canJump(self, nums):
        if not nums:
            return False

        max_index = 0

        for i in range(len(nums)):
            if i > max_index:
                return False
            if nums[i] + i > max_index:
                max_index = nums[i] + i

        if max_index >= len(nums) - 1:
            return True
        else:
            return False

