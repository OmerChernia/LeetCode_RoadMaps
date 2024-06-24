class Solution(object):
    def removeElement(self, nums, val):
        l = 0
        r = len(nums) - 1

        if not nums:
            return 0

        while r >= 0 and nums[r] == val:
            r -= 1

        while l <= r:
            if nums[l] == val:
                nums[l] = nums[r]
                r -= 1
                while r >= 0 and nums[r] == val:
                    r -= 1
            l += 1

        return r + 1

