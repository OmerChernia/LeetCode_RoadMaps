class Solution(object):
    def majorityElement(self, nums):
        num = nums[0]
        amount = 1

        for i in range(1, len(nums)):
            if nums[i] == num:
                amount += 1
            else:
                amount -= 1
            if amount == 0:
                num = nums[i]
                amount = 1

        return num
