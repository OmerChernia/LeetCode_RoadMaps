class Solution(object):
    def rotate(self, nums, k):
        l = len(nums)
        start_point = l - (k % l)

        first_arr = nums[start_point:]
        second_arr = nums[:start_point]

        nums[:] = first_arr + second_arr