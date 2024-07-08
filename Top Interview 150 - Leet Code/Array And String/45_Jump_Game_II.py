class Solution(object):
    def jump(self, nums):
        farthest = 0
        jumps = 0
        curr_end = 0

        for i in range(len(nums) -1):
            if i + nums[i] > farthest:
                farthest = i + nums[i]

            if i == curr_end:
                jumps += 1
                curr_end = farthest

        return jumps
