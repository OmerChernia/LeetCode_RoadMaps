class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        min_len = float('inf')
        j = 0
        temp_sum = 0
        temp_cnt = 2

        for i in range(len(nums)):
            temp_sum += nums[i]

            while (temp_sum >= target):
                min_len = min(min_len, i - j + 1)
                temp_sum -= nums[j]
                j += 1

        return min_len if min_len != float('inf') else 0

solution = Solution()
print(solution.minSubArrayLen(4, [1,4,4]))

