class Solution(object):
    def threeSum(self, nums):

        if len(nums) < 3:
            return []

        sol = []
        nums.sort()

        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            l, r = i + 1, len(nums) - 1

            while l < r:
                sum = nums[l] + nums[i] + nums[r]

                if sum == 0:
                    sol.append([nums[l],nums[i],nums[r]])
                    l += 1
                    r -= 1

                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1

                elif sum < 0:
                    l += 1
                elif sum > 0:
                    r -= 1

        return sol

solution = Solution()

print(solution.threeSum([0,0,0]))