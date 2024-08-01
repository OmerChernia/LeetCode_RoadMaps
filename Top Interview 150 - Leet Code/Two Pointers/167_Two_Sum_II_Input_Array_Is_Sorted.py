class Solution(object):
    def twoSum(self, numbers, target):
        l = 0
        r = len(numbers) - 1

        while l < r:
            if numbers[l] + numbers[r] == target:
                return [l+1, r+1]
            else:
                if numbers[l] + numbers[r] < target:
                    l += 1
                else:
                    r -= 1

        return [0, 0]


solution = Solution()
print(solution.twoSum([-3,3,4,90], 0))