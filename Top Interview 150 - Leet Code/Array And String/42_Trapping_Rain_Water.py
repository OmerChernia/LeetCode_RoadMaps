class Solution(object):
    def trap(self, height):
        def minus_one_to_arr(arr):
            for i in range(len(arr)):
                if arr[i] != 0:
                    arr[i] -= 1
            return arr

        sol = 0
        temp = -1

        for j in range(max(height)):
            for i in range(len(height)):
                if height[i] != 0:
                    if temp != -1 and (i-temp) != 1:
                        sol += i - (temp+1)
                    temp = i
            minus_one_to_arr(height)
            temp = -1

        return sol


solution = Solution()
print(solution.trap([4,2,0,3,2,5]))