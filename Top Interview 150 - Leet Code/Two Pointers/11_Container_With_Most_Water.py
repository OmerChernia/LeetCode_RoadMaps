class Solution(object):
    def maxArea(self, height):
        max = 0

        l = 0
        r = len(height) -1

        while l < r:
            if (min(height[l] , height[r]) * (r -l)) > max:
                max = (min(height[l] , height[r]) * (r -l))

            if height[l] < height[r]:
                l += 1
            else:
                r-= 1

        return max
