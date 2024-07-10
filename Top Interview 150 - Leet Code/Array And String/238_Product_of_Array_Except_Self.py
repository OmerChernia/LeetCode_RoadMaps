class Solution(object):
    def productExceptSelf(self, nums):
        right_arr = []
        r_prod = 1
        left_arr = []
        l_prod = 1
        sol_arr = []

        for i in nums:
            r_prod *= i
            right_arr.append(r_prod)

        for j in reversed(nums):
            l_prod *= j
            left_arr.append(l_prod)

        left_arr.reverse()

        for k in range(len(nums)):
            if k == 0:
                sol_arr.append(left_arr[1])
            elif k == (len(nums)-1):
                sol_arr.append(right_arr[len(nums)-2])
            else:
                sol_arr.append(right_arr[k-1] * left_arr[k+1])

        return sol_arr
