class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """

        sol = {}

        while(n != 1):
            if n in sol:
                return False
            else:
                sol[n] = sum([int(i) ** 2 for i in str(n)])
                n = sol[n]

        return True


