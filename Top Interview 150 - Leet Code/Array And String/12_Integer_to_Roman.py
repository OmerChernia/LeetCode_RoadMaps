class Solution(object):
    def intToRoman(self, num):
        roman = [
                    (1000, 'M'),
                    (900, 'CM'),
                    (500, 'D'),
                    (400, 'CD'),
                    (100, 'C'),
                    (90, 'XC'),
                    (50, 'L'),
                    (40, 'XL'),
                    (10, 'X'),
                    (9, 'IX'),
                    (5, 'V'),
                    (4, 'IV'),
                    (1, 'I')
                ]

        l = len(str(num))
        sep_arr = []
        mod = 10
        sol = []

        for i in range(l):
            s = sum(sep_arr)
            sep_arr.append((num % mod) - s)
            mod *= 10

        sep_arr.reverse()

        for val in sep_arr:
            while val > 0:
                for r in roman:
                    if r[0] <= val:
                        sol.append(r[1])
                        val -= r[0]
                        break

        return ''.join(sol)


solution = Solution()
print(solution.intToRoman(3749))