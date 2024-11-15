class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        sol = {}
        for item in strs:
            key = "".join(sorted(item))

            if key not in sol:
                sol[key] = []

            sol[key].append(item)

        return list(sol.values())

solution = Solution()
print(solution.groupAnagrams(["eat","tea","tan","ate","nat","bat"]))
