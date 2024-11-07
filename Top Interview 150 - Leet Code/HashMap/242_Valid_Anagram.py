class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s_map = {}
        t_map = {}

        for i in s:
            if not i in s_map:
                s_map[i] = 1
            else:
                s_map[i] += 1

        for j in t:
            if not j in t_map:
                t_map[j] = 1
            else:
                t_map[j] += 1

        if s_map == t_map:
            return True

        return False


solution = Solution()
solution.isAnagram("anagram", "nanagram")

