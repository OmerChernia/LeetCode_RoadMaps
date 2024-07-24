class Solution(object):
    def isSubsequence(self, s, t):
        if not s:
            return True

        i = 0
        for j in range(len(t)):
            if t[j] == s[i]:
                i += 1
                if i == (len(s)):
                    return True

        return False
