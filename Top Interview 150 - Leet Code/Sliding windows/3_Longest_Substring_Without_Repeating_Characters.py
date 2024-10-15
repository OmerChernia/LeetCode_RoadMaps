class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        ans = 0
        temp_sub = ""
        i,j = 0,0

        if len(s) == 0:
            return 0

        if len(s) == 1:
            return 1

        for i in range(len(s)):
            temp_sub += s[i]
            j = i+1

            while((j < len(s))):
                if(not(s[j] in temp_sub)):
                    temp_sub += s[j]
                    j += 1
                else:
                    sub_len = len(temp_sub)
                    ans = max(ans, sub_len)
                    break

            ans = max(ans, len(temp_sub))
            temp_sub = ""

        return ans

