class Solution(object):
    def isPalindrome(self, s):
        n = ''.join(char for char in s if char.isalnum()).lower()
        l = 0
        r = len(n)-1
        while(l<r):
            if n[l] != n[r]:
                return False
            l += 1
            r -= 1

        return True