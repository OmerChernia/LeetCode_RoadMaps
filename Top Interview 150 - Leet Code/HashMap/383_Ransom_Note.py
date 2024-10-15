class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        m = {}

        for item in magazine:
            if item in m:
                m[item] += 1
            else:
                m[item] = 1

        for let in ransomNote:
            if m[let] in m and m[let] > 0:
                m[let] -= 1
            else:
                return False

        return True