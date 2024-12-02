class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        check = {}
        temp_s = ""

        if len(s) != len(t):
            return False

        for i in range(len(s)):
            if s[i] not in check:
                # Initialize an empty list for the key if it doesn't exist
                check[s[i]] = []
            check[s[i]].append(t[i])  # Append the character from t to the list

        # Rebuild the transformed string using the first mapped value for each character in s
        for j in s:
            j = check[j].pop(0)  # Use and remove the first value from the list
            temp_s += j

        # Check if the rebuilt string matches t
        if temp_s == t:
            return True

        return False


solution = Solution()
print(solution.isIsomorphic("foo", "bar"))  # Expected output: True
