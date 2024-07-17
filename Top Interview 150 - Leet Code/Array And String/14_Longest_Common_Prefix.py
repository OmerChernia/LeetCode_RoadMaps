class Solution(object):
    def longestCommonPrefix(self, strs):
        longest = ""
        flag = False
        let = strs[0][0]
        let_l = 1

        while flag == False:
            for i in range(len(strs)):
                if strs[i][let_l] != let:
                    if let_l == 1:
                        return ""
                    flag = True
                elif i == (len(strs) - 1):
                    let += strs[i][let_l]
                    let_l += 1

        return let


solution = Solution()

strs1 = ["flower", "flow", "flight"]
print(self.longestCommonPrefix(strs1))  # Output: "fl"
