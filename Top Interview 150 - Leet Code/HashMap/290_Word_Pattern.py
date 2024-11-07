class Solution(object):
    def wordPattern(self, pattern, s):
        check_s = s.split()
        if len(check_s) != len(pattern):
            return False

        check_map = {}

        for i in range(len(pattern)):
            if pattern[i] not in check_map:
                check_map[pattern[i]] = check_s[i]
            else:
                if check_map[pattern[i]] != check_s[i]:
                    return False

        flipped_dict = {value: key for key, value in check_map.items()}

        result = []
        for item in check_s:
            result.append(flipped_dict[item])

        result = ''.join(result)

        if result != pattern:
            return False

        return True
