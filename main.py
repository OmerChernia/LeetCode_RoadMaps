from collections import defaultdict

class MinStack(object):
    def __init__(self):
        self.stck = []
        self.min_stck = []

    def push(self, val):
        self.stck.append(val)
        val = min(val, self.min_stck[-1] if self.min_stck else val)
        self.min_stck.append(val)

    def pop(self):
        self.stck.pop()
        self.min_stck.pop()

    def top(self):
        return self.stck[-1]

    def getMin(self):
        return self.min_stck[-1]

class Solution(object):

    # --------------------------------------------------- #Contains Duplicate

    def containsDuplicate(self, nums):
        seen_set = set()

        for num in nums:
            if num in seen_set:
                return True
            seen_set.add(num)

        return False

    #--------------------------------------------------- #Is Anagram

    #Solution Number 1 - using sorting:
    def isAnagram(self, s, t):
        if len(s) != len(t):
            return False
        else:
            sortedS = ''.join(sorted(s))
            sortedT = ''.join(sorted(t))

        if sortedS == sortedT:
            return True
        else:
            return False

    # Solution Number 2 - using Hashmaps:
    def isAnagram2(self, s, t):
        if len(s) != len(t):
            return False

        s_map = {}
        t_map = {}

        for i in range(len(s)):
            s_map[s[i]] = 1+s_map.get(s[i],0) #python way to avoid keyError
            t_map[t[i]] = 1+t_map.get(t[i],0)

        for c in s_map:
            if s_map[c] != t_map.get(c,0):
                return False

        return True

    # --------------------------------------------------- #Two Sum

    #Solution Number 1 - O(n^2)
    def twoSum(self, nums, target):
        ret = []
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    ret.append(i)
                    ret.append(j)
                    return ret

        return ret

    # Solution Number 2 - O(n), using HashMap
    def twoSum(self, nums, target):
        ret_map = {} # index : value

        for i,n in enumerate(nums):
            diff = target - n
            if diff in ret_map:
                return [ret_map[diff] ,i]
            ret_map[n] = i

        return

    # --------------------------------------------------- #Group Anagram

    def groupAnagrams(self, strs): #using default dict
        ret_map = defaultdict(list)
        result = []

        for s in strs:
            sorted_s = ''.join(sorted(s))
            ret_map[sorted_s].append(s)

        for value in ret_map.values():
            result.append(value)

        return result

    # --------------------------------------------------- #Top K Frequent Elements

    # Solution Number 1 - bad runtime, good memory
    def topKFrequent(self, nums, k):
        ret_map = defaultdict(int)
        ans = []

        for i in nums:
            ret_map[i] += 1

        for i in range(k):
            max_val = max(ret_map, key=ret_map.get)
            ans.append(max_val)
            del ret_map[max_val]

        return ans

    # Solution Number 2
    def topKFrequent(self, nums, k):
        count = {}
        freq = [[] for i in range(len(nums) + 1)]

        for n in nums:
            count[n] = 1 + count.get(n, 0)

        for n, c in count.items():
            freq[c].append(n)

        res = []
        for i in range(len(freq) - 1, 0, -1):
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res

    # --------------------------------------------------- #Product except self

    def productExceptSelf(self, nums):
        n = len(nums)

        # Initialize output array with all elements set to 1
        result = [1] * n

        # Compute left products
        left_product = 1
        for i in range(n):
            result[i] *= left_product
            left_product *= nums[i]

        # Compute right products and multiply with the corresponding left product
        right_product = 1
        for i in range(n - 1, -1, -1):
            result[i] *= right_product
            right_product *= nums[i]

        return result

    # --------------------------------------------------- #Valid Sudoku

    def isValidSudoku(self,board):
        def is_valid(arr):
            bucket_checks = [0] * 10
            for num in arr:
                if num != ".":
                    num = int(num)
                    bucket_checks[num] += 1
                    if bucket_checks[num] > 1:
                        return False
            return True

        # Check rows
        for row in board:
            if not is_valid(row):
                return False

        # Check columns
        for col in zip(*board):
            if not is_valid(col):
                return False

        # Check 3x3 subgrids
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = [board[x][y] for x in range(i, i+3) for y in range(j, j+3)]
                if not is_valid(subgrid):
                    return False

        return True

    # --------------------------------------------------- #Encod and Decode Strings

    def encode(self, strs):
        sul = ""
        for val in strs:
            sul += ';'
            sul += val
            sul += ':'

        new_sul = sul[1:-1]

        return new_sul

    def decode(self, str):
        sul = []
        word = ""
        for i in range(len(str)):
            if str[i] != ':' and str[i] != ';':
                word += str[i]
            elif str[i] == ";" and str[i-1] != ':':
                word+= str[i]
            elif str[i-1] == ':' and str[i+1] == ';':
                word += str[i]

            if str[i] == ';' and str[i-1] == ':' or i == len(str)-1:
                sul.append(word)
                word = ""

        return sul

    # --------------------------------------------------- #Longest Consecutive Sequence

    def longestConsecutive(self, nums):
        if not nums:
            return 0

        nums_set = set(nums)
        lon_con = 1
        current_streak = 0
        current_num = 0

        for i in nums_set:
            if i-1 not in nums_set:
                current_num = i
                current_streak = 1

            while current_num + 1 in nums_set:
                current_num += 1
                current_streak += 1

            lon_con = max(current_streak, lon_con)

        return lon_con

    # --------------------------------------------------- #Valid Parentheses

    def isValid(self,s):
        stack = []
        closeToOpen = {')' : '(', ']' : '[', '}' : '{'}

        for c in s:
            if c in closeToOpen:
                if stack and stack[-1] == closeToOpen[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)

        return True if not stack else False

    # --------------------------------------------------- #Evaluate Reverse Polish Notation

    def evalRPN(self,tokens):
        a = 0
        b = 0
        ops = {'+': lambda x, y: x + y,
               '-': lambda x, y: x - y,
               '*': lambda x, y: x * y,
               '/': lambda x, y: int(x / y)}
        sul_stck = []

        for i in tokens:
            if i not in ops:
                sul_stck.append(int(i))
            elif len(sul_stck) >=2:
                b = sul_stck.pop()
                a = sul_stck.pop()
                s = ops[i](a,b)
                sul_stck.append(s)

        return sul_stck[-1]

    # --------------------------------------------------- #Generate Parentheses

    def generateParenthesis(self, n):
        stack = []
        res = []

        def backtrack(openN, closeN):
            if openN == closeN == n:
                res.append("".join(stack))
                return

            if openN < n:
                stack.append('(')
                backtrack(openN+1, closeN)
                stack.pop()

            if closeN <openN:
                stack.append(')')
                backtrack(openN, closeN+1)
                stack.pop()

        backtrack(0,0)
        return res

    # --------------------------------------------------- #Daily Temperatures

    def dailyTemperatures(self, temperatures):
        res = [0]*len(temperatures)

        stack = []

        for i,t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackT, stackInd = stack.pop()
                res[stackInd] = (i - stackInd)
            stack.append([t, i])
        return res

    # --------------------------------------------------- #Is Palindrome

    def isPalindrome(self,s):
        def clean_and_lower(input_string):
            # Remove non-alphanumeric characters and convert to lowercase
            cleaned_string = ''.join(char.lower() for char in input_string if char.isalnum())
            return cleaned_string

        fixed_s = clean_and_lower(s)
        s_len = len(fixed_s) -1
        for i, n in enumerate(fixed_s):
            if n != fixed_s[s_len]:
                return False
            if i >= s_len:
                break
            s_len -= 1
        return True

    # --------------------------------------------------- #Two Sum II - Input Array Is Sorted

    def twoSum(self, numbers, target):
        sul = []
        j = len(numbers) -1
        i = 0

        while i < j:
            sum = numbers[i] + numbers[j]
            if sum > target:
                j -= 1
            elif sum == target:
                return [i +1, j+1]
            elif sum < target:
                i+= 1

        return []

    # --------------------------------------------------- #3Sum

    def threeSum(self, nums):
        sort_nums = sorted(nums)
        n = len(nums)
        sul = []

        for a in range(n - 2):
            if a > 0 and sort_nums[a] == sort_nums[a - 1]:
                continue
            b = a + 1
            c = n - 1

            while b < c:
                total_sum = sort_nums[a] + sort_nums[b] + sort_nums[c]

                if total_sum == 0:
                    sul.append([sort_nums[a], sort_nums[b], sort_nums[c]])

                    # Skip duplicate values of b and c
                    while b < c and sort_nums[b] == sort_nums[b + 1]:
                        b += 1
                    while b < c and sort_nums[c] == sort_nums[c - 1]:
                        c -= 1

                    b += 1
                    c -= 1
                elif total_sum < 0:
                    b += 1
                else:
                    c -= 1

        return sul

    # --------------------------------------------------- #Container With Most Water

    def maxArea(self, height):
        max = float('-inf')
        l = 0
        r = len(height)-1
        shift = len(height)-1

        while l < r:
            a = height[l]
            b = height[r]
            vol = min(a,b) * shift
            if max < vol:
                max = vol
            if a < b:
                l += 1
            else:
                r -= 1

            shift -= 1

        return max

    # --------------------------------------------------- #Trapping Rain Water

    def trap(self, height):
        temp = 0
        l = 0
        r = len(height)-1
        lf = False
        rf = False

        for i in range(max(height)):
            while l < r:
                if height[l] > i and not lf:
                    lf = True
                if height[r] > i and not rf:
                    rf = True
                if not lf:
                    l += 1
                if not rf:
                    r -= 1

                if height[r] > i and rf and lf:
                    r -= 1
                elif height[r] <= i and rf and lf:
                    temp += 1
                    r-= 1
            l = 0
            r = len(height) - 1
            lf = False
            rf = False

        return temp

    # --------------------------------------------------- #Binary Search

    def search(self, nums, target):
        r, l = 0, len(nums) -1

        while (l <= r):
            m = int(r + l / 2)
            if nums[m] < target:
                l = m - 1
            elif nums[m] > target:
                r = m+1
            elif nums[m] == target:
                return m

        return -1

    # --------------------------------------------------- #Search a 2D Matrix

    def searchMatrix(self,matrix, target):

        one_d_array = []
        for sublist in matrix:
            for element in sublist:
                one_d_array.append(element)

        l, r = 0, len(one_d_array) - 1

        while (l <= r):
            m = int((r + l) / 2)
            if one_d_array[m] < target:
                l = m + 1
            elif one_d_array[m] > target:
                r = m - 1
            elif one_d_array[m] == target:
                return True

        return False

    # --------------------------------------------------- #Koko Eating Bananas

    def minEatingSpeed(piles, h):
        return

    # --------------------------------------------------- #Find Minimum in Rotated Sorted Array

    def findMin(self,nums):
        l , r = 0, len(nums) -1

        if r == 0:
            return nums[r]

        while(l<=r):
            m = (r+l)//2
            if l == r+1:
                return min(nums[l] and nums[r])
            elif nums[m] < nums[m+1] and nums[m] < nums[m-1]: #nums[m] is the minimum
                return nums[m]
            elif nums[m] > nums[m+1] and nums[m] > nums[m-1]:#nums[m] is the maximum
                return nums[m+1]
            elif nums[m] > nums[l] and nums[m] > nums[r]:
                l = m+1
            else:
                r = m-1

        return -1

    # --------------------------------------------------- #Search in Rotated Sorted Array

    def search(self, nums, target):
        l, r = 0, len(nums) -1

        while l <= r:
            m = (l+r)//2
            if nums[m] == target:
                return m

            if nums[l] <= nums[m]:
                if nums[l] <= target < nums[m]:
                    r = m-1
                else:
                    l = m+1
            else:
                if nums[m] < target <= nums[r]:
                    l = m+1
                else:
                    r = m-1

        return -1

    # --------------------------------------------------- #Best Time to Buy and Sell Stock

    def maxProfit(self, prices):
        if len(prices) < 2:
            return 0

        max_profit = 0
        min_price = prices[0]

        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)

        return max_profit

    # --------------------------------------------------- #Longest Substring Without Repeating Characters

    def lengthOfLongestSubstring(self,s):
        temp = []
        count = 0
        max_count = 0

        for c in s:
            if c in temp:
                temp = temp[temp.index(c) + 1:]

            temp.append(c)
            count = len(temp)
            max_count = max(max_count, count)

        return max_count

    # --------------------------------------------------- #Happy Number

    def isHappy(self,n):
        seen = set()
        sum = 0

        while not n in seen:
            seen.add(n)
            sum = 0
            n_str = str(n)
            for d in n_str:
                sum += (int(d)*int(d))

            n = sum
            if n == 1:
                return True

        return False

    # --------------------------------------------------- #Contains Duplicate II

    def containsNearbyDuplicate(self,nums, k):
        d = {}

        for i, n in enumerate(nums):
            if n in d:
                if abs(d[n] - i) <= k:
                    return True
            d[n] = i

        return False

    # --------------------------------------------------- #Permutation in String

    def checkInclusion(self,s1, s2):
        window_size = len(s1)
        s1_counts = [0] * 26  # Assuming only lowercase English letters

        # Count the occurrences of characters in s1
        for char in s1:
            s1_counts[ord(char) - ord('a')] += 1

        # Initialize the counts for the first window in s2
        window_counts = [0] * 26
        for i in range(window_size):
            window_counts[ord(s2[i]) - ord('a')] += 1

        for i in range(len(s2) - window_size + 1):
            # Check if the current window is a permutation of s1
            if window_counts == s1_counts:
                return True

            # Move the window to the right
            if i + window_size < len(s2):
                window_counts[ord(s2[i]) - ord('a')] -= 1
                window_counts[ord(s2[i + window_size]) - ord('a')] += 1

        return False

    # --------------------------------------------------- #Reverse Linked List
    class ListNode(object):
       def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def reverseList(self,head):
        prev = None
        curr = head

        while curr is not None:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node

        return prev

    # --------------------------------------------------- #Merge Two Sorted Lists

    class ListNode(object):
       def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def mergeTwoLists(self,list1, list2):
        p1 = list1
        p2 = list2

        ret_head = ListNode()
        current = ret_head

        while p1 is not None and p2 is not None:
            if p1.val < p2.val:
                current.next = p1
                p1 = p1.next
            else:
                current.next = p2
                p2 = p2.next

            current = current.next

        if p1 is None:
            current.next = p2
        else:
            current.next = p1

        return ret_head.next

    # --------------------------------------------------- #Reorder List

    #class ListNode(object):
    #   def __init__(self, val=0, next=None):
    #        self.val = val
    #        self.next = next


    def reorderList(head):

        def reverseList(l):
            prev = None
            curr = l

            while curr is not None:
                new_node = ListNode(curr.val)
                new_node.next = prev
                prev = new_node
                curr = curr.next

            return prev

        p = head
        len = 0

        while p != None:
            len += 1
            p = p.next

        flip_list = reverseList(head)

        pl = head.next
        pr = flip_list

        pl_next = pl.next
        pr_next = pr.next

        curr = head

        for i in range(len // 2):
            curr.next = pr
            curr = curr.next
            curr.next = pl
            pl = pl.next
            pr = pr.next

            if pl.next is None or pr.next is None:
                break


        return head

    # --------------------------------------------------- #Remove Nth Node From End of List

    def removeNthFromEnd(self,head, n):
        dummy = ListNode(0)
        dummy.next = head
        slow = fast = dummy

        # Move fast pointer n+1 steps ahead
        for _ in range(n + 1):
            if fast is not None:
                fast = fast.next

        # Move both slow and fast pointers until fast reaches the end
        while fast is not None:
            slow = slow.next
            fast = fast.next

        # Remove the nth node from the end
        if slow.next is not None:
            slow.next = slow.next.next

        return dummy.next

    # --------------------------------------------------- #Remove Nth Node From End of List
    #class ListNode(object):
    #    def __init__(self, val=0, next=None):
    #       self.val = val
    #         self.next = next


    def addTwoNumbers(self,l1, l2):
        counter1 = 0
        counter2 = 0
        res = 0
        sum1 = 0
        sum2 = 0
        ret = ListNode(0)

        while l1 is not None:
            sum1 += l1.val * (10^counter1)
            counter1 += 1
            l1 = l1.next

        while l2 is not None:
            sum2 += l2.val * (10^counter2)
            counter2 += 1
            l2 = l2.next

        res = sum1+sum2
        curr = ret.next

        for c in str(res):
            curr = ListNode(int(c))
            curr = curr.next

    return ret.next

    # --------------------------------------------------- #Remove Nth Node From End of List

     # class ListNode(object):
     #     def __init__(self, x):
     #         self.val = x
     #         self.next = None

    def hasCycle(self,head):
        dummy = ListNode(0)
        p1 = dummy
        p2 = dummy
        dummy = dummy.next
        p1_ctr = 0
        p2_ctr = 0
        flag = 0

        while p2.next is not None:
            p1 = p1.next
            p2 = p2.next.next

            if p2.next == p1:
                return 1

            elif p1_ctr != 0 and p1 == p2:
                return 1

        return 0


    # --------------------------------------------------- #Remove Nth Node From End of List

    def findDuplicate(self, nums):
        sum = 0
        nums_sums = 0
        max = max(nums)

        for i in range (1,max+1):
            sum += i

        for j in nums:
            nums_sums += j

        return j-i

    # --------------------------------------------------- #Remove Nth Node From End of List

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def invertTree(root):

    if root.left is None and root.right is None:
        return root

    else:
        root.left = invertTree(root.left)
        root.right = invertTree(root.right)
        temp = root.left
        root.left = root.right
        root.right = temp

    return root





















































