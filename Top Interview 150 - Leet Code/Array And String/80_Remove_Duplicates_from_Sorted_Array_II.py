class Solution(object):
    def removeDuplicates(self, nums):
        if not nums:
            return 0

        # The position to place the next valid element
        write_index = 1
        count = 1  # To keep track of the count of the current element

        # Start from the second element of the array
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                if count < 2:
                    nums[write_index] = nums[i]
                    write_index += 1
                count += 1
            else:
                nums[write_index] = nums[i]
                write_index += 1
                count = 1

        return write_index