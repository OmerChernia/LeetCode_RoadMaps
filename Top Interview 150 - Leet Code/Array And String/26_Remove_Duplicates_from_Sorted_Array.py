class Solution(object):
    def removeDuplicates(self, nums):
        if not nums:
            return 0

        red, green = 0,0

        for i in range(len(nums)):
            if nums[red] == nums[green]:
                green +=1
            else:
                red+=1
                temp = nums[red]
                nums[red] = nums[green]
                nums[green] = temp
                green +=1

        return red +1

