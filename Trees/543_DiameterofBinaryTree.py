# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def diameterOfBinaryTree(self, root):

        if root is None:
            return 0

        else:
            l_sum = 1+ self.diameterOfBinaryTree(root.left)
            r_sum = 1+ self.diameterOfBinaryTree(root.right)

            return 1+ max(l_sum, r_sum)
