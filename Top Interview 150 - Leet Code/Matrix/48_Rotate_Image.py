class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """

        if not matrix:
            return []

        if len(matrix) != len(matrix[0]):
            return []

        if len(matrix) == 1:
            return matrix

        for i in range(len(matrix[0])):
            for j in range(len(matrix[1])):
                if (i != j) and (i > j):
                    temp = matrix[i][j]
                    matrix[i][j] = matrix[j][i]
                    matrix[j][i] = temp

        for k in range(len(matrix[0])):
            matrix[k].reverse()

        return matrix

solution = Solution()
print(solution.rotate([[1,2,3],[4,5,6],[7,8,9]])) # [[7,4,1],[8,5,2],[9,6,3]]

