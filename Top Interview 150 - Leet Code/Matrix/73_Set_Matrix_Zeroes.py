class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        if not matrix:
            return []

        row_count = len(matrix)
        col_count = len(matrix[0])
        rows = set()
        cols = set()

        for i in range(row_count):
            for j in range(col_count):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)

        for i in rows:
            for j in range(col_count):
                matrix[i][j] = 0

        for j in cols:
            for i in range(row_count):
                matrix[i][j] = 0

        return matrix
