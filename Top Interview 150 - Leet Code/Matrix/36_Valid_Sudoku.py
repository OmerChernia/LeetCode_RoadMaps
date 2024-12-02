class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        row = [set() for _ in range(9)]
        col = [set() for _ in range(9)]
        box = [set() for _ in range(9)]

        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    continue
                if (board[i][j] in row[i] or
                    board[i][j] in col[j] or
                    board[i][j] in box[(i // 3) * 3 + (j // 3)]):
                    return False
                row[i].add(board[i][j])
                col[j].add(board[i][j])
                box[(i // 3) * 3 + (j // 3)].add(board[i][j])
        return True

solution = Solution()
print(solution.isValidSudoku([["5","3",".",".","7",".",".",".","."],
                               ["6",".",".","1","9","5",".",".","."],
                               [".","9","8",".",".",".",".","6","."],
                               ["8",".",".",".","6",".",".",".","3"],
                               ["4",".",".","8",".","3",".",".","1"],
                               ["7",".",".",".","2",".",".",".","6"],
                               [".","6",".",".",".",".","2","8","."],
                               [".",".",".","4","1","9",".",".","5"],
                               [".",".",".",".","8",".",".","7","9"]]))