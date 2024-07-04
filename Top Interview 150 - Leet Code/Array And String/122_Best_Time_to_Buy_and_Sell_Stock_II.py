class Solution(object):
    def maxProfit(self, prices):
        sum = 0
        if not prices:
            return 0

        for i in range(1, len(prices)):
            prices_change = prices[i] - prices[i-1]
            if prices_change > 0:
                sum += prices_change

        return sum


