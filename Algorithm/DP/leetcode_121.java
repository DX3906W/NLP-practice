class Solution {
    public int maxProfit(int[] prices) {

        int max = 0;
        for(int i=0;i<prices.length-1;i++){
            int buy = prices[i];
            for(int j=i+1;j<prices.length;j++){
                if(max<prices[j]-prices[i])
                    max = prices[j] - prices[i];
            }
        }
        
        return max;

    }
}
