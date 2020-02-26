class Solution {
    public int climbStairs(int n) {
        int[] record = new int[n];

        if(n == 1) return 1;

        record[0] = 1;
        record[1] = 1;

        for(int i=0;i<n;i++){
            if(i+1<n)
                record[i+1] += record[i];
            if(i+2<n)
                record[i+2] += record[i];
        }

        return record[n-1];

    }
}
