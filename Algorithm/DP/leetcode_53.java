
class Solution {
    public int maxSubArray(int[] nums) {

        int[] record = new int[nums.length];

        int max=  nums[0];
        record[0] = nums[0];

        for(int i=1;i<nums.length;i++){
            record[i] = Integer.MIN_VALUE;
        }

        for(int i=1;i<nums.length;i++){
            record[i] = Math.max(nums[i], record[i-1] + nums[i]);
            if(record[i]>max)
                max = record[i];
        }

        return max;

    }
}
