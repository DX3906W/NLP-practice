class Solution {
    public int rob(int[] nums) {

        int[] record = new int[nums.length];
        int max = 0;

        if(nums.length==0) return 0;
        if(nums.length==1) return nums[0];
        if(nums.length==2) return Math.max(nums[0], nums[1]);

        record[0] = nums[0];
        record[1] = nums[1];

        for(int i=0;i<nums.length;i++){
            if(i-3>=0&&i-2>=0)
                record[i] = Math.max(record[i-3], record[i-2]) + nums[i];
            else if(i-2>=0&&i-3<0)
                record[i] = record[i-2] + nums[i];
            if(max<record[i])
                max=  record[i];
        }
        return max;

    }
}
