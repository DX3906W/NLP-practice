class Solution {
    public int minPathSum(int[][] grid) {

        int[][] record = new int[grid.length][grid[0].length];
        record[0][0] = grid[0][0];
        for(int i=1;i<grid.length;i++){
            record[i][0] = record[i-1][0] + grid[i][0];
        }
        for(int j=1;j<grid[0].length;j++){
            record[0][j] = record[0][j-1] + grid[0][j];
        }

        for(int i=1;i<grid.length;i++){
            for(int j=1;j<grid[0].length;j++){
                record[i][j] = Math.min(record[i-1][j], record[i][j-1]) + grid[i][j];
            }
        }

        return record[grid.length-1][grid[0].length-1];

    }
}
