
#5. Longest Palindromic Substring
#https://leetcode.com/problems/longest-palindromic-substring/

def dp(s):

    if not s: return 0

    if len(s)==1: return 1

    record = [[0]*10 for x in range(len(s))]

    for i in range(len(s)):
        record[i][i] = 1

    for i in range(len(s)):
        for j in range(0, i+1):
            
