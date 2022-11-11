def find_valid_queries(finalDf):
    valid_Queries = (finalDf.Query.value_counts()>300)
    valid_Queries = valid_Queries[valid_Queries].index
    # print(*valid_Queries, sep='\n')
    
    return valid_Queries