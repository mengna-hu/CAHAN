# import xlwt
import numpy as np
#获取药物的邻接矩阵，矩阵中的元素为Jaccard系数
def get_DDAdjM(file,n):
    DDAM=[[0]*n for i in range(n)]   
    count=0
    with open(file, 'r') as f:
        reader = f.readlines()
        # print(reader[0])
        n = len(reader)
        for i in range(n):
            row = reader[i]
            data = str(row).strip('\n').split("\t")
            # print(data)
            r,c=int(data[0]),int(data[1])
            DDAM[r][c] = float(data[2])
            DDAM[c][r] = float(data[2])
            count+=1
        #print(count)
        return DDAM


#获取药物和疾病的邻接矩阵，有连接则元素值为1，否则为0
def get_USAdjM(file,m,n):
    USAM=[[0]*n for i in range(m)]
    count=0
    with open(file, 'r') as f:
        reader = f.readlines()
        n = len(reader)
        for i in range(n):
            row = reader[i]
            data = str(row).strip('\n').split("\t")
            # print(data)
            r,c=int(data[1]),int(data[3])
            USAM[r][c] = 1
            count+=1
            #print(r,c)
        #print(count)
        return USAM


# def ns(SCORE,m,n,threshold):
#     FLAG=[[0]*n for i in range(m)]
#     for i in range(m):
#         for j in range(n):
#             if SCORE[i][j]==0:
#                 FLAG[i][j]=-1
#             elif SCORE[i][j]>=threshold:
#                 FLAG[i][j]=1
#     return FLAG


def main():
    file='./raw/dd_simi.txt'
    m=1482
    DDAM=get_DDAdjM(file,m)
    np.save('DDAM',DDAM)

    n=793
    file2='./raw/us_edges.txt'
    USAM=get_USAdjM(file2,m,n)
    np.save('USAM',USAM)

    DDAM=np.array(DDAM)
    USAM=np.array(USAM)
    #dic:记录每种疾病关联的药物数目：ith_nonzero_num
    dic= {i: USAM[: , i].nonzero()[0] for i in range(n)}
    #print(dic)
    SCORE=[[0]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if USAM[i][j]==1:
                SCORE[i][j]=1
            else:   # 计算𝐶𝑜𝑟𝑟𝑒𝑙𝑎𝑡𝑖𝑜𝑛
                SCORE[i][j]=round(np.dot(DDAM[i],USAM[:,j])/len(dic[j]),4)
    np.save('SCORE_DD',SCORE)



if __name__=='__main__':
    main()