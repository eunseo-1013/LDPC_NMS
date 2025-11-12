import torch
import torch.nn as nn
import numpy as np
import pyldpc
import matplotlib.pyplot as plt
import pyldpc
import numpy as np

#WiMAX 표준 QC-LDPC 부호 사용
#QC-LDPC base matrix를 실제 H 행렬로 확장하는 함수
import numpy as np

def read_qc_ldpc(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 줄 수 = m, 각 줄 요소 수 = n
    m = len(lines)
    n = len(lines[0].split())

    base = np.zeros((m, n), dtype=int)
    for i, line in enumerate(lines):
        base[i] = list(map(int, line.split()))
    return base


def expand_qc_ldpc(base, z):
    m, n = base.shape
    H = np.zeros((m*z, n*z), dtype=int)

    for i in range(m):
        for j in range(n):
            shift = base[i, j]
            if shift >= 0:
                I = np.eye(z, dtype=int)
                H[i*z:(i+1)*z, j*z:(j+1)*z] = np.roll(I, shift, axis=1)
    return H

# 예시 사용
base = read_qc_ldpc("NMS\\iiregular_nms\\wman_N0576_R34_z24.txt")
H = expand_qc_ldpc(base, 24)
print("H shape:", H.shape)
print("1의 비율:", np.sum(H)/H.size)



'''
# codeword 길이 ,dv,dc,dict로 지정하면 가변(Irregular) LDPC 코드를 생성함
#G 생성 할 것인가? ,희소형태로 h,g 만들 것인가?,난수 seed
H, G = pyldpc.make_ldpc(8, [0,0,4,4], [0,0,0,0,5], systematic=True, sparse=True,seed=42)
SNR=4
print(G.T) # 실제 G 행렬
G=G.T
print(f"G={G.shape[0]}x{G.shape[1]}")
#print(G)
print("------------")

m_len=G.shape[0] # 메세지 비트 크기
#c=G.shape[1] # codeword 크기

print("H (Parity-check matrix):\n", H.toarray())
print("\nG (Generator matrix):\n", G.toarray())

SNR=4
G=[[1 ,0 ,0, 0, 0 ,0 ,1, 0],
 [0, 1, 0, 0, 0, 1, 0, 0],
 [0, 0, 1, 0, 0, 1, 1, 1],
 [0, 0, 0, 1, 0, 0, 0, 1],
 [0, 0, 0, 0, 1, 1, 1, 1]]

H = [[0, 1, 1, 0, 1, 1, 0, 0],
     [1, 0, 1, 0, 1, 0, 1, 0],
     [0, 0, 1, 1, 1, 0, 0, 1]]'''
'''
m_len=5 # 메세지 비트 크기
w=[] # 초기 가중치 
H=np.array(H)
m=np.zeros(m_len,dtype=int)
m=np.array(m)
c=m@G
print(np.array(c))

y = c
y[1]=1 # 일단 노이즈...
print("-------------------------")


r=[] # 초기 r
for i in range(H.shape[1]):
    r.append(4*y[i]*(len(y)/len(m))*SNR)

print('----- r -----')
print(r)

M=[] # m x y == 3 x 8
for j in range(H.shape[0]):
    temp=[]
    for i in range(len(y)):
       temp.append(r[j])
    M.append(temp)

print("-------- M --------")
print(M)
M=np.array(M) ## 3 x 8

   
E_ij=[] 
for j in range(H.shape[0]):
    E=0
    temp=[]
    for i in range(H.shape[1]):
        if(H[j][i]==0): 
            temp.append(torch.tensor(0)) 
            continue
        #---  연결 되어져 있다면 -----
        # 자기 자신 제외 하기
        mask = np.ones(len(c), dtype=bool)
        mask[i] = False
        filtered_M=M[j][mask]
        
        # min 뽑기
        min=np.min(np.abs(filtered_M))
        # 부호들의 곱
        sign = np.sign(filtered_M)
        sign[sign==0]=1
        product_sign= np.prod(sign)

        temp.append(torch.tensor(product_sign * min ))
        
    E_ij.append(temp)

print("------- EIj ----------")
print(E_ij)
    
L=[]
for i in range(len(E_ij[0])):
    temp=r[i]
    for j in range(len(E_ij)):
        temp=temp+E_ij[j][i]
    L.append(temp)
print("------ L ----------")
print(L)
Z=[]
for i in range(len(L)):
    if(L[i]>=0):
        Z.append(0)
    else:
        Z.append(1)

print("----------- Z ---------")
print(Z)



 '''      






