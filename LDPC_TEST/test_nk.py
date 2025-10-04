import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message
import random
import time

#ldpc 파라미터
n = 63   # data + parity

dv_range=np.arange(2,8,2)
dc_range=[3,7,9]
 # 한 비트가 몇개의 패리티 검사식에 연결되는가?
 # 한 패리티 검사식이 몇개의 비트와 연결되는가?

snr_range=np.arange(0,8,2)

time_list=[]
plt.figure(figsize=(10, 6))
for i in range(len(dv_range)):
    dv=dv_range[i]
    dc=dc_range[i]

    H, G = make_ldpc(n, dv, dc, systematic=True, sparse=True)
    k = G.shape[1]

    random.seed(42) # 실험의 일정함을 위해
    num_messages = 1000  
    random_message = np.random.randint(2, size=(num_messages, k)) 

    start=time.time()
    ber_results=[]
    for snr in snr_range:
        total_bits = 0
        error_bits = 0
        for a in range(num_messages):
            message=random_message[a]
            # 인코딩 + 채널 노이즈 추가
            encoded = encode(G, message, snr)
            # 디코딩
            decoded = decode(H, encoded, snr,maxiter=500)
            # 복원된 메시지 추출
            retrieved = get_message(G, decoded)
            # 오류 비트 계산
            total_bits += k
            error_bits += np.sum(message != retrieved)

        ber = error_bits / total_bits
        ber_results.append(ber)
    end=time.time()
    time_list.append(end-start)
    plt.semilogy(snr_range, ber_results, marker='o', label=f"dv={dv}, dc={dc}")

plt.title("LDPC SNR - BER for Different (dv,dc)")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend(loc='upper right')   # plot에 지정한 label 불러서 범례 생성
plt.grid(True)
plt.show()


plt.figure(figsize=(8,6))
x_labels = [f"({dv},{dc})" for dv,dc in zip(dv_range, dc_range)]
plt.plot(x_labels, time_list, marker='o')
plt.title("time per n,k")
plt.xlabel("(n,k)")
plt.ylabel("total_time")
plt.grid()
plt.show()





