import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message
import random


n = 63   # data 길이
dv = 2   # 한 비트가 몇개의 패리티 검사식에 연결되는가?
dc = 3   # 한 패리티 검사식이 몇개의 비트와 연결되는가?
# H 패리티 검사식
# G.cT=0

H, G = make_ldpc(n, dv, dc, systematic=True, sparse=True)
# G (nxk)
k = G.shape[1]

random.seed(42) # 실험의 일정함을 위해

num_messages = 1000  # 만들 메시지 개수
random_message = np.random.randint(2, size=(num_messages, k)) #랜덤생성

snr_range = np.arange(0,10, 2)  # SNR 

ber_results = []  # Bit Error Rate 저장용

for snr in snr_range:
    total_bits = 0
    error_bits = 0

    for i in range(num_messages):
       
        message=random_message[i]

        # 인코딩 + 채널 노이즈 추가
        encoded = encode(G, message, snr)

        # 디코딩
        decoded = decode(H, encoded, snr)

        # 복원된 메시지 추출
        retrieved = get_message(G, decoded)

        # 오류 비트 계산
        total_bits += k
        error_bits += np.sum(message != retrieved)

    ber = error_bits / total_bits
    ber_results.append(ber)
    print(ber)

# 시각화 snr - ber
plt.figure(figsize=(8, 6))
plt.semilogy(snr_range, ber_results, marker='o')
plt.title("LDPC SNR - BER ")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True)
plt.show()

#시각화 n,k - ber
'''
plt.figure(figsize=(8, 6))
plt.semilogy(snr_range, ber_results, marker='o')
plt.title("LDPC Bit Error Rate vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which='both')
plt.show()'''
