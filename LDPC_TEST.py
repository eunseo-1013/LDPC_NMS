import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message

# -------------------------------
# 1. LDPC 파라미터 설정
# -------------------------------
n = 63   # 코드워드 길이
dv = 2   # variable node degree
dc = 3   # check node degree
H, G = make_ldpc(n, dv, dc, systematic=True, sparse=True)
k = G.shape[1]


num_messages = 1000  # 만들 메시지 개수
random_message = np.random.randint(2, size=(num_messages, k)) #랜덤생성

snr_range = np.arange(0, 11, 2)  # SNR 0, 2, 4, 6, 8, 10 dB

ber_results = []  # Bit Error Rate 저장용

# -------------------------------
# 3. 시뮬레이션
# -------------------------------
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
    print(f"SNR={snr}dB -> BER={ber:.5f}")

# -------------------------------
# 4. 결과 시각화
# -------------------------------
plt.figure(figsize=(7, 5))
plt.semilogy(snr_range, ber_results, marker='o')
plt.title("LDPC Bit Error Rate vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which='both')
plt.show()
