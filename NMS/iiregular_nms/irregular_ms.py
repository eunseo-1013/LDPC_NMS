import torch
import torch.nn as nn
import numpy as np
import pyldpc
import matplotlib.pyplot as plt
import pyldpc
import numpy as np
import sympy as sp

#WiMAX 표준 QC-LDPC base matrix를 실제 H 행렬로 확장
def read_qc_ldpc(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

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

base = read_qc_ldpc("NMS\\iiregular_nms\\wman_N0576_R34_z24.txt")
z=24
H = expand_qc_ldpc(base, z)
print("H shape:", H.shape)
print("1의 비율:", np.sum(H)/H.size)
print(H)

def generate_G(H):
    m,n=H.shape
    k=n-m
    G, _, _ = pyldpc.make_ldpc(n, k, parity_check_matrix=H)
    '''H= sp.Matrix(H)
    H, _ = H.rref()
    A=H[:,n-k:]
    print("A.shape : ")
    print(A.shape)
    A_np = np.array(A.tolist(), dtype=float)
    A_int = np.round(A_np).astype(int) # 반올림 후 정수 변환
    
    G = np.hstack((np.eye(k, dtype=int), A_int.T))
    #G=np.hstack((np.eye(k,dtype=int),A.T))'''
    return G
'''   
G=generate_G(H)
print(G.shape)
print(G)
G=G.T'''


print("-----------------------------------")
def cyclic_shift(vector, shift, Z):
    """
    GF(2) 상에서 순환 행렬 곱셈을 구현합니다.
    (벡터를 'shift'값 만큼 순환 이동 시킵니다.)
    """
    if shift == -1:  # -1은 영 행렬(Zero Matrix)을 의미
        return np.zeros(Z, dtype=int)
    
    # np.roll을 사용하여 벡터를 shift 값만큼 순환 이동
    # 모든 LDPC 비트는 정수 0 또는 1이므로 % 2 연산이 필요 없습니다.
    # (단, 앞선 GF(2) 덧셈/뺄셈 후에는 % 2가 필요합니다.)
    return np.roll(vector, shift)

# C-LDPC 인코딩 (H의 QLT 구조 이용)
def encoding_not_G(H,message):
    m,n=H.shape
    k=n-m
    # H를 정보 부분(Hm)과 패리티 부분(Hp)으로 분리
    Hm = H[:, :k]  # 0~k
    Hp = H[:, k:]  # k~n
    k=k//z
    # 메시지를 (K_b x Z) = 18x24 블록으로 재구성
    m_blocks = message.reshape((k, z))

    # 3. 신드롬 s 계산: s = Hm * m^T
    # s_blocks는 (M_b x Z) = 6x24 크기
    s_blocks = np.zeros((m, z), dtype=int)
    
    for i in range(m):       # H의 각 행 (0 ~ 5)
        for j in range(k):   # 메시지의 각 블록 (0 ~ 17)
            shift = Hm[i, j]
            if shift != -1:
                shifted_m = cyclic_shift(m_blocks[j], shift, z)
                s_blocks[i] = (s_blocks[i] + shifted_m) % 2 # GF(2) 덧셈

    # 4. 패리티 p 계산 (순방 대입): Hp * p^T = s
    # p_blocks는 (M_b x Z) = 6x24 크기
    p_blocks = np.zeros((m, z), dtype=int)

    for i in range(m):  # 각 패리티 블록 p_i (i=0...5)
        # a) 이미 계산된 p_j (j < i)의 영향 합산
        sum_prev_p = np.zeros(z, dtype=int)
        for j in range(i):  # j = 0 부터 i-1 까지
            shift = Hp[i, j]
            if shift != -1:
                shifted_p = cyclic_shift(p_blocks[j], shift, z)
                sum_prev_p = (sum_prev_p + shifted_p) % 2

        # b) p_i가 만족해야 할 목표 벡터 계산
        # B(i,i)*p_i = s_i + sum(B(i,j)*p_j) (for j < i)
        target = (s_blocks[i] + sum_prev_p) % 2

        # c) p_i 계산: p_i = (B(i,i))^-1 * target
        # B(i,i)는 'shift'값의 순환 행렬입니다.
        # (B(i,i))^-1는 '-shift' (또는 Z-shift) 값의 순환 행렬입니다.
        diag_shift = Hp[i, i]
        
        if diag_shift == -1:
            raise ValueError(f"오류: H_p[{i},{i}]가 영 행렬입니다. 인코딩 불가.")
        
        # 순환 역행렬 곱셈 (즉, 반대 방향 시프트)
        p_blocks[i] = cyclic_shift(target, -diag_shift, z)
    
    # 1D 벡터로 펼쳐서 반환
    parity = p_blocks.flatten()
    codeword = np.concatenate([message, parity]).astype(int)
    return codeword

# --- 검증 함수 ---
    






# 결과 저장 리스트
loss_data = []
ber_data = []
lr=0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --- Neural Min-Sum Decoder ---
class NMSDecoder(nn.Module):
    def __init__(self, H_rows, H_cols, n_vars, n_checks, num_iterations=5):
        super(NMSDecoder, self).__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.num_iterations = num_iterations
        self.weights = nn.Parameter(torch.full((num_iterations, 1), 0.5))
        self.H_rows = H_rows
        self.H_cols = H_cols
        H_rows = H_rows.to(device)
        H_cols=H_cols.to(device)

    def forward(self, channel_llrs):
        batch_size = channel_llrs.shape[0]
        v2c_msgs = torch.zeros(batch_size, len(self.H_cols))
        c2v_msgs = torch.zeros(batch_size, len(self.H_cols))

        for i in range(self.num_iterations):
            # 1. Variable → Check
            c2v_aggregated = torch.zeros(batch_size, self.n_vars)
            c2v_aggregated = c2v_aggregated.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
            var_llrs = channel_llrs + c2v_aggregated
            v2c_msgs = var_llrs[:, self.H_cols] - c2v_msgs

            # 2. Check → Variable (Min-Sum)
            c2v_msgs_temp = torch.zeros_like(v2c_msgs)
            for c_idx in range(self.n_checks):
                connected_edges = (self.H_rows == c_idx).nonzero().squeeze(-1)
                if connected_edges.numel() == 0:
                    continue

                incoming_msgs = v2c_msgs[:, connected_edges]
                if incoming_msgs.dim() == 1:
                    incoming_msgs = incoming_msgs.unsqueeze(1)

                signs = torch.prod(torch.sign(incoming_msgs), dim=1, keepdim=True)
                abs_vals = torch.abs(incoming_msgs)

                for j, edge_idx in enumerate(connected_edges):
                    other_indices = [l for l in range(len(connected_edges)) if l != j]
                    other_msgs_abs = abs_vals[:, other_indices]
                    if other_msgs_abs.shape[1] == 0:
                        min_abs_vals = torch.zeros(batch_size)
                    else:
                        min_abs_vals = torch.min(other_msgs_abs, dim=1).values

                    sign_in = torch.sign(incoming_msgs[:, j])
                    sign_in[sign_in == 0] = 1
                    out_sign = signs.squeeze(1) / sign_in
                    c2v_msgs_temp[:, edge_idx] = out_sign * min_abs_vals

            c2v_msgs = c2v_msgs_temp * self.weights[i]

        final_c2v_aggregated = torch.zeros(batch_size, self.n_vars)
        final_c2v_aggregated = final_c2v_aggregated.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
        final_llrs = channel_llrs + final_c2v_aggregated

        return final_llrs


# --- 데이터 생성 함수 ---
def generate_data(batch_size, n_bits, k_bits, snr_db, H):
    messages_np = np.random.randint(0, 2, size=(batch_size, k_bits))
    '''if hasattr(G_matrix, "toarray"):
        G_matrix = G_matrix.toarray()'''
    
    codewords_list = []
    for i in range(batch_size):
        message = messages_np[i]
        codeword = encoding_not_G(H,message)
        codewords_list.append(codeword)

    codewords_np = np.vstack(codewords_list)
    messages = torch.from_numpy(messages_np).float()
    codewords = torch.from_numpy(codewords_np).float()

    transmitted_signal = 1 - 2 * codewords
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / (2 * (k_bits / n_bits) * snr_linear)
    noise = torch.randn_like(transmitted_signal) * np.sqrt(noise_variance)
    received_signal = transmitted_signal + noise
    channel_llrs = 2 * received_signal / noise_variance 
    
    messages = torch.from_numpy(messages_np).float()
    codewords = torch.from_numpy(codewords_np).float()
    return channel_llrs, messages


# --- 2. 훈련 루프 ---
if __name__ == '__main__':
    EPOCHS = 5
    BATCH_SIZE = 10
    SNR_DB = 4.0
    LEARNING_RATE=0.5
    m,n=H.shape
    k = n-m


    # 인덱스 준비
    H_rows, H_cols = H.nonzero()
    H_rows = torch.tensor(H_rows, dtype=torch.long)
    H_cols = torch.tensor(H_cols, dtype=torch.long)

    # 모델 및 최적화 설정
    model = NMSDecoder(H_rows, H_cols, n_vars=n, n_checks=m, num_iterations=5)
    #model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    print("--- 훈련 시작 ---")
    for epoch in range(EPOCHS):
        train_llrs, train_messages = generate_data(BATCH_SIZE, n, k, SNR_DB, H)
        #train_llrs = train_llrs.to(device)
        train_messages = train_messages.to(device)
        output_llrs = model(train_llrs)

        # LLR 부호 반전 후 loss 계산
        logits_for_1 = -output_llrs[:,: k]
        loss = criterion(logits_for_1, train_messages)

        # Hard decision (LLR < 0 → 1)
        decoded_codeword_bits = (output_llrs < 0).float()
        decoded_message_bits = decoded_codeword_bits[:, :k]

        errors = (decoded_message_bits != train_messages).sum().item()
        current_ber = errors / (BATCH_SIZE * k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_data.append(loss.item())
        ber_data.append(current_ber)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}, BER: {current_ber:.6f}")

    print("--- 훈련 종료 ---")
    # --- 3. 결과 플롯 ---
    date = np.array(range(1, EPOCHS + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(date, loss_data)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCEWithLogits)")
    plt.title("Training Loss per learning rate")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(date, ber_data)
    plt.xlabel("Epoch")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("Bit Error Rate vs. Epoch per learning rate")
   # legends = [f"learning rate={lr[i]}, SNR={SNR_DB}" for i in range(len(lr))]
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
