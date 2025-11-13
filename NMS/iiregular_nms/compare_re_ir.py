import torch
import torch.nn as nn
import numpy as np
import pyldpc
import matplotlib.pyplot as plt
import pyldpc
import numpy as np
import sympy as sp

#WiMAX 표준 QC-LDPC base matrix를 실제 H 행렬로 확장



z=24
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




'''
def generate_G(H):
    m,n=H.shape
    k=n-m
    G, _, _ = pyldpc.make_ldpc(n, k, parity_check_matrix=H)
    H= sp.Matrix(H)
    H, _ = H.rref()
    A=H[:,n-k:]
    print("A.shape : ")
    print(A.shape)
    A_np = np.array(A.tolist(), dtype=float)
    A_int = np.round(A_np).astype(int) # 반올림 후 정수 변환
    G = np.hstack((np.eye(k, dtype=int), A_int.T))
    #G=np.hstack((np.eye(k,dtype=int),A.T))

    return G 

G=generate_G(H)

print(G.shape)

print(G)

G=G.T '''

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
def encoding_not_G(H, message):
    m, n = H.shape
    k = n - m
    K_b = k // z
    M_b = m // z
    # 1. H를 정보 부분과 패리티 부분으로 분리
    Hm = H[:, :k]
    Hp = H[:, k:]
    # 2. 메시지 블록화 (K_b x Z)
    m_blocks = message.reshape((K_b, z))
    # 3. 신드롬 s = Hm * m^T 계산
    s_blocks = np.zeros((M_b, z), dtype=int)
    for i in range(M_b):
        for j in range(K_b):
            shift = Hm[i*z:(i+1)*z, j*z:(j+1)*z]
            if np.any(shift):
                shift_idx = np.where(shift[0])[0][0]
                shifted_m = np.roll(m_blocks[j], shift_idx)
                s_blocks[i] = (s_blocks[i] + shifted_m) % 2
    # 4. 패리티 계산
    p_blocks = np.zeros((M_b, z), dtype=int)
    for i in range(M_b):
        sum_prev_p = np.zeros(z, dtype=int)
        for j in range(i):
            shift = Hp[i*z:(i+1)*z, j*z:(j+1)*z]
            if np.any(shift):
                shift_idx = np.where(shift[0])[0][0]
                shifted_p = np.roll(p_blocks[j], shift_idx)
                sum_prev_p = (sum_prev_p + shifted_p) % 2
        target = (s_blocks[i] + sum_prev_p) % 2
        diag_shift = np.where(Hp[i*z, i*z:(i+1)*z])[0][0]
        p_blocks[i] = np.roll(target, -diag_shift)
    parity = p_blocks.flatten()
    codeword = np.concatenate([message, parity]).astype(int)

    return codeword




#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", device)


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

        #H_rows = H_rows.to(device)

        #H_cols=H_cols.to(device)

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

        codeword = encoding_not_G(H,message) # g 행렬 없이 codeword 만듦!

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



# --- 데이터 생성 함수 ---

def generate_data_regular(batch_size, n_bits, k_bits, snr_db, G_matrix):

    messages_np = np.random.randint(0, 2, size=(batch_size, k_bits))
    if hasattr(G_matrix, "toarray"):
        G_matrix = G_matrix.toarray()

    codewords_list = []
    for i in range(batch_size):
        message = messages_np[i]
        codeword = (message @ G_matrix) % 2
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

    return channel_llrs, messages




'''

base = read_qc_ldpc("NMS\\iiregular_nms\\wman_N0576_R34_z24.txt")
z=24
H = expand_qc_ldpc(base, z)
print("H shape:", H.shape)
print("1의 비율:", np.sum(H)/H.size)
print(H)
# 결과 저장 리스트'''

loss_data = []

ber_data = []

fer_data=[]
lr=0.005



# --- 2. 훈련 루프 ---

if __name__ == '__main__':
    EPOCHS = 5
    BATCH_SIZE = 500
    SNR_DB = 4.0
    LEARNING_RATE=0.005
    print("--------------- regural ----------------------")
    # R 비율 비슷하게 ( 같은 n 에서 )
    H, G = pyldpc.make_ldpc(576, 3, 12, systematic=True, sparse=True)
    G = G.T
    k = G.shape[0]
    n = G.shape[1]
    m = H.shape[0]
    print("regular-------------")
    print("H shape:", H.shape)
    print("1의 비율:", np.sum(H)/H.size)
    print(f"regural LDPC 코드 생성 완료: n={n}, k={k}, m={m} (G.shape={G.shape})")
    # 인덱스 준비
    H_rows, H_cols = H.nonzero()
    H_rows = torch.tensor(H_rows, dtype=torch.long)
    H_cols = torch.tensor(H_cols, dtype=torch.long)
    G_for_encode = G.T
    # ---- (수정된 부분) systematic 위치 계산 ----
    G_enc_np = np.array(G_for_encode)
    sys_positions = []
    for jcol in range(G_enc_np.shape[1]):
        col = G_enc_np[:, jcol]
        ones = np.where(col == 1)[0]
        if len(ones) == 1:
            sys_positions.append(int(ones[0]))
        else:
            # fallback (비시스템매틱 대응)
            sys_positions.append(jcol)
    sys_positions = np.array(sys_positions, dtype=int)
    #print("Systematic positions:", sys_positions)
    # 모델 및 최적화 설정
    model = NMSDecoder(H_rows, H_cols, n_vars=n, n_checks=m, num_iterations=5)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    current_loss_lst = []
    current_ber_lst = []
    current_fer_lst=[]


    print("--- regular ldpc 훈련 시작 ---")
    for epoch in range(EPOCHS):
        train_llrs, train_messages = generate_data_regular(BATCH_SIZE, n, k, SNR_DB, G_for_encode)
        output_llrs = model(train_llrs)
        # LLR 부호 반전 후 loss 계산
        logits_for_1 = -output_llrs[:, sys_positions]
        loss = criterion(logits_for_1, train_messages)
        # Hard decision (LLR < 0 → 1)
        decoded_codeword_bits = (output_llrs < 0).float()
        decoded_message_bits = decoded_codeword_bits[:, sys_positions]


        errors = (decoded_message_bits != train_messages).sum().item()
        current_ber = errors / (BATCH_SIZE * k)
        errors_matrix = (decoded_message_bits != train_messages)
        frame_has_error = torch.any(errors_matrix, dim=1)
        # 에러가 있는 프레임(True)의 총 개수를 셉니다.
        frame_errors_total = frame_has_error.sum().item()
        # 현재 배치의 FER을 계산합니다.
        current_fer = frame_errors_total / BATCH_SIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_loss_lst.append(loss.item())
        current_ber_lst.append(current_ber)
        current_fer_lst.append(current_fer)
        #if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}, BER: {current_ber:.6f}")
    print("--- regular ldpc 훈련 종료 ---")

    loss_data.append(current_loss_lst)
    fer_data.append(current_fer_lst)
    ber_data.append(current_ber_lst)

    print("--------------------------------------------")

    # 인덱스 준비
    base = read_qc_ldpc("NMS\\iiregular_nms\\wman_N0576_R34_z24.txt")
    z=24
    H = expand_qc_ldpc(base, z)
    print("H shape:", H.shape)
    print("1의 비율:", np.sum(H)/H.size)
    print(H)
    m,n=H.shape
    k = n-m
    H_rows, H_cols = H.nonzero()
    H_rows = torch.tensor(H_rows, dtype=torch.long)
    H_cols = torch.tensor(H_cols, dtype=torch.long)
    # 모델 및 최적화 설정

    model = NMSDecoder(H_rows, H_cols, n_vars=n, n_checks=m, num_iterations=5)
    #model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- 훈련 시작(irrgular) ---")
    current_loss_lst = []
    current_ber_lst = []
    current_fer_lst=[]

    for epoch in range(EPOCHS):

        train_llrs, train_messages = generate_data(BATCH_SIZE, n, k, SNR_DB, H)

        #train_llrs = train_llrs.to(device)

        #train_messages = train_messages.to(device)

        output_llrs = model(train_llrs)
        # LLR 부호 반전 후 loss 계산
        logits_for_1 = -output_llrs[:,: k]
        loss = criterion(logits_for_1, train_messages)
        # Hard decision (LLR < 0 → 1)
        decoded_codeword_bits = (output_llrs < 0).float()
        decoded_message_bits = decoded_codeword_bits[:, :k]
        errors = (decoded_message_bits != train_messages).sum().item()
        current_ber = errors / (BATCH_SIZE * k)

        errors_matrix = (decoded_message_bits != train_messages)
        frame_has_error = torch.any(errors_matrix, dim=1)
        # 에러가 있는 프레임(True)의 총 개수를 셉니다.
        frame_errors_total = frame_has_error.sum().item()
        # 현재 배치의 FER을 계산합니다.
        current_fer = frame_errors_total / BATCH_SIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_loss_lst.append(loss.item())
        current_ber_lst.append(current_ber)
        current_fer_lst.append(current_fer)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}, BER: {current_ber:.6f}")
    print("--- 훈련 종료 ---")
    loss_data.append(current_loss_lst)
    ber_data.append(current_ber_lst)
    fer_data.append(current_fer_lst)

    # -------------------- 결과 그래프 ------------------------------
    # 2. 서브플롯 생성: 2행 1열, x축 공유

    date = np.array(range(1, EPOCHS + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # 두 그래프 사이의 수직 간격을 좁힙니다.
    fig.subplots_adjust(hspace=0.05) 

    # 3. 중요: 두 개의 서브플롯 모두에 동일한 데이터를 플롯합니다.
    ax1.plot(date, loss_data[0], label="regular")
    ax1.plot(date, loss_data[1], label="irregular")
    ax2.plot(date, loss_data[0], label="regular")
    ax2.plot(date, loss_data[1], label="irregular")

    # 4. y축 범위 설정 (핵심)
    # ax1 (위쪽 그래프)는 높은 값 범위를 표시
    ax1.set_ylim(15, 17)  # 9.5 ~ 11 사이의 값만 표시
    # ax2 (아래쪽 그래프)는 낮은 값 범위를 표시
    ax2.set_ylim(0.5, 2)   # 0.5 ~ 2 사이의 값만 표시

    # 5. 축 숨기기
    # 위쪽 그래프(ax1)의 아래쪽 축선을 숨깁니다.
    ax1.spines['bottom'].set_visible(False)
    # 아래쪽 그래프(ax2)의 위쪽 축선을 숨깁니다.
    ax2.spines['top'].set_visible(False)
    # 위쪽 그래프(ax1)의 x축 틱(눈금)을 제거합니다. (레이블은 sharex=True로 이미 숨겨짐)
    ax1.tick_params(axis='x', length=0)
    ax2.xaxis.tick_bottom()

    # 6. 끊어진 축을 시각적으로 표시하는 대각선 '//' 그리기
    d = .015  # 대각선 크기
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # 위쪽 그래프의 좌하단
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 위쪽 그래프의 우하단

    kwargs.update(transform=ax2.transAxes)  # 좌표계를 아래쪽 그래프로 변경
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 아래쪽 그래프의 좌상단
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 아래쪽 그래프의 우상단

    # 7. 레이블 및 제목 설정
    ax1.set_title("Loss (regular vs irregular)")
    ax2.set_xlabel("Epoch")
    # Y축 레이블은 두 그래프 중앙에 공통으로 추가
    fig.text(0.04, 0.5, 'Loss (BCEWithLogits)', va='center', rotation='vertical')

    # 8. 범례(Legend) 및 그리드
    # 범례는 한쪽에만 표시해도 됩니다.
    ax1.legend(loc='upper right') 
    ax1.grid(True)
    ax2.grid(True)

    plt.show()


   
    plt.figure(figsize=(10, 6))
    plt.plot(date, loss_data[0],label="regular")
    plt.plot(date,loss_data[1],label="irregular")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCEWithLogits)")
    plt.title("Loss (regular vs irregular)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



    plt.figure(figsize=(10, 6))
    plt.plot(date, ber_data[0],label="regular")
    plt.plot(date,ber_data[1],label="irregular")
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER (regular vs irregular)")
   # legends = [f"learning rate={lr[i]}, SNR={SNR_DB}" for i in range(len(lr))]
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()



    plt.figure(figsize=(10, 6))
    plt.plot(date, fer_data[0],label="regular")
    plt.plot(date,fer_data[1],label="irregular")
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Frame Error Rate (FER)")
    plt.title("FER (regular vs irregular)")
   # legends = [f"learning rate={lr[i]}, SNR={SNR_DB}" for i in range(len(lr))]
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()