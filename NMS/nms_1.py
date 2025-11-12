import torch
import torch.nn as nn
import numpy as np
import pyldpc
import matplotlib.pyplot as plt

# --- 1. LDPC 파라미터 및 H 행렬 생성 (pyldpc 사용) ---
n_in = 63 # 전체 부호어 길이 (N)
dv = [3] # H 행렬 열 가중치 (dv)
dc = [7] # H 행렬 행 가중치 (dc)




# 결과를 저장할 리스트 초기화
loss_data = [] # [[loss_lst for dv,dc=2,3], [loss_lst for dv,dc=3,7]]
ber_data = []  # [[ber_lst for dv,dc=2,3], [ber_lst for dv,dc=3,7]]

# --- NMSDecoder 클래스 정의 (외부로 이동) ---
class NMSDecoder(nn.Module):
    """Neural Min-Sum (NMS) 디코더 PyTorch 모델"""
    def __init__(self, H_rows, H_cols, n_vars, n_checks, num_iterations=5):
        super(NMSDecoder, self).__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.num_iterations = num_iterations
        # 0.5로 초기화된 학습 가능한 가중치 (Min-Sum 보정 계수)
        self.weights = nn.Parameter(torch.full((num_iterations, 1), 0.5))
        self.H_rows = H_rows
        self.H_cols = H_cols

    def forward(self, channel_llrs):
        batch_size = channel_llrs.shape[0]
        v2c_msgs = torch.zeros(batch_size, len(self.H_cols))
        c2v_msgs = torch.zeros(batch_size, len(self.H_cols))
        
        for i in range(self.num_iterations):
            # 1. V2C Update (Variable to Check)
            c2v_aggregated = torch.zeros(batch_size, self.n_vars)
            c2v_aggregated = c2v_aggregated.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
            var_llrs = channel_llrs + c2v_aggregated
            v2c_msgs = var_llrs[:, self.H_cols] - c2v_msgs
            
            # 2. C2V Update (Check to Variable - Min-Sum)
            c2v_msgs_temp = torch.zeros_like(v2c_msgs)
            
            # Matplotlib은 NMSDecoder 내부에 있을 때 H_rows, H_cols에 접근할 수 없습니다.
            # 이 코드는 성능상의 이유로 C++ 등으로 구현하는 것이 좋지만,
            # 현재 코드 구조를 유지하기 위해 그대로 사용합니다.
            for c_idx in range(self.n_checks):
                connected_edges = (self.H_rows == c_idx).nonzero().squeeze(-1)
                if connected_edges.numel() == 0: continue
                incoming_msgs = v2c_msgs[:, connected_edges]
                if incoming_msgs.dim() == 1: incoming_msgs = incoming_msgs.unsqueeze(1)
                
                # Min-Sum 알고리즘: 부호 계산 및 최소 절대값 계산
                signs = torch.prod(torch.sign(incoming_msgs), dim=1, keepdim=True)
                abs_vals = torch.abs(incoming_msgs)
                
                for j, edge_idx in enumerate(connected_edges):
                    # 현재 엣지를 제외한 나머지 메시지의 절대값
                    other_indices = [l for l in range(len(connected_edges)) if l != j]
                    other_msgs_abs = abs_vals[:, other_indices]
                    
                    if other_msgs_abs.shape[1] == 0: 
                        min_abs_vals = torch.zeros(batch_size)
                    else: 
                        min_abs_vals = torch.min(other_msgs_abs, dim=1).values
                        
                    # 최종 부호 계산 (나에게서 나가는 메시지의 부호 = (모두의 부호) / (나에게 들어온 메시지의 부호))
                    sign_in = torch.sign(incoming_msgs[:, j])
                    sign_in[sign_in == 0] = 1 # 0이면 1로 처리하여 나누기 오류 방지
                    out_sign = signs.squeeze(1) / sign_in
                    
                    c2v_msgs_temp[:, edge_idx] = out_sign * min_abs_vals
            
            # 3. Neural Weight 적용
            c2v_msgs = c2v_msgs_temp * self.weights[i]
            
        # 4. 최종 LLR 계산
        final_c2v_aggregated = torch.zeros(batch_size, self.n_vars)
        final_c2v_aggregated = final_c2v_aggregated.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
        final_llrs = channel_llrs + final_c2v_aggregated
        
        return final_llrs

# --- 데이터 생성 함수 정의 (외부로 이동) ---
def generate_data(batch_size, n_bits, k_bits, snr_db, G_matrix):
    """훈련용 데이터 배치 생성 (실제 LDPC 부호화 적용)"""
    # 메시지 생성 (K 비트)
    messages_np = np.random.randint(0, 2, size=(batch_size, k_bits))
    
    # G 행렬이 sparse일 경우 toarray() 변환
    if hasattr(G_matrix, "toarray"):
        G_matrix = G_matrix.toarray()
        
    # pyldpc.encode는 (N x K) 형태의 G를 기대하므로 G_matrix는 (63, 22) 형태여야 함
    # (이미 전치되어 넘어왔다고 가정)
    
    codewords_list = []
    for i in range(batch_size):
        message = messages_np[i]
        # pyldpc.encode는 G_matrix가 (N x K) 형태일 때 정상 동작
        codeword = pyldpc.encode(G_matrix, message, snr_db) 
        codewords_list.append(codeword)
    
    codewords_np = np.vstack(codewords_list)

    # 채널 전송 및 수신
    messages = torch.from_numpy(messages_np).float()
    codewords = torch.from_numpy(codewords_np).float()
    transmitted_signal = 1 - 2 * codewords # BPSK 변조: 0 -> 1, 1 -> -1
    
    # 잡음 분산 계산: $\sigma^2 = 1 / (2 \cdot R \cdot SNR_{\text{lin}})$
    snr_linear = 10 ** (snr_db / 10.0)
    # R = k/n
    noise_variance = 1.0 / (2 * (k_bits / n_bits) * snr_linear) 
    
    noise = torch.randn_like(transmitted_signal) * np.sqrt(noise_variance)
    received_signal = transmitted_signal + noise
    
    # LLR 계산: L = 4 * R / $\sigma^2$
    channel_llrs = 2 * received_signal / noise_variance
    
    return channel_llrs, messages

# --- 2. 훈련 루프 ---
if __name__ == '__main__':
    EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    SNR_DB = 4.0
    
    for i in range(len(dv)):
        d_v = dv[i]
        d_c = dc[i]
        label = f"dv={d_v}, dc={d_c}"
        
        # LDPC 코드 생성: N=63
        H, G = pyldpc.make_ldpc(n_in, d_v, d_c, systematic=True, sparse=True)
        
        # G 행렬의 크기가 (N x K) 형태로 생성되었다고 가정하고 (pyldpc의 특징),
        # G 행렬을 (K x N) 형태로 전치하여 k와 n을 정확히 정의합니다.
        if G.shape[0] > G.shape[1]:
            G = G.T
        
        k = G.shape[0] # 메시지 길이 (K)
        n = G.shape[1] # 부호어 길이 (N)
        m = H.shape[0] # 패리티 비트 수 (M)
        
        print(f"\n--- {label} 코드 ---")
        print(f"LDPC 코드 생성 완료: n={n}, k={k}, m={m} (G.shape={G.shape})")
        
        # 디코딩에 사용할 H 행렬의 인덱스
        H_rows, H_cols = H.nonzero()
        H_rows = torch.tensor(H_rows, dtype=torch.long)
        H_cols = torch.tensor(H_cols, dtype=torch.long)
        
        # pyldpc.encode에 사용할 G 행렬은 (N x K) 형태여야 함
        G_for_encode = G.T 
        
        # 모델 및 최적화 설정
        model = NMSDecoder(H_rows, H_cols, n_vars=n, n_checks=m, num_iterations=5)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        current_loss_lst = []
        current_ber_lst = []
        
        print("--- 훈련 시작 ---")
        for epoch in range(EPOCHS):
            # 훈련 데이터 생성 및 G_for_encode (N x K) 전달
            train_llrs, train_messages = generate_data(BATCH_SIZE, n, k, SNR_DB, G_for_encode)
            
            # Forward Pass 및 Loss 계산
            output_llrs = model(train_llrs)
            loss = criterion(output_llrs[:, :k], train_messages)
            
            # BER 계산 (Bit Error Rate)
            # LLRs를 Hard Decision (0 또는 1)으로 변환: LLR > 0 이면 0, LLR < 0 이면 1
            decoded_bits = (output_llrs[:, :k] < 0).float()
            errors = (decoded_bits != train_messages).sum().item()
            current_ber = errors / (BATCH_SIZE * k)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 데이터 수집
            current_loss_lst.append(loss.item())
            current_ber_lst.append(current_ber)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, BER: {current_ber:.6f}")
                
        print("--- 훈련 종료 ---")
        print(f"훈련된 {label} 가중치:", model.weights.data.squeeze().tolist())
        
        # 결과 저장
        loss_data.append(current_loss_lst)
        ber_data.append(current_ber_lst)


# --- 3. 결과 플롯 ---
date = np.array(range(1, EPOCHS + 1))
labels = [f"dv={dv[i]}, dc={dc[i]}" for i in range(len(dv))]

# 1. Loss 그래프
plt.figure(figsize=(10, 6))
for i, loss_lst in enumerate(loss_data):
    plt.plot(date, loss_lst, label=labels[i])
    
plt.xlabel("Epoch")
plt.ylabel("Loss (BCEWithLogits)")
plt.title("Training Loss vs. Epoch")
plt.legend()
plt.grid(True)
plt.show()

# 2. BER 그래프
plt.figure(figsize=(10, 6))
for i, ber_lst in enumerate(ber_data):
    plt.plot(date, ber_lst, label=labels[i])
    
plt.xlabel("Epoch")
plt.ylabel("Bit Error Rate (BER)")
plt.title("Bit Error Rate vs. Epoch")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()