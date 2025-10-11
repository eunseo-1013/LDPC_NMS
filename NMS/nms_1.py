import torch
import torch.nn as nn
import numpy as np
import pyldpc

# --- 1. LDPC 파라미터 및 H 행렬 생성 (pyldpc 사용) ---
n_in = 63
d_v = 3
d_c = 7

H, G = pyldpc.make_ldpc(n_in, d_v, d_c, systematic=True, sparse=True)

k = G.shape[1]
n = G.shape[0]
m = H.shape[0]

H_rows, H_cols = H.nonzero()
H_rows = torch.tensor(H_rows, dtype=torch.long)
H_cols = torch.tensor(H_cols, dtype=torch.long)

print(f"LDPC 코드 생성 완료: n={n}, k={k}, m={m} (실제 행렬 기준)")


class NMSDecoder(nn.Module):
    """Neural Min-Sum (NMS) 디코더 PyTorch 모델"""
    def __init__(self, H_rows, H_cols, n_vars, n_checks, num_iterations=5):
        super(NMSDecoder, self).__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.num_iterations = num_iterations
        self.weights = nn.Parameter(torch.full((num_iterations, 1), 0.5))
        self.H_rows = H_rows
        self.H_cols = H_cols

    def forward(self, channel_llrs):
        batch_size = channel_llrs.shape[0]
        v2c_msgs = torch.zeros(batch_size, len(self.H_cols))
        c2v_msgs = torch.zeros(batch_size, len(self.H_cols))
        for i in range(self.num_iterations):
            c2v_aggregated = torch.zeros(batch_size, self.n_vars)
            c2v_aggregated = c2v_aggregated.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
            var_llrs = channel_llrs + c2v_aggregated
            v2c_msgs = var_llrs[:, self.H_cols] - c2v_msgs
            c2v_msgs_temp = torch.zeros_like(v2c_msgs)
            for c_idx in range(self.n_checks):
                connected_edges = (self.H_rows == c_idx).nonzero().squeeze(-1)
                if connected_edges.numel() == 0: continue
                incoming_msgs = v2c_msgs[:, connected_edges]
                if incoming_msgs.dim() == 1: incoming_msgs = incoming_msgs.unsqueeze(1)
                signs = torch.prod(torch.sign(incoming_msgs), dim=1, keepdim=True)
                abs_vals = torch.abs(incoming_msgs)
                for j, edge_idx in enumerate(connected_edges):
                    other_msgs_abs = abs_vals[:, [l for l in range(len(connected_edges)) if l != j]]
                    if other_msgs_abs.shape[1] == 0: min_abs_vals = torch.zeros(batch_size)
                    else: min_abs_vals = torch.min(other_msgs_abs, dim=1).values
                    sign_in = torch.sign(incoming_msgs[:, j])
                    sign_in[sign_in == 0] = 1 
                    out_sign = signs / sign_in.unsqueeze(1)
                    c2v_msgs_temp[:, edge_idx] = out_sign.squeeze() * min_abs_vals
            c2v_msgs = c2v_msgs_temp * self.weights[i]
        final_c2v_aggregated = torch.zeros(batch_size, self.n_vars)
        final_c2v_aggregated = final_c2v_aggregated.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
        final_llrs = channel_llrs + final_c2v_aggregated
        return final_llrs


def generate_data(batch_size, n_bits, k_bits, snr_db, G_matrix):
    """훈련용 데이터 배치 생성 (실제 LDPC 부호화 적용)"""
    messages_np = np.random.randint(0, 2, size=(batch_size, k_bits))
    
    if hasattr(G_matrix, "toarray"):
        G_matrix = G_matrix.toarray()
        
    # pyldpc는 한 번에 한 메시지만 처리 가능하므로, for 루프를 사용해 하나씩 처리
    codewords_list = []
    for i in range(batch_size):
        message = messages_np[i] # 메시지를 하나씩 꺼냄
        codeword = pyldpc.encode(G_matrix, message, snr_db)
        codewords_list.append(codeword)
    
    codewords_np = np.vstack(codewords_list) # 처리된 결과들을 다시 하나로 합침

    messages = torch.from_numpy(messages_np).float()
    codewords = torch.from_numpy(codewords_np).float()
    transmitted_signal = 1 - 2 * codewords
    snr_linear = 10 ** (snr_db / 10.0)
    noise_variance = 1.0 / (2 * (k_bits / n_bits) * snr_linear)
    noise = torch.randn_like(transmitted_signal) * np.sqrt(noise_variance)
    received_signal = transmitted_signal + noise
    channel_llrs = 2 * received_signal / noise_variance
    return channel_llrs, messages


if __name__ == '__main__':
    EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    SNR_DB = 4.0
    model = NMSDecoder(H_rows, H_cols, n_vars=n, n_checks=m, num_iterations=5)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("--- 훈련 시작 ---")
    for epoch in range(EPOCHS):
        train_llrs, train_messages = generate_data(BATCH_SIZE, n, k, SNR_DB, G)
        output_llrs = model(train_llrs)
        loss = criterion(output_llrs[:, :k], train_messages)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")
    print("--- 훈련 종료 ---")
    print("훈련된 가중치:", model.weights.data.squeeze().tolist())