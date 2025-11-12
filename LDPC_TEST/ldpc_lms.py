import torch
import torch.nn as nn
import numpy as np
import pyldpc
import matplotlib.pyplot as plt

# --- 1. LDPC 파라미터 및 H 행렬 생성 (pyldpc 사용) ---
n_in = 63  # 전체 부호어 길이 (N)
dv = [2,2,4,6]   # 열 가중치
dc = [3,7,7,9]   # 행 가중치
lr = [0.005]



# 결과 저장 리스트
loss_data = []
ber_data = []


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
def generate_data(batch_size, n_bits, k_bits, snr_db, G_matrix):
    messages_np = np.random.randint(0, 2, size=(batch_size, k_bits))
    if hasattr(G_matrix, "toarray"):
        G_matrix = G_matrix.toarray()

    codewords_list = []
    for i in range(batch_size):
        message = messages_np[i]
        codeword = pyldpc.encode(G_matrix, message, snr_db)
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


# --- 2. 훈련 루프 ---
if __name__ == '__main__':
    EPOCHS = 50
    BATCH_SIZE = 1000
    SNR_DB = 4.0

    for j in range(len(lr)):
        LEARNING_RATE = lr[j]

        for i in range(len(dv)):
            d_v = dv[i]
            d_c = dc[i]
            label = f"dv={d_v}, dc={d_c}"

            # LDPC 코드 생성
            H, G = pyldpc.make_ldpc(n_in, d_v, d_c, systematic=True, sparse=True)
            if G.shape[0] > G.shape[1]:
                G = G.T

            k = G.shape[0]
            n = G.shape[1]
            m = H.shape[0]

            print(f"\n--- {label} 코드 ---")
            print(f"LDPC 코드 생성 완료: n={n}, k={k}, m={m} (G.shape={G.shape})")

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
            print("Systematic positions:", sys_positions)

            # 모델 및 최적화 설정
            model = NMSDecoder(H_rows, H_cols, n_vars=n, n_checks=m, num_iterations=5)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            current_loss_lst = []
            current_ber_lst = []

            print("--- 훈련 시작 ---")
            for epoch in range(EPOCHS):
                train_llrs, train_messages = generate_data(BATCH_SIZE, n, k, SNR_DB, G_for_encode)

                output_llrs = model(train_llrs)

                # LLR 부호 반전 후 loss 계산
                logits_for_1 = -output_llrs[:, sys_positions]
                loss = criterion(logits_for_1, train_messages)

                # Hard decision (LLR < 0 → 1)
                decoded_codeword_bits = (output_llrs < 0).float()
                decoded_message_bits = decoded_codeword_bits[:, sys_positions]

                errors = (decoded_message_bits != train_messages).sum().item()
                current_ber = errors / (BATCH_SIZE * k)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_loss_lst.append(loss.item())
                current_ber_lst.append(current_ber)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}, BER: {current_ber:.6f}")

            print("--- 훈련 종료 ---")
            print(f"훈련된 {label} 가중치:", model.weights.data.squeeze().tolist())

            loss_data.append(current_loss_lst)
            ber_data.append(current_ber_lst)

    # --- 3. 결과 플롯 ---
    date = np.array(range(1, EPOCHS + 1))
    labels = [f"dc= {dc[i]} ,dv ={dv[i]} " for i in range(len(dv))]

    plt.figure(figsize=(10, 6))
    for i, loss_lst in enumerate(loss_data):
        plt.plot(date, loss_lst, label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCEWithLogits)")
    plt.title("Training Loss per learning rate")
    legends = [f"dv={dv[i]}, dc={dc[i]}, SNR={SNR_DB}" for i in range(len(dv))]
    plt.legend(legends)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for i, ber_lst in enumerate(ber_data):
        plt.plot(date, ber_lst, label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("Bit Error Rate vs. Epoch per learning rate")
   # legends = [f"learning rate={lr[i]}, SNR={SNR_DB}" for i in range(len(lr))]
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
