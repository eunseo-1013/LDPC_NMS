import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pyldpc
import matplotlib.pyplot as plt

# LDPC 파라미터 설정
N_BITS_LIST = [576, 1152, 1728]
dv = 2  # 변수 노드 차수 (dv) # 한 열에 있는 1의 개수
dc = 3  # 체크 노드 차수 (dc) # 한 행에 있는 1의 개수

LEARNING_RATE = 0.005
EPOCHS = 50
BATCH_SIZE = 1000
SNR_DB = 4.0  # 신호 대 잡음비
NUM_ITERATIONS = 5  # 디코더 반복 횟수

# 모든 N에 대한 결과 저장을 위한 딕셔너리
all_results = {}


class NOMSDecoder(nn.Module):
    def __init__(self, H_rows, H_cols, n_vars, n_checks, num_iterations=5):
        super(NOMSDecoder, self).__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.num_iterations = num_iterations
        self.weights = nn.Parameter(torch.full((num_iterations, 1), 0.7))
        self.biases = nn.Parameter(torch.full((num_iterations, 1), 0.1))
        self.H_rows = H_rows
        self.H_cols = H_cols

    def decoding(self, channel_llrs):
        batch_size = channel_llrs.shape[0]
        device = channel_llrs.device
        v2c_msgs = torch.zeros(batch_size, len(self.H_cols), device=device)
        c2v_msgs = torch.zeros(batch_size, len(self.H_cols), device=device)

        for i in range(self.num_iterations):
            # 1. Variable → Check
            c2v_aggregated = torch.zeros(batch_size, self.n_vars, device=device)
            c2v_aggregated.scatter_add_(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
            var_llrs = channel_llrs + c2v_aggregated
            v2c_msgs = var_llrs[:, self.H_cols] - c2v_msgs

            # 2. Check → Variable (NOMS 로직 적용)
            c2v_msgs_temp = torch.zeros_like(v2c_msgs)
            for c_idx in range(self.n_checks):
                connected_edges = (self.H_rows == c_idx).nonzero(as_tuple=False).squeeze(-1)
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
                        min_abs_vals = torch.zeros(batch_size, device=device)
                    else:
                        min_abs_vals = torch.min(other_msgs_abs, dim=1).values

                    sign_in = torch.sign(incoming_msgs[:, j])
                    sign_in[sign_in == 0] = 1
                    out_sign = signs.squeeze(1) / sign_in

                    normalized_val = min_abs_vals * self.weights[i]
                    offset_val = normalized_val - self.biases[i]
                    clamped_val = torch.clamp(offset_val, min=0)

                    c2v_msgs_temp[:, edge_idx] = out_sign * clamped_val

            c2v_msgs = c2v_msgs_temp

        # 최종 LLR 계산
        final_c2v_aggregated = torch.zeros(batch_size, self.n_vars, device=device)
        final_c2v_aggregated.scatter_add_(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
        final_llrs = channel_llrs + final_c2v_aggregated

        return final_llrs


## 수정된 데이터 생성 함수
def generate_data(batch_size, n_bits, k_bits, snr_db, G_matrix, device):
    messages_np = np.random.randint(0, 2, size=(batch_size, k_bits))  # CPU에서 NumPy로 생성

    if hasattr(G_matrix, "toarray"):
        G_matrix = G_matrix.toarray()  # G_matrix는 G (k_bits x n_bits)

    codewords_np = (messages_np @ G_matrix) % 2
    messages = torch.from_numpy(messages_np).float().to(device)  # PyTorch 텐서로 변환 후 지정된 디바이스로 이동
    codewords = torch.from_numpy(codewords_np).float().to(device)

    transmitted_signal = 1 - 2 * codewords  # (0, 1) -> (+1, -1) BPSK
    snr_linear = 10 ** (snr_db / 10.0)
    code_rate = k_bits / n_bits
    noise_variance = 1.0 / (2 * code_rate * snr_linear + 1e-9)

    # 잡음 추가
    noise = torch.randn_like(transmitted_signal) * np.sqrt(noise_variance)
    received_signal = transmitted_signal + noise
    channel_llrs = 2 * received_signal / noise_variance

    return channel_llrs, messages


# 2. 훈련 루프
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for n_bits_target in N_BITS_LIST:
        loss_data = []
        ber_data = []
        fer_data = []

        # LDPC 코드 생성
        try:
            H, G = pyldpc.make_ldpc(n_bits_target, dv, dc, systematic=True, sparse=True)
        except Exception as e:
            print(f"N={n_bits_target}, dv={dv}, dc={dc} 코드 생성 실패: {e}")
            continue  # 다음 N_BITS로 넘어감

        if G.shape[0] > G.shape[1]:
            G = G.T

        k_bits = G.shape[0]
        n_bits_actual = G.shape[1]
        m_bits = H.shape[0]

        if n_bits_actual != n_bits_target:
            print(f"요청한 N : {n_bits_target}과 실제 생성된 N : {n_bits_actual}이 다릅니다.")

        print(f"\nLDPC 코드 파라미터 (dv={dv}, dc={dc})")
        print(f"N = {n_bits_actual}")
        print(f"K = {k_bits}")
        print(f"M = {m_bits}")

        # 인덱스
        H_rows, H_cols = H.nonzero()
        H_rows = torch.tensor(H_rows, dtype=torch.long).to(device)
        H_cols = torch.tensor(H_cols, dtype=torch.long).to(device)

        G_for_encode = G
        if hasattr(G_for_encode, "toarray"):
            G_for_encode = G_for_encode.toarray()

        G_enc_np = G_for_encode  # Systematic 위치 계산
        G_T_np = G_enc_np.T
        sys_col_indices = [-1] * k_bits

        for j_col in range(n_bits_actual):
            col_data = G_T_np[j_col, :]
            ones_indices = np.where(col_data == 1)[0]

            if len(ones_indices) == 1:
                msg_bit_index = ones_indices[0]
                if sys_col_indices[msg_bit_index] == -1:
                    sys_col_indices[msg_bit_index] = j_col

        missing_indices = [i for i, val in enumerate(sys_col_indices) if val == -1]
        if missing_indices:
            print(f"{len(missing_indices)}개의 systematic 위치를 찾지 못했습니다.")
            sys_positions = np.array(list(range(k_bits)), dtype=int)
        else:
            sys_positions = np.array(sys_col_indices, dtype=int)

        # N_BITS 마다 새 모델과 옵티마이저 생성
        model = NOMSDecoder(H_rows, H_cols, n_vars=n_bits_actual, n_checks=m_bits, num_iterations=NUM_ITERATIONS).to(
            device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print(f"\n훈련 시작 (N = {n_bits_actual})")
        for epoch in range(EPOCHS):
            train_llrs, train_messages = generate_data(BATCH_SIZE, n_bits_actual, k_bits, SNR_DB, G_for_encode, device)
            output_llrs = model.decoding(train_llrs)

            logits_for_loss = -output_llrs[:, sys_positions]  # LLR은 target=0일 때 값이 커지므로(0일 확률이 높음), -LLR을 logit으로 사용
            loss = criterion(logits_for_loss, train_messages)

            decoded_codeword_bits = (output_llrs < 0).float()
            decoded_message_bits = decoded_codeword_bits[:, sys_positions]
            total_bit_errors = (decoded_message_bits != train_messages).sum().item()
            current_ber = total_bit_errors / (BATCH_SIZE * k_bits)

            frame_has_errors = torch.any(decoded_message_bits != train_messages, dim=1)
            total_frame_errors = frame_has_errors.sum().item()
            current_fer = total_frame_errors / BATCH_SIZE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.weights.clamp_(min=0)
                model.biases.clamp_(min=0)

            loss_data.append(loss.item())
            ber_data.append(current_ber)
            fer_data.append(current_fer)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], N = {n_bits_actual}, Loss : {loss.item():.6f}, BER : {current_ber:.6f}, FER : {current_fer:.6f}")

        print(f"훈련된 가중치 (Alpha) :", model.weights.data.squeeze().tolist())
        print(f"훈련된 오프셋 (Beta) :", model.biases.data.squeeze().tolist())

        # 훈련 결과를 딕셔너리에 저장
        all_results[n_bits_actual] = {
            'loss': loss_data,
            'ber': ber_data,
            'fer': fer_data,
            'k_bits': k_bits
        }

    # 3. 결과 플롯 (모든 N_BITS 루프가 끝난 후)
    date = np.array(range(1, EPOCHS + 1))

    # --- Loss Plot ---
    plt.figure(figsize=(10, 6))
    for n_bits, data in all_results.items():
        label = f"N={n_bits}, K={data['k_bits']}, dv={dv}, dc={dc}"
        plt.plot(date, data['loss'], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCEWithLogits)")
    plt.title(f"Training Loss (SNR={SNR_DB})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- BER Plot ---
    plt.figure(figsize=(10, 6))
    for n_bits, data in all_results.items():
        label = f"N={n_bits}, K={data['k_bits']}, dv={dv}, dc={dc}"
        plt.plot(date, data['ber'], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"Training BER vs. Epoch (SNR={SNR_DB})")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.yscale('log')
    plt.show()

    # --- FER Plot ---
    plt.figure(figsize=(10, 6))
    for n_bits, data in all_results.items():
        label = f"N={n_bits}, K={data['k_bits']}, dv={dv}, dc={dc}"
        plt.plot(date, data['fer'], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Frame Error Rate (FER)")
    plt.title(f"Training FER vs. Epoch (SNR={SNR_DB})")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.yscale('log')
    plt.show()