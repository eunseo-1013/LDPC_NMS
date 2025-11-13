import torch
import torch.nn as nn
import numpy as np
import pyldpc
import matplotlib.pyplot as plt
import math

# --- LDPC 파라미터 ---
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

print(f"LDPC 코드 생성 완료: n={n}, k={k}, m={m}")

# =========================== #
#    Neural Min-Sum Decoder   #
# =========================== #

class NMSDecoder(nn.Module):
    def __init__(self, H_rows, H_cols, n_vars, n_checks, num_iterations=5):
        super(NMSDecoder, self).__init__()
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.num_iterations = num_iterations

        # 학습되는 scale 계수
        self.weights = nn.Parameter(torch.full((num_iterations, 1), 0.5))

        self.H_rows = H_rows
        self.H_cols = H_cols

    def forward(self, channel_llrs):
        batch_size = channel_llrs.shape[0]
        device = channel_llrs.device

        v2c_msgs = torch.zeros(batch_size, len(self.H_cols), device=device)
        c2v_msgs = torch.zeros(batch_size, len(self.H_cols), device=device)

        for it in range(self.num_iterations):
            # C→V 메시지 합산
            c2v_total = torch.zeros(batch_size, self.n_vars, device=device)
            c2v_total = c2v_total.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)

            # variable update
            var_llrs = channel_llrs + c2v_total
            v2c_msgs = var_llrs[:, self.H_cols] - c2v_msgs

            # check update
            new_c2v = torch.zeros_like(c2v_msgs)

            for c_idx in range(self.n_checks):
                edges = (self.H_rows == c_idx).nonzero().squeeze(-1)
                if edges.numel() == 0:
                    continue

                msgs = v2c_msgs[:, edges]
                signs = torch.sign(msgs)
                signs[signs == 0] = 1
                total_sign = torch.prod(signs, dim=1, keepdim=True)

                abs_vals = torch.abs(msgs)

                for j, e_idx in enumerate(edges):
                    others = abs_vals[:, [x for x in range(len(edges)) if x != j]]
                    if others.shape[1] == 0:
                        min_abs = torch.zeros(batch_size, device=device)
                    else:
                        min_abs = torch.min(others, dim=1).values

                    sign_out = total_sign / signs[:, j].unsqueeze(1)
                    new_c2v[:, e_idx] = sign_out.squeeze() * min_abs

            c2v_msgs = new_c2v * self.weights[it]

        # 최종 LLR
        final_total = torch.zeros(batch_size, self.n_vars, device=device)
        final_total = final_total.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
        final_llrs = channel_llrs + final_total

        return final_llrs


# =========================== #
#     Data generation         #
# =========================== #

def generate_data(batch_size, n_bits, k_bits, snr_db, G_matrix):

    messages_np = np.random.randint(0, 2, size=(batch_size, k_bits))

    if hasattr(G_matrix, "toarray"):
        Gd = G_matrix.toarray()
    else:
        Gd = G_matrix

    codewords_list = []
    for i in range(batch_size):
        cw = np.dot(Gd, messages_np[i]) % 2
        codewords_list.append(cw)

    codewords_np = np.vstack(codewords_list).astype(float)

    messages = torch.tensor(messages_np).float()
    codewords = torch.tensor(codewords_np).float()

    transmitted = 1 - 2 * codewords

    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1.0 / (2 * (k_bits / n_bits) * snr_linear)

    noise = torch.randn_like(transmitted) * np.sqrt(noise_var)
    received = transmitted + noise

    llrs = 2 * received / noise_var

    return llrs, messages


# =========================== #
#        Evaluation           #
# =========================== #

def evaluate_model(model, n_bits, k_bits, snr_db, G_matrix, num_test_frames, batch_size):
    model.eval()
    device = next(model.parameters()).device

    total_bit_err = 0
    total_frame_err = 0
    processed = 0

    batches = math.ceil(num_test_frames / batch_size)

    with torch.no_grad():
        for _ in range(batches):
            current = min(batch_size, num_test_frames - processed)
            if current <= 0:
                break

            llrs, msgs = generate_data(current, n_bits, k_bits, snr_db, G_matrix)
            llrs = llrs.to(device)
            msgs = msgs.to(device)

            out = model(llrs)

            # Hard decision
            preds = (out[:, :k_bits] < 0).float()

            total_bit_err += (preds != msgs).sum().item()
            total_frame_err += torch.any(preds != msgs, dim=1).sum().item()

            processed += current

    ber = total_bit_err / (processed * k_bits)
    fer = total_frame_err / processed
    return ber, fer

# =========================== #
#        Main Train           #
# =========================== #

if __name__ == "__main__":
    EPOCHS = 100
    BATCH_SIZE = 512
    LR = 0.001
    TRAIN_SNR = 4.0

    model = NMSDecoder(H_rows, H_cols, n_vars=n, n_checks=m, num_iterations=5)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print("--- Training Start ---")
    for epoch in range(EPOCHS):
        llrs, msgs = generate_data(BATCH_SIZE, n, k, TRAIN_SNR, G)
        out = model(llrs)

        # Loss 동일하게 통일!
        loss = criterion(-out[:, :k], msgs)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss={loss.item():.5f}")

    print("훈련된 scale:", model.weights.data.squeeze().tolist())

    # ----------------- 평가 -----------------
    SNR_LIST_TEST = [1.0, 2.0, 3.0, 4.0, 5.0]
    NUM_TEST_FRAMES = 100000

    ber_list = []
    fer_list = []

    print("\n--- Testing ---")
    for snr in SNR_LIST_TEST:
        ber, fer = evaluate_model(model, n, k, snr, G, NUM_TEST_FRAMES, BATCH_SIZE)
        print(f"SNR={snr:.1f} dB | BER={ber:.2e} | FER={fer:.2e}")
        ber_list.append(ber)
        fer_list.append(fer)

    plt.figure(figsize=(10,6))
    plt.semilogy(SNR_LIST_TEST, ber_list, 'bo-', label="BER")
    plt.semilogy(SNR_LIST_TEST, fer_list, 'rs--', label="FER")
    plt.grid(True, which="both")
    plt.legend()
    plt.xlabel("SNR(dB)")
    plt.ylabel("Error Rate")
    plt.title("NMS Performance (Comparison version)")
    plt.show()