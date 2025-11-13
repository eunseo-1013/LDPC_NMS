import torch
import torch.nn as nn
import numpy as np
import pyldpc
import matplotlib.pyplot as plt
import math

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

class NeuralNormalizedMSDecoder(nn.Module):
    def __init__(self, H_rows, H_cols, n_vars, n_checks, num_iterations=5):
        super(NeuralNormalizedMSDecoder, self).__init__()   # pytorch nn.Module 초기화 함수 상속
        self.n_vars = n_vars
        self.n_checks = n_checks
        self.num_iterations = num_iterations
        
        self.cn_scales = nn.Parameter(torch.zeros(num_iterations, 1))
        
        self.H_rows = H_rows
        self.H_cols = H_cols

    def forward(self, channel_llrs):
        batch_size = channel_llrs.shape[0]
        device = channel_llrs.device
        
        # 초기화
        v2c_msgs = torch.zeros(batch_size, len(self.H_cols), device=device)
        c2v_msgs = torch.zeros(batch_size, len(self.H_cols), device=device)

        for i in range(self.num_iterations):
            c2v_aggregated = torch.zeros(batch_size, self.n_vars, device=device)
            c2v_aggregated = c2v_aggregated.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
            
            # V-Node 업데이트
            var_llrs = channel_llrs + c2v_aggregated
            # V2C 메시지 계산
            v2c_msgs = var_llrs[:, self.H_cols] - c2v_msgs
            
            # C-Node 업데이트
            c2v_msgs_temp = torch.zeros_like(v2c_msgs)
            
            scale_val = torch.sigmoid(self.cn_scales[i])

            for c_idx in range(self.n_checks):
                connected_edges = (self.H_rows == c_idx).nonzero().squeeze(-1)
                if connected_edges.numel() == 0: continue
                
                incoming_msgs = v2c_msgs[:, connected_edges]
                if incoming_msgs.dim() == 1: incoming_msgs = incoming_msgs.unsqueeze(1)
                
                signs = torch.sign(incoming_msgs)
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                total_sign = torch.prod(signs, dim=1, keepdim=True)
                
                magnitude = torch.abs(incoming_msgs)

                for j, edge_idx in enumerate(connected_edges):
                    others = magnitude[:, [l for l in range(len(connected_edges)) if l != j]]
                    
                    if others.shape[1] == 0: 
                        min_magnitudes = torch.zeros(batch_size, device=device)
                    else: 
                        min_magnitudes = torch.min(others, dim=1).values
                        
                    # max(0, min - offset)
                    normalized_magnitude = scale_val * min_magnitudes
                    
                    sign_in = signs[:, j]
                    sign_in = torch.where(sign_in == 0, torch.ones_like(sign_in), sign_in)
                    out_sign = total_sign * sign_in.unsqueeze(1)
                    
                    # --- 정규화된 크기(magnitude)에 부호를 다시 곱함 ---
                    c2v_msgs_temp[:, edge_idx] = out_sign.squeeze(1) * normalized_magnitude
            
            c2v_msgs = c2v_msgs_temp

        # 최종 LLR 계산
        final_c2v_aggregated = torch.zeros(batch_size, self.n_vars, device=device)
        final_c2v_aggregated = final_c2v_aggregated.scatter_add(1, self.H_cols.expand(batch_size, -1), c2v_msgs)
        final_llrs = channel_llrs + final_c2v_aggregated
        
        return final_llrs


def generate_data(batch_size, n_bits, k_bits, snr_db, G_matrix):
    """훈련용 데이터 배치 생성 (실제 LDPC 부호화 적용)"""
    messages_np = np.random.randint(0, 2, size=(batch_size, k_bits))
    
    if hasattr(G_matrix, "toarray"):
        G_matrix_dense = G_matrix.toarray()
    else:
        G_matrix_dense = G_matrix
        
    codewords_list = []
    for i in range(batch_size):
        message = messages_np[i]
        codeword = np.dot(G_matrix_dense, message) % 2
        codewords_list.append(codeword)
    
    codewords_np = np.vstack(codewords_list).astype(float) 

    messages = torch.from_numpy(messages_np).float()
    codewords = torch.from_numpy(codewords_np).float()
    
    transmitted_signal = 1 - 2 * codewords
    snr_linear = 10 ** (snr_db / 10.0)
    
    code_rate = k_bits / n_bits
    noise_variance = 1.0 / (2 * code_rate * snr_linear)
    noise = torch.randn_like(transmitted_signal) * np.sqrt(noise_variance)
    
    received_signal = transmitted_signal + noise
    channel_llrs = 2 * received_signal / noise_variance
    
    return channel_llrs, messages

def evaluate_model(model, n_bits, k_bits, snr_db, G_matrix, num_test_frames, batch_size):
    """
    모델의 FER(프레임 오류율)과 BER(비트 오류율) 평가
    """
    model.eval()  # 평가 모드
    device = next(model.parameters()).device  # 모델이 사용하는 디바이스 확인

    total_bit_errors = 0
    total_frame_errors = 0
    total_frames_processed = 0
    
    # 배치 크기가 나누어 떨어지지 않아도 math.ceil을 사용해 정확한 프레임 수를 평가
    num_batches = math.ceil(num_test_frames / batch_size)
    
    with torch.no_grad():
        for _ in range(num_batches):
            current_batch_size = min(batch_size, num_test_frames - total_frames_processed)
            if current_batch_size <= 0:
                break
            
            # 테스트용 새 데이터 생성
            test_llrs, test_messages = generate_data(current_batch_size, n_bits, k_bits, snr_db, G)
            test_llrs = test_llrs.to(device)
            test_messages = test_messages.to(device)
            
            # 모델 추론
            output_llrs = model(test_llrs)
            
            # --- Hard Decision (경판정) ---
            # LLR 값이 0보다 크면 1, 작으면 0으로 결정 (정보 비트[0:k]에 대해서만)
            predicted_bits = (output_llrs[:, :k] < 0).float()
            
            # --- 오류 계산 ---
            # 1. 비트 오류 (BER)
            bit_errors = (predicted_bits != test_messages).sum().item()
            total_bit_errors += bit_errors
            
            # 2. 프레임 오류 (FER)
            # torch.any(..., dim=1)는 각 프레임(행)별로 오류가 하나라도 있는지 확인
            frame_errors = torch.any(predicted_bits != test_messages, dim=1).sum().item()
            total_frame_errors += frame_errors
            
            total_frames_processed += current_batch_size

    if total_frames_processed == 0:
        return 0.0, 0.0
    
    # 평균 BER/FER 계산
    ber = total_bit_errors / (total_frames_processed * k_bits)
    fer = total_frame_errors / total_frames_processed
    
    return ber, fer

if __name__ == '__main__':
    EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    SNR_DB = 4.0
    
    # --- NNMS 모델 생성 ---
    model = NeuralNormalizedMSDecoder(H_rows, H_cols, n_vars=n, n_checks=m, num_iterations=5)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- 훈련 시작 ---")
    for epoch in range(EPOCHS):
        model.train() # 훈련 모드
        train_llrs, train_messages = generate_data(BATCH_SIZE, n, k, SNR_DB, G)
        
        output_llrs = model(train_llrs)
        
        loss = criterion(-output_llrs[:, :k], train_messages) # 정보 비트에 대해서만 손실 계산
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    print("--- 훈련 종료 ---")
    model.eval() # 평가 모드
    
    print("학습된 정규화 계수(sigmoid(cn_scales)):", 
      torch.sigmoid(model.cn_scales.detach()).squeeze().tolist())
    
    # --- FER/BER 성능 평가 ---
    print("\n--- 성능 평가 (FER/BER) ---")
    
    # 평가할 SNR 범위 (dB)
    # 리스트 조절하여 워터폴 구간 찾기
    SNR_LIST_TEST = [1.0, 2.0, 3.0, 4.0, 5.0] # n=63, n=576
    # SNR_LIST_TEST = [2.0, 3.0, 4.0, 5.0, 6.0]   # n=1152, n=1728
    
    # 테스트할 프레임 수
    NUM_TEST_FRAMES = 100000    # 10e-3 ~ 10e-4 구간
    
    # 결과 저장 리스트
    results_ber = []
    results_fer = []
    
    for snr in SNR_LIST_TEST:
        ber, fer = evaluate_model(model, n, k, snr, G, NUM_TEST_FRAMES, BATCH_SIZE)
        print(f"SNR: {snr:.1f} dB  |  BER: {ber:.2e}  |  FER: {fer:.2e}")
    
        results_ber.append(ber)
        results_fer.append(fer)    
    
    # --- 그래프 그리기 ---
    print("\n--- 결과 그래프 생성 ---")
    
    plt.figure(figsize=(10, 6))
    # Y축을 로그 스케일로 설정 (semilogy)
    plt.semilogy(SNR_LIST_TEST, results_ber, 'bo-', label='BER (Bit Error Rate)')
    plt.semilogy(SNR_LIST_TEST, results_fer, 'rs--', label='FER (Frame Error Rate)')
    
    plt.title(f'NNMS Decoder Performance (n={n}, k={k}, dv={d_v}, dc={d_c})')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate (Log Scale)')
    plt.legend() 
    plt.grid(True, which="both", ls="--") 
    
    # Y축 최소값 설정 (0이 아닌 작은 값, 예: 10e-6)
    if k > 0: # k가 0보다 클 때만 (즉, 코드가 유효할 때만)
        min_error_rate = 1 / (NUM_TEST_FRAMES * k) # 관측 가능한 최소 BER
        plt.ylim(bottom=max(min_error_rate / 10, 1e-7)) # 최소 BER보다 조금 낮게 설정
    else:
        plt.ylim(bottom=1e-7)
    
    plt.show()