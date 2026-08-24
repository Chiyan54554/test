# 本專案的 Kimi Delta Attention（KDA）

本專案實作的是用於因果語言模型的多頭 Kimi Delta Attention（KDA）。模型以 token 序列為輸入，逐 token 預測下一個 token，因此只能使用目前位置之前的資訊。

## KDA 層的處理方式

每個 KDA layer 會從 hidden state 產生 query、key、value、decay、輸出 gate 與 beta。query、key 與 value 會先通過因果 depthwise convolution；query 和 key 接著做 RMSNorm 與 RoPE 位置編碼。

KDA 維護每個 attention head 的 recurrent state。每一個 token 都先用 decay 衰減既有 state，再根據 key、value 與 beta 加入新的 delta 更新。最後用 query 從 state 讀出輸出，並乘上 SiLU gate 後通過輸出投影。

這種 recurrent state 讓模型不需要建立完整的 token 對 token attention matrix，因此適合長序列。正式 CUDA 訓練與推論會使用 `flash-linear-attention` 的 `chunk_kda` kernel；CPU 與測試時才保留 reference recurrence。

## 這個 32M 模型的設定

預設模型共有 32,167,716 個可訓練參數。它使用 8,192 個 BPE vocabulary、512 維 hidden size、9 層、4 個 attention heads、每個 head 128 維，以及 992 維 SwiGLU FFN。輸入 embedding 與輸出 language-model head 共用權重。

KDA head dimension 固定採用 128，這是為了相容 `chunk_kda` 與 FlashKDA 的最佳化 kernel。模型另外使用 Pre-RMSNorm、QK normalization、RoPE、SwiGLU、residual scaling 與 fused input projection。

## 使用限制

RAG 只會把這份參考資料放入 prompt，不會驗證模型輸出的每一個字。模型若無法依據資料可靠地重述內容，應縮短回答、降低 temperature，或改用更大的基座模型與 context-aware SFT 資料。
