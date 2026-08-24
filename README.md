# 32M 中文 Kimi Delta Attention 模型

此專案以 **Linux + NVIDIA CUDA** 作為正式訓練與推論環境。正式執行必須使用 `flash-linear-attention` 的 `chunk_kda`；reference recurrence 僅保留給單元測試與開發除錯。

## Linux 部署前檢查

在 Linux GPU 容器或主機內，先確認 NVIDIA runtime，再安裝依賴並驗證 KDA kernel：

```bash
nvidia-smi
uv sync --extra cuda --extra data --extra traditional
uv run kda-doctor
```

`kda-doctor` 必須輸出 GPU 名稱與 `kda backend`。若失敗，先修正 GPU passthrough、PyTorch CUDA wheel 或 FLA，再開始訓練。

此專案採 v2 模組化架構。`src/kda_llm/model.py` 保留作為舊程式的相容入口；新開發請從 `models/`、`training/`、`inference/` 與 `workflows/` 匯入。模型有 32,167,716 個可訓練參數，設定為 8,192 個 BPE 詞彙、512 維 hidden size、9 層與 4 個 128 維 attention heads；輸入與輸出 token embedding 共用權重。

模型包含現代 LLM 元件：Pre-RMSNorm、RoPE、QK Normalization、用於局部上下文的因果 depthwise convolution、KDA 輸出 gate、SwiGLU、權重共用與 residual scaling。

```text
kda_llm/
  models/     # config、layers、KDA attention、language model、CUDA kernels
  training/   # memmap input、token scheduler、checkpoint adapter、profiler、engine
  inference/  # checkpoint loading、sampling、cache-aware generation
  workflows/  # 下載到訓練的完整 pipeline
  cli/        # 穩定的 kda-* 指令入口
```

新版 checkpoint 會標記 `format_version=2`；載入沒有版本欄位的舊 checkpoint 時，會自動轉換舊 KDA projection 權重。舊 projection checkpoint 續訓時會安全重建 AdamW state，避免不正確對應 optimizer moment。

## Chunkwise KDA Kernel

在 Linux CUDA 環境安裝分塊 KDA backend：

```powershell
uv sync --extra cuda
```

模型會自動呼叫 `flash-linear-attention` 的 `chunk_kda` kernel。若使用 NVIDIA SM90 以上架構與 CUDA 12.9 以上版本，還可安裝 Moonshot 的 FlashKDA backend，FLA 會自動轉送至更快的實作。正式 CUDA 訓練會要求 kernel 可用，避免靜默退回 reference recurrence。

FlashKDA 需要 128 維 KDA heads，因此模型採用 4 個 128 維 head，而不是 8 個 64 維 head。

安裝基本相依套件並執行內建的前向傳播測試：

```powershell
uv sync
uv run kda-self-test
```

## 中文 Tokenizer

## 設定檔

可調整的模型架構放在 [configs/model_32m.json](configs/model_32m.json)，訓練超參數放在 [configs/train_gpu.json](configs/train_gpu.json)。複製後改名建立實驗設定；CLI 同名選項會覆寫 JSON 值。`model_32m.json` 的 `vocab_size` 會由 pipeline 自動傳給 tokenizer，必須與 checkpoint 使用的 tokenizer 相同。

正式訓練用 `max_tokens` 決定總訓練量，steps 會由 `batch_size × grad_accum × seq_len` 自動推導。因此調整 batch 或序列長度時，總 token budget 不會改變。`--steps` 僅保留給短暫 smoke test。

`train_gpu.json` 以 16 GB 級 GPU 的保守起點 `batch_size=160`、`seq_len=256` 設定。若有更多可用 VRAM，再逐步提高 micro-batch；出現 CUDA OOM 時先降低 `batch_size`，不要嘗試用 allocator 設定掩蓋真實顯存不足。

```powershell
uv run kda-train --model-config configs/model_32m.json --train-config configs/train_gpu.json --train-data data/train.bin --val-data data/valid.bin
```

準備一或多個 UTF-8 編碼的中文預訓練 `.txt` 語料後，建立 8k BPE tokenizer：

```powershell
uv run kda-build-tokenizer --input data/chinese_corpus.txt --output tokenizer/chinese
```

此 tokenizer 以子詞編碼壓縮詞彙：常見中文字、詞語與標點會合併成較短的 token；罕見字則回退為可逆的 UTF-8 byte token。因此不用在每個中文字之間插入空格。

指令會產生 `tokenizer/chinese.model` 與 `tokenizer/chinese.vocab`。模型必須使用這個 tokenizer 所編碼的中文語料預訓練後，才會具備中文能力。

### 轉換為繁體中文

若來源包含簡體中文，請在建立 tokenizer 前先轉換為台灣繁體。此步驟會逐行串流處理，且不會覆蓋原始語料：

```powershell
uv sync --extra traditional
uv run kda-convert-traditional --input data/train.txt --output data/train_zh_hant.txt
uv run kda-convert-traditional --input data/valid.txt --output data/valid_zh_hant.txt
```

預設的 OpenCC `s2twp` 模式會進行簡體轉台灣正體及常見慣用詞轉換。接下來以 `train_zh_hant.txt` 建立 tokenizer，並使用轉換後的 train/valid 檔案進行資料編碼。

### 資料清理與去重

在訓練 tokenizer 前，先清理繁體語料。清理器採串流模式，不會把語料全部讀入記憶體；它會正規化 Unicode 與空白、移除控制字元、過濾過短或過長文件、過濾中文比例過低與網址比例過高的內容、排除重複字元垃圾資料，並以 SQLite 暫存索引做精確去重。完成後會產生可追蹤的 JSON 統計報告。

```powershell
uv run kda-clean-corpus --input data/train_zh_hant.txt --output data/train_clean.txt --min-chars 20 --min-cjk-ratio 0.15
```

預設適合中文通用預訓練。技術、英文或程式碼比例較高的來源，可降低 `--min-cjk-ratio`。目前只做精確去重；近似去重、個資偵測與更嚴格的品質分類，應在正式大規模訓練前補上。

### 從 Hugging Face 下載語料

安裝選用的 dataset client，接著從任一公開或已授權的 Hugging Face dataset 串流下載指定文字欄位。下載前請閱讀資料集卡，確認授權、設定名稱與文字欄位名稱。

```powershell
uv sync --extra data
uv run kda-download-hf --dataset ORG/DATASET --config CONFIG --split train --text-column text --output data/train.txt
```

`data` extra 也包含 `zstandard`，可讀取部分 Hugging Face 資料集使用的 `.zst` 壓縮檔。

下載器使用 Hugging Face 串流模式，不會先下載完整資料集。若只需要部分語料，在命令最後加上 `--limit 10000`，達到 10,000 份有效文字文件後就會停止。若資料集需要授權，先登入：

```powershell
uv run hf auth login
```

### 混合多個資料集

使用 JSON manifest 可讓每個資料集獨立指定設定、split、文字欄位與比例 `weight`。範例格式請見 [configs/hf_sources.example.json](configs/hf_sources.example.json)。將其中的 placeholder 改成實際資料集後執行：

```powershell
uv run kda-download-hf --sources configs/hf_sources.json --total-documents 1000000 --output data/train.txt
```

各來源會依 manifest 順序串流，並依 `weight` 分配文件名額。例如 60% 與 40% 的兩個來源搭配 `--total-documents 1000000`，分別只抓 600,000 與 400,000 份有效文件。達標即停止，不會下載其餘資料集內容。

下載、繁體轉換與 token 編碼每 1,000 份文件或行數會顯示進度；若需要調整頻率，加入 `--progress-every 100`。

## 資料編碼與訓練

將清理後的中文語料分成訓練集與驗證集，並編碼為緊湊的 `uint16` token stream：

```powershell
uv run kda-prepare-data --tokenizer tokenizer/chinese.model --input data/train.txt --output data/train.bin
uv run kda-prepare-data --tokenizer tokenizer/chinese.model --input data/valid.txt --output data/valid.bin
```

先執行短暫的 smoke run。有效 batch size 為 `batch-size * grad-accum` 個序列；預設為 64 個、每個序列 256 tokens。

```powershell
uv run kda-train --train-data data/train.bin --val-data data/valid.bin --steps 100 --out-dir checkpoints/smoke
```

在正式 CUDA 環境可先用相同的短測試加上 `--compile`，比較訓練 log 的 tokens/sec；若 Triton kernel 出現 graph break 或沒有更快，使用 `--no-compile` 回退即可。訓練器會以背景執行緒預取下一個 batch，將 pinned host memory 非同步傳入 GPU，並在 CUDA 訓練時要求 `chunk_kda` 可用，避免靜默退回較慢的 reference recurrence。

### 設定多資料集訓練比例

若要精確控制每個資料集的訓練比例，請將每個來源分別編碼成 `.bin` 檔，再使用 weighted source manifest。範例請見 [configs/train_sources.example.json](configs/train_sources.example.json)。`weight` 不必加總為 1，程式會自動正規化；每個 micro-batch 會依權重選取一個來源，因此長期 token 比例會接近設定值。

```powershell
uv run kda-train --train-sources configs/train_sources.json --val-data data/valid.bin --steps 100 --out-dir checkpoints/smoke
```

例如範例設定的比例為知識資料 40%、書籍 30%、問答 20%、程式碼 10%。

確認訓練 loss 能穩定下降、沒有出現 `NaN` 後，再增加 `--steps` 進行正式訓練。腳本使用 next-token prediction，定期儲存模型與 optimizer state；若 CUDA GPU 支援 bf16，會自動以 bfloat16 訓練。

開發時可使用 reference fallback；正式長 context 訓練建議在 CUDA 環境安裝 chunkwise KDA backend。

### 生成文字

訓練完成後，使用同一份 tokenizer 與 checkpoint 產生延續文字：

```powershell
uv run kda-generate --checkpoint runs/smoke/checkpoints/kda-step-100.pt --tokenizer runs/smoke/tokenizer/chinese.model --prompt "人工智慧的未來" --max-new-tokens 128 --temperature 0.8 --top-p 0.95
```

預設會自動選擇 CUDA；可用 `--device cpu` 除錯。生成器會檢查 checkpoint 與 tokenizer 詞彙大小是否相符，避免錯用 tokenizer。`--top-k 0` 可停用 top-k 過濾。

## 監督式微調（SFT）

預訓練讓模型學會中文續寫；SFT 讓它學會依指令回答。流程會只對 assistant 回覆計算 loss，使用者訊息和格式 token 會被 mask，不會把「重複 prompt」當成學習目標。SFT 必須沿用預訓練時的 tokenizer 與 checkpoint。

預設的 [configs/sft_sources.json](configs/sft_sources.json) 使用 [alpaca-data-gpt4-chinese-zhtw](https://huggingface.co/datasets/erhwenkuo/alpaca-data-gpt4-chinese-zhtw) 的 40,000 筆台灣繁中資料，採串流下載並在達到額度後停止：

```powershell
uv sync --extra cuda --extra data --extra traditional
uv run kda-download-sft --sources configs/sft_sources.json --output runs/sft/raw.jsonl --limit 40000
uv run kda-prepare-sft --input runs/sft/raw.jsonl --tokenizer runs/smoke/tokenizer/chinese.model --output runs/sft/sft.pt --max-length 512
uv run kda-sft --checkpoint runs/smoke/checkpoints/kda-step-30000.pt --dataset runs/sft/sft.pt --out-dir runs/sft/checkpoints --device cuda --compile
```

`kda-sft` 預設跑 2 epochs、learning rate `8e-5`，每個 epoch 都會輸出可直接供 `kda-generate` 使用的 v2 checkpoint。將 `kda-step-30000.pt` 改成實際預訓練完成的 checkpoint 路徑即可。訓練資料與模型均沿用舊權重載入邏輯，因此也能以舊版 checkpoint 開始 SFT。

若已取得 [PromptPair-TW](https://huggingface.co/datasets/liswei/PromptPair-TW) 的存取權並接受其 `CC BY-NC-SA 4.0` 條款，可改用 [configs/sft_sources.example.json](configs/sft_sources.example.json) 的第二個來源；請依資料集頁面條款確認商業使用與衍生資料的限制。每個來源可用 `limit` 控制數量，工具會依內容雜湊去重。

## 一鍵流程

若不想手動執行每個步驟，安裝所有需要的選用套件後可直接執行完整流程。它會依序下載限量資料、轉台灣繁體、清理與精確去重、切分 train/valid、訓練 tokenizer、編碼 token stream，最後執行訓練。

```powershell
uv sync --extra cuda --extra data --extra traditional
uv run kda-pipeline --total-documents 100000 --progress-every 1000 --steps 100 --work-dir runs/smoke --device cuda --compile
```

所有產物會放在 `runs/smoke/`，原始下載、繁體語料、tokenizer、token stream 與 checkpoint 不會散落在專案根目錄。專案會從 PyTorch 的 CUDA 12.8 wheel index 安裝 `torch`；`--device cuda` 會在 CUDA 無法使用時立即停止，而不會悄悄退回 CPU。先以預設的 100,000 份文件和 100 steps 確認流程；成功後再提高 `--total-documents` 與 `--steps`。

同步後先驗證 CUDA：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

若下載、轉檔或編碼中斷，使用相同設定加上 `--resume` 即可重用已完整完成的資料階段，只有未完成階段會重新執行。輸出會先寫入 `.partial` 暫存檔，成功後才原子更名為最終檔案。

```powershell
uv run kda-pipeline --total-documents 100000 --steps 100 --work-dir runs/smoke --device cuda --resume
```
