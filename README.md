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
uv run --extra cuda --extra data kda-sft-pipeline --checkpoint runs/smoke/checkpoints/kda-step-30000.pt --tokenizer runs/smoke/tokenizer/chinese.model --device cuda
```

此指令會依序下載、編碼與訓練；預設跑 2 epochs、learning rate `8e-5`，最終 checkpoint 會在 `runs/sft/checkpoints/kda-sft-epoch-2.pt`，可直接供 `kda-generate` 使用。下載或編碼中斷後，用相同指令重跑即可重用已完成階段；使用 `--no-resume` 可強制重做。將 `kda-step-30000.pt` 改成實際預訓練完成的 checkpoint 路徑即可。訓練資料與模型均沿用舊權重載入邏輯，因此也能以舊版 checkpoint 開始 SFT。

顯存不足時，以較小的 `--batch-size` 搭配 `--grad-accum` 維持等效 batch。例如 15.5 GiB GPU 訓練 64M 模型、長度 512 時，建議使用 `--batch-size 8 --grad-accum 4`，等效 batch 仍為 32，但峰值顯存大幅降低。

SFT 訓練必須從預訓練 checkpoint 開始，不要將中斷的 SFT checkpoint 當作新的 base checkpoint。正常情況下，loss 會逐步下降但不應在數百 step 內趨近於 `0.0000`；若發生，先停止訓練並確認使用的是最新版程式。

SFT checkpoint 生成時必須帶 `--chat`，讓推論 prompt 與訓練對話模板一致；重複輸出時可提高 `--repetition-penalty`：

```powershell
uv run kda-generate --checkpoint runs/sft/checkpoints/kda-sft-epoch-2.pt --tokenizer runs/smoke/tokenizer/chinese.model --prompt "請用繁體中文介紹 KDA。" --chat --system "請清楚、簡潔地回答問題。" --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.1 --device cuda
```

## RAG 知識庫

RAG 會在生成前從本地技術文件找出相關段落，將其作為帶來源的參考資料放進對話 prompt，適合補足 32M 模型不熟悉的 KDA、專案或領域知識。專案已附一份 [knowledge/kda.md](knowledge/kda.md) 作為可直接測試的起始知識庫；也可將論文摘要、設計文件或其他 Markdown/UTF-8 文字檔放入同一資料夾後建立索引：

```powershell
uv run kda-build-rag --input knowledge --output runs/rag/knowledge.json
```

生成時帶入索引即可；RAG 會自動啟用對話模板，並要求模型在資料不足時回答不知道。`--show-sources` 會列出實際檢索的文件，便於檢查答案依據。

```powershell
uv run kda-generate --checkpoint runs/sft/checkpoints/kda-sft-epoch-2.pt --tokenizer runs/smoke/tokenizer/chinese.model --prompt "什麼是 Kimi Delta Attention？" --rag-index runs/rag/knowledge.json --show-sources --temperature 0.4 --top-k 20 --top-p 0.8 --repetition-penalty 1.1 --device cuda
```

目前內建的是零額外依賴的 BM25 檢索器，最適合數百至數萬個 `.md`/`.txt` 片段與明確技術術語。資料量更大或需要同義句語意搜尋時，再改用 embedding model 與向量資料庫。

當模型規模不足以可靠重述技術資料時，使用抽取式模式可完全避開模型幻覺：它會依問題檢索，但直接回傳帶來源的段落，不會進行生成。

```powershell
uv run kda-generate --prompt "什麼是 Kimi Delta Attention？" --rag-index runs/rag/knowledge.json --rag-answer-mode extractive --show-sources --checkpoint runs/sft/checkpoints/kda-sft-epoch-2.pt --tokenizer runs/smoke/tokenizer/chinese.model
```

若要更短的可驗證回答，使用 `cited` 模式。它不會載入或呼叫 32M 模型，而是從最高分的段落選出與問題最相關的句子，並以 `[來源編號]` 標記；這是目前技術問答的預設建議。

```bash
uv run kda-generate --checkpoint runs/sft/checkpoints/kda-sft-epoch-2.pt --tokenizer runs/smoke/tokenizer/chinese.model --prompt "KDA 為什麼適合長序列？" --rag-index runs/rag/knowledge.json --rag-answer-mode cited --show-sources --rag-min-score 1.0
```

本地 RAG 預設要求最高 BM25 分數至少為 `1.0`；若證據過弱，程式會停止並說明無法可靠回答，而不是生成猜測。不同領域與索引規模的分數不相同，可用 `--rag-min-score 0` 關閉此保護，或先逐步調低門檻。

### Hybrid 檢索與驗證

BM25 擅長精確術語，embedding 向量檢索擅長同義說法。先用多語 embedding 建立一次可快取的 `.npz` 向量索引，再以 hybrid 模式融合兩者；可選的 cross-encoder reranker 會只重排候選段落，不會修改來源內容。

```bash
uv sync --extra cuda --extra retrieval
uv run kda-build-vector-rag --index runs/rag/knowledge.json --output runs/rag/knowledge.vector.npz --device cuda
```

`verified` 會先由 KDA 生成，再逐句以來源文字驗證；沒有足夠字詞證據的句子會被移除，若全部不通過就回覆資料不足。`--source-conflict refuse` 會在不同來源對同一重疊主張給出不同數字時停止回答；此為保守的啟發式檢查，不應取代人工判讀。

```bash
uv run kda-generate \
  --checkpoint runs/sft/checkpoints/kda-sft-epoch-2.pt \
  --tokenizer runs/smoke/tokenizer/chinese.model \
  --prompt "Kimi Delta Attention 的 chunkwise 演算法有何作用？" \
  --rag-index runs/rag/knowledge.json \
  --vector-index runs/rag/knowledge.vector.npz \
  --retrieval-mode hybrid \
  --reranker \
  --source-conflict refuse \
  --rag-answer-mode verified \
  --verification-min-overlap 0.5 \
  --show-sources \
  --device cuda
```

預設 embedding 模型為 `intfloat/multilingual-e5-small`，reranker 為 `BAAI/bge-reranker-v2-m3`。兩者第一次使用都會從 Hugging Face 下載；若改用其他 embedding 模型，建立與查詢向量索引時必須使用相同的 `--embedding-model`。

### 網路 RAG

加入 `--web-search` 後，預設會從免費的 arXiv 與中文 Wikipedia 查詢論文摘要與百科內容，將來源 URL 放入模型 context，不需要 API key。arXiv 摘要會先以 NLLB 翻成中文，再由 OpenCC 轉為台灣繁體，最後才交給 32M 模型，因此不會要求小型基座自行處理英翻中；第一次使用會下載約 600M 的翻譯模型。先安裝翻譯選用依賴：

```bash
uv sync --extra translation
```

一般生成模式會要求模型以繁體中文回答；`--rag-answer-mode extractive` 會直接回傳已翻譯的來源摘要。若只想保留原文，可加上 `--no-translate-web-sources`：

```bash
uv run kda-generate --checkpoint runs/rag_sft_clean/checkpoints/kda-sft-epoch-8.pt --tokenizer runs/smoke/tokenizer/chinese.model --prompt "Kimi Delta Attention 是什麼？" --web-search --show-sources --temperature 0.2 --top-k 20 --top-p 0.8 --repetition-penalty 1.05 --device cuda
```

預設翻譯模型為 `facebook/nllb-200-distilled-600M`，使用 `eng_Latn -> zho_Hant`；可用 `--translation-model` 改為本機或 Hugging Face 上相容的 NLLB 模型。NLLB 模型採 `CC-BY-NC`，不可直接用於商業產品。

若要搜尋一般即時網路資訊，指定 `--web-provider brave`。先在 Brave 建立 API key，將 [.env.example](.env.example) 複製為 `.env` 後填入 key；`.env` 已被 Git 忽略，不會提交：

```bash
cp .env.example .env
# 編輯 .env：BRAVE_SEARCH_API_KEY=你的 API key
uv run kda-generate --checkpoint runs/rag_sft_clean/checkpoints/kda-sft-epoch-8.pt --tokenizer runs/smoke/tokenizer/chinese.model --prompt "Kimi Delta Attention 的最新技術報告是什麼？" --web-search --web-provider brave --show-sources --temperature 0.2 --top-k 20 --top-p 0.8 --repetition-penalty 1.05 --device cuda
```

網路搜尋是可選功能，啟用後才會對外傳送 prompt。結果摘要與 URL 應視為可查證來源，不代表模型輸出的每句話都已驗證。

### RAG-SFT

一般 SFT 不會教模型如何讀取 reference context；對 32M 模型，應在一般 SFT 後再做一段低 learning-rate 的 RAG-SFT。下列一鍵流程會從索引產生「參考資料 + 問題 + 帶 `[來源]` 的回答」樣本，並預設加入 25% 的資料不足拒答樣本，再以 `1e-5` 做 8 epochs 短期微調：

```powershell
uv run kda-rag-sft-pipeline --checkpoint runs/sft/checkpoints/kda-sft-epoch-2.pt --tokenizer runs/smoke/tokenizer/chinese.model --rag-index runs/rag/knowledge.json --device cuda
```

最終 checkpoint 位於 `runs/rag_sft/checkpoints/kda-sft-epoch-8.pt`。這是讓模型學會 grounded response 的起始流程；加入更多獨立的高品質技術文件後，重建索引並以 `--no-resume` 重跑，效果會比重複同一份小型知識庫更可靠。

資料庫包含多份互相獨立的文件時，可讓每筆訓練樣本帶入兩個來源，訓練模型輸出多來源標記；`--refusal-ratio` 控制拒答樣本比例。這不會消除 32M 的能力上限，但能顯著降低它在缺乏證據時編造答案的機率：

```bash
uv run kda-rag-sft-pipeline --checkpoint runs/sft/checkpoints/kda-sft-epoch-2.pt --tokenizer runs/smoke/tokenizer/chinese.model --rag-index runs/rag/knowledge.json --context-chunks 2 --refusal-ratio 0.25 --no-resume --device cuda
```

## 擴大基座與資料品質

32M 模型適合驗證 KDA 架構、訓練流程與 RAG；它不足以在沒有來源時可靠記住廣泛技術知識。可先使用 [configs/model_64m.json](configs/model_64m.json) 做中間驗證：640 hidden size、10 層、5 個 128 維 KDA heads、1,792 FFN 維度，共 64,375,730 個參數，仍維持 `chunk_kda` 所需的 128 維 head。若 64M 在固定評測集上明顯提升，再升至 [configs/model_100m.json](configs/model_100m.json)：768 hidden size、12 層、6 個 128 維 KDA heads，約 1 億參數。

升級模型前，先建立固定的技術領域評測集。continued pretraining 應加入至少 1 億至 3 億 tokens 的高品質、去重技術文件，並保留一部分通用繁中語料避免語言能力退化。之後使用多來源的 context-QA RAG-SFT，而不是反覆訓練同一份文件；只有評測集的 grounded correctness 明顯提升，才值得繼續放大模型或訓練 token budget。

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

64M 驗證可使用同一條 pipeline，但務必使用獨立的 work directory，避免和 32M 的 tokenizer、資料狀態與 checkpoint 混在一起：

```bash
uv run kda-pipeline --model-config configs/model_64m.json --train-config configs/train_64m_gpu.json --total-documents 100000 --steps 100 --work-dir runs/64m_smoke --device cuda --compile
```

`train_64m_gpu.json` 使用 64 個序列的 micro-batch 與 `grad_accum=2`，適合作為 15.5 GiB GPU 的保守起點；不要讓 64M 沿用 32M 的 `batch_size=160`。確認 smoke run 的 loss 能下降後，建立更大且授權明確的來源 manifest，再改用 `--max-tokens` 執行正式訓練。`--steps` 與 `--max-tokens` 互斥；前者只用於短測，後者才是正式 token budget。
