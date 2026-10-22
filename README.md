# 強化 RAG 與提示工程之大型語言模型在特殊教育 IEP 生成任務的應用

**Enhancing RAG and Prompt Engineering to Enhance LLM-based Text Generation in Special Education**

一套輔助特教教師撰寫**個別化教育計畫（Individualized Education Program, IEP）**的生成系統。系統以 **RAG（Retrieval-Augmented Generation）** 整合特教法規、課綱與教材，搭配 **ReAct 式提示工程**與**多輪上下文記憶**，產出有法規依據、可直接應用於教育現場的 IEP 內容，減輕特教教師的文書負擔。

🎬 [Demo 影片](https://youtu.be/NlvRbcs7pFk) ・ 📊 [Benchmark 實驗結果](https://pse.is/8g35q5) ・ 📝 [三模型生成結果比較](https://hackmd.io/@RuoWei/Hk45yIPGbg) ・ 📋 [盲測表單](https://www.surveycake.com/s/aMzkK)

---

## 目錄

- [專案動機](#專案動機)
- [系統架構](#系統架構)
- [方法](#方法)
  - [Retriever 檢索器](#retriever-檢索器)
  - [Generator 生成器](#generator-生成器)
  - [Context Memory 上下文記憶](#context-memory-上下文記憶)
- [實驗](#實驗)
- [結果](#結果)
- [專案結構](#專案結構)
- [團隊成員](#團隊成員)
- [參考文獻](#參考文獻)

---

## 專案動機

特教老師需為每位身心障礙學生撰寫 IEP，規劃學習目標、教學策略與評量方式，但撰寫過程繁瑣耗時。直接使用通用 LLM 輔助時，常因缺乏特教領域知識而出現幻覺、不符教材或語意籠統等問題。

**主要挑戰**

1. LLM 缺乏特教領域知識，生成內容常出現錯誤與幻覺。
2. 特教資料量少且多為非結構化文本，不利於模型微調。
3. IEP 撰寫需明確步驟與結構，現有模型缺乏分階段生成與內容組織能力。

**問題定義**

1. 如何在不做 Fine-tune 的情況下，讓 LLM 掌握特教領域知識（導入 RAG）。
2. 如何正確擷取非結構化文本內容，再透過 RAG 進行檢索。
3. 如何在提示長度有限的情況下，兼顧教育專業知識與生成架構。

---

## 系統架構

系統由 **Retriever（檢索器）** 與 **Generator（生成器）** 兩大模組組成：

```
                    Start / Query
                          │
   Files                  ▼
     │              Query Expand
     ▼                    │
Files Abstracting         ▼
 (Chunking)         Query Embedding
     │                    │
     ▼                    ▼
Files Embedding ──► Vector DB ──► Similarity Search      ┐
                                        │                 │ Retriever
                                        ▼                 │
                                  Files Reranking         ┘
                                        │
             Prompt Engineering ──► LLM Generate           ── Generator
                                        │
                                        ▼
                                 End / Response
```

| 元件 | 採用方法 |
|------|----------|
| Chunking | Overlapping Window（重疊式切塊） |
| Embedding | Qwen3 Embedding 0.6B（Instruction-aware） |
| Vector DB | FAISS `IndexFlatIP` + L2 normalization（等價 Cosine Similarity），取 Top-100 |
| Reranker | BGE m3 v2（Cross-Encoder） |
| LLM | Qwen3-4B |
| Prompt | ReAct-inspired 結構化提示 + IEP 專用 Prompt Template |
| Memory | Profile Memory + Multi-turn Memory（保留最近 10 組對話） |

---

## 方法

### Retriever 檢索器

#### 1. File Abstracting / Chunking

參考 HiChunk [1]，比較三種切塊策略：

| 策略 | 結果 |
|------|------|
| 固定長度切塊（Fixed-size） | 易在條文未結束時截斷，語意斷裂，不適合法律文本 |
| **重疊式切塊（Overlapping Window）** | ✅ 相鄰 chunk 保留重疊內容，降低語意截斷造成的資訊遺失，且不顯著增加檢索噪音 |
| 層次式切塊（Hierarchical） | 法律條文存在大量跨段落語意依賴，即使先經地端模型整理格式，仍難切出語意獨立單元 |

另曾嘗試於 chunking 後用地端模型做摘要，但摘要會刪減具法定意義的細節，因此**未採用**摘要式 File Abstracting。

#### 2. Embedding with Qwen3

採用 **Qwen3 Embedding（0.6B）** [3]：效能接近大型 embedding 模型、推論成本低、支援 Instruction-aware 語義對齊，且對中文法規文本適應良好。

- **Query**：使用指令式嵌入
  ```
  {Instruction} {Query} <|endoftext|>
  ```
- **File**：chunk 原文直接嵌入，**不加** Instruction，維持法律條文語意完整

> 注意：Instruction 設計不當可能增加語義變異或降低 Recall，實作上應保持 Instruction 一致且簡潔。

#### 3. Vector Database with FAISS

- 使用 `IndexFlatIP`（精確搜尋，不壓縮、不近似），結果穩定可重現
- 所有向量先做 L2 normalization，使內積等價於 Cosine Similarity
- 每筆 Query 取 **Top-100** chunk 作為候選集合，兼顧 Recall 並為 Reranking 保留語義多樣性

#### 4. Reranking

比較兩種 Reranker：

| | Qwen3 4B（LLM-based Classifier） | **BGE m3 v2（Encoder-only）** ✅ |
|---|---|---|
| 打分方式 | $\text{score}(q,d)=\dfrac{e^{P(\text{yes}\mid q,d)}}{e^{P(\text{yes}\mid q,d)}+e^{P(\text{no}\mid q,d)}}$ | $\text{score}(q,d)=\mathbf{w}^\top\mathbf{h}_{\text{CLS}}+b$ |
| 輸入 | Query + Document + Prompt | `[CLS] Query [SEP] Document [SEP]` |
| 優點 | 具推理能力，prompt 彈性高 | 不受 prompt 影響、排序穩定、推論快 |
| 缺點 | 依賴 prompt、成本與延遲高、相似文件難區分 | 偏重語義相關性，非開放式推理 |

法律條文重點在精準匹配而非開放推理，最終選用 **BGE m3 v2**。

### Generator 生成器

#### 1. Prompt Engineering

參考 ReAct [4] 的「思考 → 行動」結構化控制精神，但考量法規場景，**不允許模型自由推理或補充背景知識**，轉化為「行為約束式 Prompt」，讓模型扮演「受控的文件解讀者」。

System Prompt 包含：

- **【角色設定】** 根據「文件內容」嚴格遵循回答規則的中文助手
- **【Thought】** 目的分析 → 資訊檢索計畫 → 資訊彙整與驗證 → 答案建構
- **【回答規則】**
  1. 回答只能依據文件內容，不可編造
  2. 引用來源必須明確寫出檔名，例如《xxx.pdf》
  3. 不可引用未出現在文件列表中的檔名
  4. 文件沒有答案時，直接回答「文件裡沒有提到相關資訊」
  5. 回答自然、簡單，一般中文口吻即可
  6. Final Answer 只回答一次，不得重複或加入多餘說明
  7. Thought 須完整推理，但不得在 Final Answer 中重複

#### 2. IEP Prompt Template

**角色**：具備台灣特殊教育專業的 IEP 制定專家，涵蓋視障、聽障、自閉症、學習障礙等多障別需求。

**依據來源**：

- 《十二年國民基本教育身心障礙相關之特殊需求領域課程綱要》
- 《十二年國民基本教育高級中等教育階段學校集中式特殊教育班服務群科課程綱要》
- 《十二年國民基本教育特殊教育課程實施規範》

**輸出欄位**：現況能力分析、學期目標（SMART 原則）、教學策略、評量方式、RAG 查詢結果引用、多障別差異化需求。

**Thought–Action–Observation 結構**（每個主要欄位皆依此輸出）：

| 區段 | 用途 |
|------|------|
| 【專家思考 Thought】 | 資料分析、課綱對照、策略制定（不列入最終 IEP） |
| 【IEP 撰寫 Action】 | 正式 IEP 內容：具體、可觀察、可操作 |
| 【查詢結果 Observation】 | 僅呈現 RAG 回傳的課綱／法規原文，不得改寫或捏造 |

**風險控制**：禁止捏造醫療資訊、診斷或視力／聽力數據；資料不足時標示「未提供」；所有欄位皆需產生。

### Context Memory 上下文記憶

支援多輪對話中逐步補充學生資料、修正目標或調整策略：

| 層級 | 內容 |
|------|------|
| **Profile Memory**（學生資料永久記憶） | 年級、教育安置、障別、長期教育需求等準靜態資訊 |
| **Multi-turn Memory**（對話多輪記憶） | 教師補充說明、已生成欄位、修正與回饋 |
| **記憶管理策略** | 僅保留最近 10 組對話；舊對話自動捨棄；兩種記憶分開管理，避免學生核心資料被誤刪 |

---

## 實驗

### Dataset

目前沒有針對台灣特殊教育的公開 RAG benchmark，因此自建小型任務導向資料集：

- **來源**：優質特教網（教材資料庫、教學實務案例）、《特殊教育法》及施行細則、線上特教教材
- **結構**：`(Query, Document/Passage, Ground Truth Answer)` 三元組
- **規模**：52 筆問答樣本
- **評估**：人工驗證，**完全命中（Exact Match）** 標準答案才算正確（不可多答或少答、不可加入文件外資訊、無語意偏差）

### Ablation Study

| # | 設定 | 目的 |
|---|------|------|
| 1 | Qwen3-4B + RAG + Prompt Engineering（完整系統） | 主要比較基準 |
| 2 | Qwen3-4B + RAG | 評估 Prompt Engineering 的貢獻 |
| 3 | Qwen3-4B（Base Model） | 檢驗僅靠模型內部知識的表現與幻覺 |
| 4 | TAIDE / LLaMA（台灣資料微調） | 在地化模型對照 |
| 5 | GPT-5.1 mini | 通用大型語言模型對照 |

- **RAG 的必要性**：比較 1、2、3
- **Prompt Engineering 的貢獻**：比較 1、2
- **專用系統 vs 通用／在地化模型**：比較 1、4、5

### Expert Assessment

- 訪談交大資源教室 4 位教師
- 以線上表單邀請特教老師與領域專家進行**情境式盲測評估**

統一學生情境（國中一年級、輕度／高功能自閉症，含溝通、社交、情緒、感官與學習特質描述），由三個模型生成 IEP 特殊需求領域（社會技巧）學期目標，以隨機順序、不標示來源呈現：

- 版本一：Gemini 3 Pro
- 版本二：Qwen3-4B + RAG + Prompt Engineering（本專案）
- 版本三：GPT-5 mini

受試者依五項指標以 Likert 量表（1–5）評分，並進行整體排序與開放式回饋。

---

## 結果

### 定量結果（52 題，Exact Match）

| | Qwen3-4B + RAG + PE | Qwen3-4B + RAG | Qwen3-4B | TAIDE/LLaMA | GPT |
|---|---|---|---|---|---|
| 正確 | 47 | 47 | 5 | 9 | 12 |
| 錯誤 | 5 | 5 | 47 | 43 | 40 |
| **ACC** | **0.904** | **0.904** | 0.096 | 0.173 | 0.231 |

- **RAG 具決定性影響**：Qwen3-4B 的 ACC 從 0.096 提升至 0.904。
- **通用與在地化模型不足**：GPT 與 TAIDE/LLaMA 在完全命中標準下明顯落後。
- **Prompt Engineering 未反映在 ACC 上**：Exact Match 只判斷對錯，不評估結構完整性、專業語氣與實務可用性，因此 PE 的效益改由專家評估判斷。

### 專家評估結果

**Prompt Engineering 有效性**：4 位特教教師一致認為，加入 PE 的輸出在結構完整性、專業用語與 IEP 欄位實務可行性上，皆明顯優於未加入 PE 的版本。

**三版本盲測評分（Likert 1–5）**

| 評估指標 | 版本一<br>Gemini 3 Pro | 版本二<br>**本專案** | 版本三<br>GPT-5 mini |
|------|:---:|:---:|:---:|
| 目標設計明確、可量化 | 3.00 | 3.17 | 3.17 |
| 展現特殊教育的深入理解 | **3.50** | 3.17 | 3.17 |
| 正確引用課程綱要、正式法規文件 | 2.00 | **4.50** | 2.50 |
| 內容層次分明、容易閱讀與理解 | 3.50 | 3.50 | 3.50 |
| 符合實務需求，可直接應用於 IEP 撰寫 | 2.83 | **3.50** | 2.83 |
| **整體平均** | 2.966 | **3.568** | 3.034 |

- 版本二在**法規引用正確性**（4.5 vs 2.0／2.5）與**實務可用性**上優勢明顯。
- 多數教師將版本二評為最符合實務需求；版本一被指出需要最多後續人工修改。
- 質性回饋：版本二最貼近真實 IEP 寫作風格，目標與指標符合課綱指引，不需人工來回對照。

---

## 專案結構

依專案程式碼目錄整理（詳見 GitHub repo）：

```
LLM-FINAL-PROJECT/
├── api/
├── config/
├── data/
├── generator/
├── llm_final_frontend/
├── retriever/
│   ├── file_abstractor.py
│   ├── file_embedding.py
│   ├── file_loader.py
│   ├── pipeline.py
│   ├── query_embedding.py
│   ├── query_expand.py
│   ├── reranker.py
│   ├── similarity_search.py
│   └── vector_store.py
└── utils/
    ├── chunk.py
    ├── chunk_hier.py
    ├── img_reader.py
    └── pdf_reader.py
```

---

## 結論

本專案建立一套專為特殊教育打造的 IEP 撰寫輔助系統，能同時「找到正確資料」並「生成高品質內容」。系統以《特殊教育法》、十二年國教課綱、IEP 範例與教材構建向量化知識庫，完成前後端整合，並透過客製化 Prompt 與多輪對話記憶，生成貼近教育現場、可實際應用的 IEP 目標、教學調整與評量方式。

**未來方向**：持續完善模型、優化流程架構、擴充資料庫，並探索 LLM 於更多專業領域的應用。

---

## 團隊成員

國立陽明交通大學（National Yang Ming Chiao Tung University）

| 成員 | 所屬單位 |
|------|----------|
| 張若薇 | Institute of Computer Science and Engineering |
| 林家賢 | The Graduate Degree Program of Artificial Intelligence |
| 張瑞顯 | Institute of Artificial Intelligence Innovation |
| 李宜蓁 | Institute of Artificial Intelligence Innovation |

---

## 參考文獻

1. Chen, Z., Liu, J., Xiao, Y., Wang, Z., & Zhang, Y. (2024). *HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking.* arXiv:2403.09755.
2. Xiao, S., Liu, Z., Shao, X., & Zhang, Y. (2024). *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv:2402.03216.
3. Zhang, Y., Li, M., Long, D., et al. (2025). *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models.* arXiv:2506.05176.
4. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
