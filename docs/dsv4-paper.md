---
title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
source: "https://arxiv.org/html/2606.19348"
language: "en"
word_count: 21349
---

DeepSeek-AI  
research@deepseek.com

###### Abstract

We present a preview version of DeepSeek-V4 series, including two strong Mixture-of-Experts (MoE) language models — DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) — both supporting a context length of one million tokens. DeepSeek-V4 series incorporate several key upgrades in architecture and optimization: (1) a hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to improve long-context efficiency; (2) Manifold-Constrained Hyper-Connections (mHC) that enhance conventional residual connections; (3) and the Muon optimizer for faster convergence and greater training stability. We pre-train both models on more than 32T diverse and high-quality tokens, followed by a comprehensive post-training pipeline that unlocks and further enhances their capabilities. DeepSeek-V4-Pro-Max, the maximum reasoning effort mode of DeepSeek-V4-Pro, redefines the state-of-the-art for open models, outperforming its predecessors in core tasks. Meanwhile, DeepSeek-V4 series are highly efficient in long-context scenarios. In the one-million-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2. This enables us to routinely support one-million-token contexts, thereby making long-horizon tasks and further test-time scaling more feasible. The model checkpoints are available at [https://huggingface.co/collections/deepseek-ai/deepseek-v4](https://huggingface.co/collections/deepseek-ai/deepseek-v4).

![Refer to caption](https://arxiv.org/html/2606.19348v1/x1.png)

Figure 1: Left: benchmark performance of DeepSeek-V4-Pro-Max and its counterparts. Right: inference FLOPs and KV cache size of DeepSeek-V4 series and DeepSeek-V3.2.

## 1 Introduction

The emergence of reasoning models \[o1, dsr1\] has established a new paradigm of test-time scaling, driving substantial performance gains for Large Language Models (LLMs). However, this scaling paradigm is fundamentally constrained by the quadratic computational complexity of the vanilla attention mechanism \[transformer\], which creates a prohibitive bottleneck for ultra-long contexts and reasoning processes. Concurrently, the emergence of long-horizon scenarios and tasks — from complex agentic workflows to massive cross-document analysis — has also made efficient support for ultra-long contexts critical for future progress. While recent open-source efforts \[dsv3, qwen3, minimax\_m2, kimi\_k2\] have advanced general capabilities, this core architectural inefficiency in handling ultra-long sequences remains a key impediment, limiting further gains from test-time scaling and hindering further exploration into long-horizon scenarios and tasks.

In order to break the efficiency barrier in ultra-long contexts, we develop the DeepSeek-V4 series, including the preview versions of DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated). Through architectural innovations, DeepSeek-V4 series achieve a dramatic leap in computational efficiency for processing ultra-long sequences. This breakthrough enables efficient support for a context length of one million tokens, ushering in a new era of million-length contexts for next-generation LLMs. We believe our capability to efficiently handle ultra-long sequences unlocks the next frontier of test-time scaling, paves the way for deeper research into long-horizon tasks, and establishes a necessary foundation for exploring future paradigms like online learning.

Compared with the DeepSeek-V3 architecture \[dsv3\], DeepSeek-V4 series retain the DeepSeekMoE framework \[deepseekmoe\] and Multi-Token Prediction (MTP) strategy, while introducing several key innovations in architecture and optimization. To enhance long-context efficiency, we design a hybrid attention mechanism combining Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA). CSA compresses the KV caches along the sequence dimension and then performs DeepSeek Sparse Attention (DSA) \[dsv32\], whereas HCA applies more aggressive compression to the KV caches but keeps dense attention. To strengthen modeling capability, we incorporate Manifold-Constrained Hyper-Connections (mHC) \[mhc\] that upgrade conventional residual connections. Additionally, we introduce the Muon \[muon, muon\_kimi\] optimizer to the training of DeepSeek-V4 series, leading to faster convergence and improved training stability.

To enable efficient training and inference for DeepSeek-V4 series as well as productive development, we introduce several infrastructure optimizations. First, we design and implement a single fused kernel for MoE modules that fully overlaps computation, communication, and memory access. Second, we employ TileLang \[wangtilelang\], a Domain-Specific Language (DSL) to balance development productivity and runtime efficiency. Third, we provide efficient batch-invariant and deterministic kernel libraries to ensure bitwise reproducibility across training and inference. Fourth, for the training framework, we extend the autograd framework with tensor-level checkpointing for fine-grained recomputation control; and we enhance training efficiency with a hybrid ZeRO strategy for the Muon optimizer, cost-effective mHC implementations via recomputation and fused kernels, and two-stage contextual parallelism to manage compressed attention. Fifth, for the inference framework, we design a heterogeneous KV cache structure with on-disk storage strategies to enable efficient shared-prefix reuse. In addition, during the post-training stage, we incorporate FP4 quantization-aware training for MoE expert weights and the indexer QK path to reduce memory and computation.

By employing hybrid CSA and HCA, along with precision optimizations on computation and storage, DeepSeek-V4 series achieve significantly lower inference FLOPs and a substantially reduced KV cache size compared with DeepSeek-V3.2, especially in long-context settings. The right part of Figure 1 demonstrates the estimated single-token inference FLOPs and accumulated KV cache size of DeepSeek-V3.2 and DeepSeek-V4 series. In the scenario of 1M-token context, even DeepSeek-V4-Pro, which has a larger number of activated parameters, attains only 27% of the single-token FLOPs (measured in equivalent FP8 FLOPs) and 10% of the KV cache size relative to DeepSeek-V3.2. Furthermore, DeepSeek-V4-Flash, with its smaller number of activated parameters, pushes efficiency even further: in the 1M-token context setting, it achieves only 10% of the single-token FLOPs and 7% of the KV cache size compared with DeepSeek-V3.2. Additionally, for DeepSeek-V4 series, the routed expert parameters utilize FP4 precision. While the peak FLOPs for FP4 $\times$ FP8 operations are currently the same as FP8 $\times$ FP8 on existing hardware, they can theoretically be implemented to be $1/3$ more efficient on future hardware, which will further enhance the efficiency of DeepSeek-V4 series.

During pre-training, we train DeepSeek-V4-Flash on 32T tokens and DeepSeek-V4-Pro on 33T tokens, respectively. After pre-training, these two models can natively and efficiently support 1M-length contexts. In our internal evaluations, DeepSeek-V4-Flash-Base already surpasses DeepSeek-V3.2-Base across a majority of benchmarks with its more parameter-efficient design. DeepSeek-V4-Pro-Base further extends this advantage to set a new performance standard among DeepSeek foundation models, achieving comprehensive superiority across reasoning, coding, long-context, and world knowledge tasks.

The post-training pipeline of DeepSeek-V4 series features a two-stage paradigm: the independent cultivation of domain-specific experts, followed by unified model consolidation via on-policy distillation \[lu2025onpolicydistillation, minillm\]. Initially, for each target domain — such as mathematics, coding, agent, and instruction following — a separate expert model is trained independently. The base model first undergoes Supervised Fine-Tuning (SFT) on high-quality, domain-specific data to establish foundational capabilities. Subsequently, Reinforcement Learning (RL) is applied using Group Relative Policy Optimization (GRPO) \[dsr1\], which further optimizes the model for domain-aligned behaviors guided by reward models tailored to specific success criteria. This phase yields a diverse set of specialized experts, each excelling in its respective field. Finally, to integrate these distinct proficiencies, a single unified model is trained through on-policy distillation, wherein the unified model acts as the student learning to optimize the reverse KL loss with teacher models.

Summary of Core Evaluation Results

- Knowledge: In assessments of broad world knowledge, DeepSeek-V4-Pro-Max, the maximum reasoning effort mode of DeepSeek-V4-Pro, significantly outperforms leading open-source models on the SimpleQA \[simpleqa\] and Chinese-SimpleQA \[csimpleqa\] benchmarks. Regarding educational knowledge — evaluated via MMLU-Pro \[mmlu\_pro\], HLE \[hle\], and GPQA \[gpqa\] — DeepSeek-V4-Pro-Max shows a marginal lead over its open-source counterparts. DeepSeek-V4-Pro-Max has significantly closed the gap with the leading proprietary model, Gemini-3.1-Pro, despite still trailing it in these knowledge-based evaluations.
- Reasoning: Through the expansion of reasoning tokens, DeepSeek-V4-Pro-Max demonstrates superior performance relative to GPT-5.2 and Gemini-3.0-Pro on standard reasoning benchmarks. Nevertheless, its performance falls marginally short of GPT-5.4 and Gemini-3.1-Pro, suggesting a developmental trajectory that trails state-of-the-art frontier models by approximately 3 to 6 months. Furthermore, DeepSeek-V4-Flash-Max achieves comparable performance to GPT-5.2 and Gemini-3.0-Pro, establishing itself as a highly cost-effective architecture for complex reasoning tasks.
- Agent: On public benchmarks, DeepSeek-V4-Pro-Max is on par with leading open-source models, such as Kimi-K2.6 and GLM-5.1, but slightly worse than frontier closed models. In our internal evaluation, DeepSeek-V4-Pro-Max outperforms Claude Sonnet 4.5 and approaches the level of Opus 4.5.
- Long-Context: DeepSeek-V4-Pro-Max delivers strong results on synthetic and real use cases with a 1-million-token context window, surpassing even Gemini-3.1-Pro on academic benchmarks.
- DeepSeek-V4-Pro v.s. DeepSeek-V4-Flash: DeepSeek-V4-Flash-Max exhibits lower performance in knowledge evaluations due to its smaller parameter scale. However, it achieves comparable results on reasoning tasks when allocated a larger thinking budget. In agent evaluations, while DeepSeek-V4-Flash-Max matches the performance of DeepSeek-V4-Pro-Max on several benchmarks, it still trails its larger counterpart on more complex, high-difficulty tasks.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x2.png)

Figure 2: Overall architecture of DeepSeek-V4 series. We use hybrid CSA (Compressed Sparse Attention) and HCA (Heavily Compressed Attention) for attention layers, DeepSeekMoE for feed-forward layers, and strengthen conventional residual connections with m HC.

## 2 Architecture

Overall, DeepSeek-V4 series retain the Transformer \[transformer\] architecture and Multi-Token Prediction (MTP) modules \[meta\_mtp, dsv3\], while introducing several key upgrades over DeepSeek-V3: (1) firstly, we introduce the Manifold-Constrained Hyper-Connections (mHC) \[mhc\] to strengthen conventional residual connections; (2) secondly, we design a hybrid attention architecture, which greatly improves long-context efficiency through Compressed Sparse Attention and Heavily Compressed Attention. (3) thirdly, we employ Muon \[muon, muon\_kimi\] as the optimizer. For the Mixture-of-Experts (MoE) components, we still adopt the DeepSeekMoE \[deepseekmoe\] architecture, with only minor adjustments from DeepSeek-V3. The Multi-Token Prediction (MTP) \[ms\_mtp, meta\_mtp, eagle, dsv3\] configuration remains identical to that of DeepSeek-V3. All other unspecified details follow the settings established in DeepSeek-V3 \[dsv3\]. Figure 2 illustrates the overall architecture of DeepSeek-V4, and the details are described below.

### 2.1 Designs Inherited from DeepSeek-V3

##### Mixture-of-Experts.

As previous DeepSeek-series models \[dsvii, dsv3\], DeepSeek-V4 series also adopt the DeepSeekMoE paradigm \[deepseekmoe\] for Feed-Forward Networks (FFNs), which sets fine-grained routed experts and shared experts. Different from DeepSeek-V3, we change the activation function that computes the affinity scores from $\operatorname{Sigmoid}(\cdot)$ into $\operatorname{Sqrt}(\operatorname{Softplus}(\cdot))$. For load balancing, we also employ the auxiliary-loss-free strategy \[noaux\_tc, dsv3\], augmented by a slight sequence-wise balance loss that prevents extreme imbalance within individual sequences. For DeepSeek-V4, we remove the constraint on the number of routing target nodes, and carefully redesign the parallelism strategy to maintain training efficiency. Furthermore, compared with DeepSeek-V3, we replace the dense FFN layers in the initial several Transformer blocks with MoE layers that employ Hash routing \[hash\_layer\]. The Hash routing strategy determines the target experts of each token according to a predefined hash function with regard to the input token ID.

##### Multi-Token Prediction.

As DeepSeek-V3, DeepSeek-V4 series also set MTP modules and objectives. Given that the MTP strategy has been validated in DeepSeek-V3, we adopt the same strategy for DeepSeek-V4 series without modification.

### 2.2 Manifold-Constrained Hyper-Connections

As shown in Figure 2, DeepSeek-V4 series incorporate Manifold-Constrained Hyper-Connections (mHC) \[mhc\] to strengthen the conventional residual connections between adjacent Transformer blocks. Compared with naive Hyper-Connections (HC) \[hc\], the core idea of mHC is to constrain the residual mapping onto a specific manifold, and thus enhance the stability of signal propagation across layers while preserving model expressivity. This subsection briefly introduces the standard HC and describes how we design mHC for stable training.

##### Standard Hyper-Connections.

The standard HC expands the width of the residual stream by a factor of $n_{\text{hc}}$. Specifically, the shape of the residual stream is expanded from $\mathbb{R}^{d}$ to $\mathbb{R}^{n_{\text{hc}}\times d}$, where $d$ is the hidden size of the actual layer input. Let $X_{l}=[\mathbf{x}_{l,1};\ldots;\mathbf{x}_{l,n_{\text{hc}}}]^{T}\in\mathbb{R}^{n_{\text{hc}}\times d}$ be the residual state before the $l$ -th layer. HC introduces three linear mappings: an input mapping $A_{l}\in\mathbb{R}^{1\times n_{\text{hc}}}$, a residual transformation $B_{l}\in\mathbb{R}^{n_{\text{hc}}\times n_{\text{hc}}}$, and an output mapping $C_{l}\in\mathbb{R}^{n_{\text{hc}}\times 1}$. The update of the residual state is then formulated as:

$$
X_{l+1}=B_{l}X_{l}+C_{l}\mathcal{F}_{l}(A_{l}X_{l}),
$$

where $\mathcal{F}_{l}$ denotes the $l$ -th layer (e.g., an MoE layer), whose input and output shapes are both $\mathbb{R}^{d}$. Note that the actual layer input $A_{l}X_{l}\in\mathbb{R}^{d}$ is also $d$ -dimensional, so the expanded residual width does not influence the design of the inner layers. HC decouples the residual width from the actual hidden size, offering a complementary scaling axis with minimal computational overhead, as $n_{\text{hc}}$ is typically much smaller than the hidden size $d$. However, even though HC has demonstrated potential in improving model performance, we find that the training will frequently exhibit numerical instability when stacking multiple layers, which hinders the scaling of HC.

##### Manifold-Constrained Residual Mapping.

The core innovation of mHC is to constrain the residual mapping matrix $B_{l}$ to the manifold of doubly stochastic matrices (the Birkhoff polytope) $\mathcal{M}$, and thus enhance the stability of signal propagation across layers:

$$
B_{l}\in\mathcal{M}\coloneq\{M\in\mathbb{R}^{n\times n}\mid M\mathbf{1}_{n}=\mathbf{1}_{n},\;\mathbf{1}_{n}^{T}M=\mathbf{1}_{n}^{T},\;M\geqslant 0\}.
$$

This constraint ensures that the spectral norm of the mapping matrix $\|B_{l}\|_{2}$ is bounded by 1, so the residual transformation is non-expansive, which increases the numerical stability during both the forward pass and backpropagation. Besides, the set $\mathcal{M}$ is closed under multiplication, which guarantees stability in the scenarios of deep stacks of mHC. In addition, the input transformation $A_{l}$ and output transformation $C_{l}$ are also constrained to be non-negative and bounded via a Sigmoid function to avoid the risk of signal cancellation.

##### Dynamic Parameterization.

The parameters of three linear mappings are dynamically generated, which are decomposed into a dynamic (input-dependent) component and a static (input-independent) component. Given the input $X_{l}\in\mathbb{R}^{n_{\text{hc}}\times d}$, it is first flattened and normalized: $\hat{X}_{l}=\operatorname{RMSNorm}(\operatorname{vec}(X_{l}))\in\mathbb{R}^{1\times n_{\text{hc}}d}$. Then, we follow the conventional HC to generate the unconstrained raw parameters $\tilde{A}_{l}\in\mathbb{R}^{1\times n_{\text{hc}}}$, $\tilde{B}_{l}\in\mathbb{R}^{n_{\text{hc}}\times n_{\text{hc}}}$, and $\tilde{C}_{l}\in\mathbb{R}^{n_{\text{hc}}\times 1}$:

$$
\displaystyle\tilde{A}_{l}
$$
 
$$
\displaystyle=\alpha_{l}^{\mathrm{pre}}\cdot(\hat{X}_{l}W^{\mathrm{pre}}_{l})+S_{l}^{\mathrm{pre}},
$$
$$
\displaystyle\tilde{B}_{l}
$$
 
$$
\displaystyle=\alpha_{l}^{\mathrm{res}}\cdot\operatorname{Mat}(\hat{X}_{l}W^{\mathrm{res}}_{l})+S_{l}^{\mathrm{res}},
$$
$$
\displaystyle\tilde{C}_{l}
$$
 
$$
\displaystyle=\alpha_{l}^{\mathrm{post}}\cdot(\hat{X}_{l}W^{\mathrm{post}}_{l})^{T}+S_{l}^{\mathrm{post}},
$$

where $W^{\mathrm{pre}}_{l},W^{\mathrm{post}}_{l}\in\mathbb{R}^{n_{\text{hc}}d\times n_{\text{hc}}}$ and $W^{\mathrm{res}}_{l}\in\mathbb{R}^{n_{\text{hc}}d\times n_{\text{hc}}^{2}}$ are learnable parameters for generating the dynamic components; $\operatorname{Mat}(\cdot)$ reshapes a vector of size $1\times n_{\text{hc}}^{2}$ into a matrix of size $n_{\text{hc}}\times n_{\text{hc}}$; $S_{l}^{\mathrm{pre}}\in\mathbb{R}^{1\times n_{\text{hc}}}$, $S_{l}^{\mathrm{post}}\in\mathbb{R}^{n_{\text{hc}}\times 1}$, and $S_{l}^{\mathrm{res}}\in\mathbb{R}^{n_{\text{hc}}\times n_{\text{hc}}}$ are learnable static biases; and $\alpha_{l}^{\mathrm{pre}}$, $\alpha_{l}^{\mathrm{res}}$, $\alpha_{l}^{\mathrm{post}}\in\mathbb{R}$ are learnable gating factors initialized to small values.

##### Applying Parameter Constraints.

After obtaining the unconstrained raw parameters $\tilde{A}_{l},\tilde{B}_{l},\tilde{C}_{l}$, we then apply constraints described earlier to them to enhance the numerical stability. To be specific, for the input and output mappings, we employ a Sigmoid function $\sigma(\cdot)$ to ensure their non-negativity and boundedness:

$$
\displaystyle A_{l}
$$
 
$$
\displaystyle=\sigma(\tilde{A}_{l}),
$$
$$
\displaystyle C_{l}
$$
 
$$
\displaystyle=2\sigma(\tilde{C}_{l}).
$$

As for the residual mapping $\tilde{B}_{l}$, we project it onto the manifold of doubly stochastic matrices $\mathcal{M}$. This is achieved by the Sinkhorn-Knopp algorithm, which first applies an exponential function to $\tilde{B}_{l}$ to ensure positivity, getting $M^{(0)}=\exp(\tilde{B}_{l})$, and then iteratively performs column and row normalization:

$$
M^{(t)}=\mathcal{T}_{r}(\mathcal{T}_{c}(M^{(t-1)})),
$$

where $\mathcal{T}_{r}$ and $\mathcal{T}_{c}$ denote row and column normalization, respectively. This iteration converges to a constrained doubly stochastic matrix $B_{l}=M^{(t_{\text{max}})}$. We choose $t_{\text{max}}=20$ as a practical value.

### 2.3 Hybrid Attention with CSA and HCA

As the context length reaches extreme scales, the attention mechanism emerges as the dominant computational bottleneck in a model. For DeepSeek-V4, we design two efficient attention architectures — Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) — and employ their interleaved hybrid configuration, which substantially reduces the computational cost of attention in long-text scenarios. CSA integrates both compression and sparse attention strategies: it first compresses the Key-Value (KV) cache of every $m$ tokens into one entry, and then applies DeepSeek Sparse Attention (DSA) \[dsv32\] where each query token attends to only $k$ compressed KV entries. HCA aims for extreme compression by consolidating the KV cache of every $m^{\prime}$ ($\gg m$) tokens into a single entry. The hybrid architecture of CSA and HCA remarkably improves the long-context efficiency of DeepSeek-V4 series, making one-million-token context feasible in practice. This subsection describes the core techniques of our hybrid attention architecture, and we also provide an open-source implementation <sup>1</sup> to specify more details unambiguously.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x3.png)

Figure 3: Core architectures of CSA. It compresses the number of KV entries to 1 m \\frac{1}{m} times, and then applies DeepSeek Sparse Attention for further acceleration. Additionally, a small set of sliding window KV entries is combined with the selected compressed KV entries to enhance local fine-grained dependencies.

#### 2.3.1 Compressed Sparse Attention

The core architecture of CSA is illustrated in Figure 3, which first compresses the KV cache of each $m$ tokens into one entry, and then applies DeepSeek Sparse Attention for further acceleration.

##### Compressed Key-Value Entries.

Let $H\in\mathbb{R}^{n\times d}$ be a sequence of input hidden states, where $n$ is the sequence length and $d$ is the hidden size. CSA first computes two series of KV entries $C^{a},C^{b}\in\mathbb{R}^{n\times c}$ and their corresponding compression weights $Z^{a},Z^{b}\in\mathbb{R}^{n\times c}$, where $c$ is the head dimension:

$$
\displaystyle C^{a}
$$
 
$$
\displaystyle=H\cdot W^{aKV},\quad C^{b}=H\cdot W^{bKV},
$$
$$
\displaystyle Z^{a}
$$
 
$$
\displaystyle=H\cdot W^{aZ},\quad\penalty 10000\ \penalty 10000\ Z^{b}=H\cdot W^{bZ},
$$

where $W^{aKV},W^{bKV},W^{aZ},W^{bZ}\in\mathbb{R}^{d\times c}$ are trainable parameters. Next, each $m$ KV entries in $C^{a}$ and $C^{b}$ will be compressed into one entry according to their compression weights and learnable positional biases $B^{a},B^{b}\in\mathbb{R}^{m\times c}$, producing $C^{\text{Comp}}\in\mathbb{R}^{\frac{n}{m}\times c}$. Each compressed entry $C^{\text{Comp}}_{i}\in\mathbb{R}^{c}$ is computed by

$$
\displaystyle[S^{a}_{mi:m(i+1)-1};S^{b}_{m(i-1):mi-1}]
$$
 
$$
\displaystyle=\operatorname{Softmax}_{\text{row}}([Z^{a}_{mi:m(i+1)-1}+B^{a};Z^{b}_{m(i-1):mi-1}+B^{b}]),
$$
$$
\displaystyle C^{\text{Comp}}_{i}
$$
 
$$
\displaystyle=\sum_{j=mi}^{m(i+1)-1}S^{a}_{j}\odot C^{a}_{j}+\sum_{j=m(i-1)}^{mi-1}S^{b}_{j}\odot C^{b}_{j},
$$

where $\odot$ denotes the Hadamard product; $\operatorname{Softmax}_{\text{row}}(\cdot)$ denotes the softmax operation along the row dimension, which performs normalization across the total of $2m$ elements from both $Z^{a}$ and $Z^{b}$. When $i=0$, $Z^{b}_{m(i-1):mi-1}$ is padded with negative infinity and $C^{b}_{m(i-1):mi-1}$ is padded with zeros. Note that each $C^{\text{Comp}}_{i}$ is derived from $2m$ KV entries, but the indexes of $C^{b}$ used for $C^{\text{Comp}}_{i}$ and the indexes of $C^{a}$ used for $C^{\text{Comp}}_{i-1}$ are overlapped. Therefore, CSA in fact compresses the sequence length to $\frac{1}{m}$ times.

##### Lightning Indexer for Sparse Selection.

After obtaining the compressed KV entries $C^{\text{Comp}}$, CSA applies the DSA strategy to select top-k compressed KV entries for core attention. First, CSA performs the same compression operation used for $C^{\text{Comp}}$ to get compressed indexer keys $K^{\text{IComp}}\in\mathbb{R}^{\frac{n}{m}\times c^{I}}$, where $c^{I}$ is the indexer head dimension. Then, for a query token $t$, we produce the indexer queries $\{\mathbf{q}_{t,1}^{I};\mathbf{q}_{t,2}^{I};...;\mathbf{q}_{t,n_{h}^{I}}^{I}\}$ in a low-rank manner:

$$
\displaystyle\mathbf{c}_{t}^{Q}
$$
 
$$
\displaystyle=\mathbf{h}_{t}\cdot W^{DQ},
$$
$$
\displaystyle[\mathbf{q}_{t,1}^{I};\mathbf{q}_{t,2}^{I};...;\mathbf{q}_{t,n_{h}^{I}}^{I}]=\mathbf{q}_{t}^{I}
$$
 
$$
\displaystyle=\mathbf{c}_{t}^{Q}\cdot W^{IUQ},
$$

where $\mathbf{h}_{t}\in\mathbb{R}^{d}$ is the input hidden state of the query token $t$; $\mathbf{c}_{t}^{Q}\in\mathbb{R}^{d_{c}}$ is the compressed latent vector for queries; $d_{c}$ denotes the query compression dimension; $n_{h}^{I}$ denotes the number of indexer query heads; $W^{DQ}\in\mathbb{R}^{d\times d_{c}}$ and $W^{IUQ}\in\mathbb{R}^{d_{c}\times c^{I}n_{h}^{I}}$ are the down-projection and up-projection matrices for indexer queries, respectively. Next, the index score $I_{t,s}\in\mathbb{R}$ between the query token $t$ and a preceding compressed block $s$ ($s$ < $\operatorname{Floor}(\frac{t}{m})$) is computed by

$$
\displaystyle[w_{t,1}^{I};w_{t,2}^{I};...;w_{t,n_{h}^{I}}^{I}]=\mathbf{w}_{t}^{I}
$$
 
$$
\displaystyle=\mathbf{h}_{t}\cdot W^{w},
$$
$$
\displaystyle I_{t,s}
$$
 
$$
\displaystyle=\sum_{h=1}^{n_{h}^{I}}w_{t,h}^{I}\cdot\text{ReLU}\left(\mathbf{q}^{I}_{t,h}\cdot K^{\text{IComp}}_{s}\right),
$$

where $W^{w}\in\mathbb{R}^{d\times n_{h}^{I}}$ is a learnable matrix; $w_{t,h}^{I}\in\mathbb{R}$ is the weight of the $h$ -th indexer head. For a query token $t$, given its index scores $I_{t,:}$, we employ a top-k selector to selectively retain a subset of compressed KV entries $\mathcal{C}^{\text{SprsComp}}_{t}$ for subsequent core attention:

$$
\mathcal{C}^{\text{SprsComp}}_{t}=\left\{C^{\text{Comp}}_{s}\penalty 10000\ \Big|\penalty 10000\ I_{t,s}\in\operatorname{Top-k}(I_{t,:})\right\}.
$$

##### Shared Key-Value MQA.

After selecting the sparse KV entries, CSA then performs core attention in a Multi-Query Attention (MQA) \[mqa\] manner, where each compressed KV entry in $\mathcal{C}^{\text{SprsComp}}_{t}$ serves as both attention key and value. To be specific, for a query token $t$, we first produce attention queries $\{\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}\}$ from the compressed latent vector $\mathbf{c}_{t}^{Q}$:

$$
[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t}=\mathbf{c}_{t}^{Q}\cdot W^{UQ},
$$

where $n_{h}$ denotes the number of query heads; $W^{UQ}\in\mathbb{R}^{d_{c}\times cn_{h}}$ is the up-projection matrices for queries. Note that the latent query vector $\mathbf{c}_{t}^{Q}$ is shared with that used for the indexer queries. Next, we perform MQA on $\{\mathbf{q}_{t,i}\}$ and $\mathcal{C}^{\text{SprsComp}}_{t}$:

$$
\mathbf{o}_{t,i}=\operatorname{CoreAttn}\left(\texttt{query=}\mathbf{q}_{t,i},\texttt{key=}\mathcal{C}^{\text{SprsComp}}_{t},\texttt{value=}\mathcal{C}^{\text{SprsComp}}_{t}\right),
$$

where $\mathbf{o}_{t,i}\in\mathbb{R}^{c}$ is the core attention output of the $i$ -th head at the $t$ -th token; $\operatorname{CoreAttn}(\cdot)$ denotes the core attention operation.

##### Grouped Output Projection.

In the configuration of DeepSeek-V4, $cn_{h}$ is quite large. Therefore, directly projecting the outputs of the core attention operation $[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_{h}}]=\mathbf{o}_{t}\in\mathbb{R}^{cn_{h}}$ to a $d$ -dimensional hidden state will impose a substantial computational burden. To mitigate this cost, we design a grouped output projection strategy. To be specific, we first split $n_{h}$ outputs into $g$ groups, and then for each group of output $\mathbf{o}^{G}_{t,i}\in\mathbb{R}^{c\frac{n_{h}}{g}}$, we project it to a $d_{g}$ -dimensional intermediate output $\mathbf{o}^{G^{\prime}}_{t,i}\in\mathbb{R}^{d_{g}}$, where $d_{g}<c\frac{n_{h}}{g}$. Finally, we project the intermediate output $[\mathbf{o}^{G^{\prime}}_{t,1};\mathbf{o}^{G^{\prime}}_{t,2};...;\mathbf{o}^{G^{\prime}}_{t,g}]\in\mathbb{R}^{d_{g}g}$ to the final attention output $\mathbf{\hat{o}}_{t}\in\mathbb{R}^{d}$.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x4.png)

Figure 4: Core architectures of HCA. It performs heavier compression, where the KV entries of m ′ m^{\\prime} ( ≫ \\gg m ) tokens will be consolidated into one. Also, we additionally introduce a small set of sliding window KV entries to enhance local fine-grained dependencies.

#### 2.3.2 Heavily Compressed Attention

The core architecture of HCA is illustrated in Figure 4, which compresses the KV cache in a heavier manner, but does not employ sparse attention.

##### Compressed Key-Value Entries.

By and large, the compression strategy of HCA is similar to that of CSA, but employs a larger compression rate $m^{\prime}$ ($\gg m$) and does not perform overlapped compression. Let $H\in\mathbb{R}^{n\times d}$ be a sequence of input hidden states, HCA first computes the original KV entries $C\in\mathbb{R}^{n\times c}$ and their corresponding compression weights $Z\in\mathbb{R}^{n\times c}$:

$$
\displaystyle C
$$
 
$$
\displaystyle=H\cdot W^{KV},
$$
$$
\displaystyle Z
$$
 
$$
\displaystyle=H\cdot W^{Z},
$$

where $W^{KV},W^{Z}\in\mathbb{R}^{d\times c}$ are trainable parameters. Next, each $m^{\prime}$ KV entries in $C$ will be compressed into one according to the compression weights and learnable positional biases $B\in\mathbb{R}^{m^{\prime}\times c}$, producing $C^{\text{Comp}}\in\mathbb{R}^{\frac{n}{m^{\prime}}\times c}$. Each compressed entry $C^{\text{Comp}}_{i}\in\mathbb{R}^{c}$ is computed by

$$
\displaystyle S_{m^{\prime}i:m^{\prime}(i+1)-1}
$$
 
$$
\displaystyle=\operatorname{Softmax}_{\text{row}}(Z_{m^{\prime}i:m^{\prime}(i+1)-1}+B),
$$
$$
\displaystyle C^{\text{Comp}}_{i}
$$
 
$$
\displaystyle=\sum_{j=m^{\prime}i}^{m^{\prime}(i+1)-1}S_{j}\odot C_{j}.
$$

Through this compression operation, HCA compresses the sequence length to $\frac{1}{m^{\prime}}$ times.

##### Shared Key-Value MQA and Grouped Output Projection.

HCA also employs the shared KV MQA and grouped output projection strategies as CSA does. After the KV compression, for a query token $t$, HCA first produces attention queries $\{\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}\}$ in a low-rank manner:

$$
\displaystyle\mathbf{c}_{t}^{Q}
$$
 
$$
\displaystyle=\mathbf{h}_{t}\cdot W^{DQ},
$$
$$
\displaystyle[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t}
$$
 
$$
\displaystyle=\mathbf{c}_{t}^{Q}\cdot W^{UQ},
$$

where $\mathbf{h}_{t}\in\mathbb{R}^{d}$ is the input hidden state of the query token $t$; $n_{h}$ denotes the number of query heads; $W^{DQ}\in\mathbb{R}^{d\times d_{c}}$ and $W^{UQ}\in\mathbb{R}^{d_{c}\times cn_{h}}$ are the down-projection and up-projection matrices for queries, respectively. Next, we perform MQA on $\{\mathbf{q}_{t,i}\}$ and $C^{\text{Comp}}$:

$$
\mathbf{o}_{t,i}=\operatorname{CoreAttn}\left(\texttt{query=}\mathbf{q}_{t,i},\texttt{key=}C^{\text{Comp}},\texttt{value=}C^{\text{Comp}}\right),
$$

where $\mathbf{o}_{t,i}\in\mathbb{R}^{c}$ is the core attention output of the $i$ -th head at the $t$ -th token. Next, as CSA does, HCA splits $n_{h}$ outputs into $g$ groups, and for each group of output $\mathbf{o}^{G}_{t,i}\in\mathbb{R}^{c\frac{n_{h}}{g}}$, HCA projects it to a $d_{g}$ -dimensional intermediate output $\mathbf{o}^{G^{\prime}}_{t,i}\in\mathbb{R}^{d_{g}}$, where $d_{g}<c\frac{n_{h}}{g}$. Finally, HCA projects the intermediate output $[\mathbf{o}^{G^{\prime}}_{t,1};\mathbf{o}^{G^{\prime}}_{t,2};...;\mathbf{o}^{G^{\prime}}_{t,g}]\in\mathbb{R}^{d_{g}g}$ to the final attention output $\mathbf{\hat{o}}_{t}\in\mathbb{R}^{d}$.

#### 2.3.3 Other Details

In addition to the core architectures of CSA and HCA described above, our hybrid attention incorporates several other techniques. For writing clarity, we omit these additional techniques from the above introduction and will briefly describe them in this subsection. Also, this subsection focuses only on the core ideas of them and may omit some tiny details for simplicity. We encourage the readers to refer to our open-source implementation for unambiguous details.

##### Query and Key-Value Entry Normalization.

For both CSA and HCA, we perform an additional RMSNorm operation on each head of the queries and the only head of the compressed KV entries, just before the core attention operation. This normalization avoids exploding attention logits and may improve training stability.

##### Partial Rotary Positional Embedding.

For both CSA and HCA, we partially employ the Rotary Positional Embedding (RoPE) \[su2024roformer\] to the attention queries, KV entries, and the core attention outputs. To be specific, for each query vector and KV entry vector used in CSA and HCA, we apply RoPE to its last 64 dimensions. Since the KV entries serve as both attention keys and values, the naive core attention outputs $\{\mathbf{o}_{t,i}\}$ will carry absolute position embeddings, derived from the weighted sum of KV entries. As a countermeasure, we also apply RoPE with position $-i$ on the last 64 dimensions of each $\mathbf{o}_{t,i}$. In this way, the output of the core attention will also carry relative position embeddings — the contribution of each KV entry to the core attention outputs will also be related to the distance between the query and the KV entry.

##### Additional Branch of Sliding Window Attention.

In order to strictly preserve causality in CSA and HCA, each query attends to only preceding compressed KV blocks. Consequently, a query cannot access information from other tokens within its own compressed block. Meanwhile, recent tokens usually possess greater relevance to the query token in language modeling. For these reasons, we introduce a supplementary attention branch to both CSA and HCA in a sliding window manner, for better modeling of local dependencies. To be specific, for each query token, we additionally produce $n_{\text{win}}$ uncompressed KV entries corresponding to the recent $n_{\text{win}}$ tokens. In the core attention of CSA and HCA, these KV entries in the sliding window will be used along with the compressed KV entries.

##### Attention Sink.

In the core attention of CSA and HCA, we employ the trick of attention sink \[attn\_sink, gpt\_oss\]. To be specific, we set a series of learnable sink logits $\{z^{\prime}_{1},z^{\prime}_{2},...,z^{\prime}_{n_{h}}\}$. For the $h$ -th attention head, $\operatorname{Exp}(z^{\prime}_{h})$ will be added to the denominator of the attention score:

$$
s_{h,i,j}=\frac{\operatorname{Exp}(z_{h,i,j})}{\sum_{k}\operatorname{Exp}(z_{h,i,k})+\operatorname{Exp}(z^{\prime}_{h})},
$$

where $s_{h,i,j},z_{h,i,j}\in\mathbb{R}$ denote the attention score and attention logit of the $h$ -th attention head between the $i$ -th query token and the $j$ -th preceding token or compressed block. This technique allows each query head to adjust its total attention scores to be not equal to 1, and even to be near 0.

#### 2.3.4 Efficiency Discussion

Due to the employment of hybrid CSA and HCA, together with low-precision computation and storage, the attention module of DeepSeek-V4 series achieves remarkable efficiency in both attention FLOPs and KV cache size, especially in long-context scenarios. First, we adopt a mixed storage format for KV entries: BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions. This hybrid representation reduces the KV cache size by nearly half compared with pure BF16 storage. Second, attention computation within the lightning indexer is performed in FP4 precision, which accelerates the attention operation under extremely long contexts. Third, relative to DeepSeek-V3.2, a smaller attention top-k is chosen in DeepSeek-V4 series, thereby improving model efficiency on short- and medium-length texts. Finally, and most importantly, compressed attention and hybrid attention techniques substantially reduce both the KV cache size and the computational FLOPs.

Taking BF16 GQA8 \[ainslie2023gqa\] with a head dimension of 128 as the baseline — one of the common configurations of LLM attention — the KV cache size of DeepSeek-V4 series can be dramatically reduced to approximately $2\%$ times of that baseline in the 1M-context setting. Moreover, even when compared with DeepSeek-V3.2 \[dsv32\] — already an efficient baseline — DeepSeek-V4 series still exhibits substantial advantages in efficiency. A comparison of their inference FLOPs and KV cache size is provided in the right part of Figure 1.

Algorithm 1 Muon Optimizer for DeepSeek-V4

Learning rate $\eta$, momentum $\mu$, weight decay $\lambda$, update rescaling factor $\gamma$

for each training step $t$ do

  for each logically independent weight $W\in\mathbb{R}^{n\times m}$ do

    $G_{t}=\nabla_{W}\mathcal{L}_{t}(W_{t-1})$ $\triangleright$ Compute gradients

    $M_{t}=\mu M_{t-1}+G_{t}$ $\triangleright$ Accumulate momentum buffer

    $O^{\prime}_{t}=\operatorname{HybridNewtonSchulz}(\mu M_{t}+G_{t})$ $\triangleright$ Nesterov trick and hybrid Newton-Schulz

    $O_{t}=O^{\prime}_{t}\cdot\sqrt{\max(n,m)}\cdot\gamma$ $\triangleright$ Rescale the update RMS

    $W_{t}=W_{t-1}\cdot\left(1-\eta\lambda\right)-\eta O_{t}$ $\triangleright$ Perform weight decay and update

  end for

end for

### 2.4 Muon Optimizer

We employ the Muon \[muon, muon\_kimi\] optimizer for the majority of modules in DeepSeek-V4 series due to its faster convergence and improved training stability. The full algorithm of our Muon optimization is summarized in Algorithm 1.

##### Basic Configurations.

We maintain the AdamW \[adamW\] optimizer for the embedding module, the prediction head module, the static biases and gating factors of mHC modules, and the weights of all RMSNorm modules. All other modules are updated with Muon. Following muon\_kimi, we also apply weight decay to Muon parameters, use the Nesterov \[nesterov, muon\] trick, and rescale the Root Mean Square (RMS) of the update matrix for reutilization of our AdamW hyper-parameters. Different from them, we use hybrid Newton-Schulz iterations for orthogonalization.

##### Hybrid Newton-Schulz Iterations.

For a given matrix $M$, let its Singular Value Decomposition (SVD) be $M=U\Sigma V^{T}$. The Newton-Schulz iterations aim to approximately orthogonalize $M$ to be $UV^{T}$. Usually, $M$ will be first normalized as $M_{0}=M/||M||_{F}$ to ensure its maximum singular value does not exceed 1. Then, each Newton-Schulz iteration performs the following operation:

$$
M_{k}=aM_{k-1}+b(M_{k-1}M_{k-1}^{T})M_{k-1}+c(M_{k-1}M_{k-1}^{T})^{2}M_{k-1}.
$$

Our hybrid Newton-Schulz performs 10 iterations over two distinct stages. During the first 8 steps, we use coefficients $(a,b,c)=(3.4445,-4.7750,2.0315)$ to drive rapid convergence, bringing the singular values close to 1. In the final 2 steps, we switch to coefficients $(a,b,c)=(2,-1.5,0.5)$, which stabilize the singular values precisely at 1.

##### Avoiding Exploding Attention Logits.

The attention architecture of DeepSeek-V4 series allows us to directly apply RMSNorm on the attention queries and KV entries, which effectively prevents attention logits from exploding. Consequently, we do not employ the QK-Clip technique \[muon\_kimi\] in our Muon optimizer.

## 3 General Infrastructures

### 3.1 Fine-Grained Communication-Computation Overlap in Expert Parallelism

Mixture-of-Experts (MoE) can be accelerated via Expert Parallelism (EP). However, EP requires complex inter-node communication and imposes substantial demands on interconnect bandwidth and latency. To alleviate the communication bottleneck in EP and achieve higher end-to-end performance under lower interconnection bandwidth requirements, we propose a fine-grained EP scheme that fuses communication and computation into a single pipelined kernel for communication-computation overlapping.

##### Communication Latency Can Be Hidden.

The key insight of our EP scheme is that the communication latency can be effectively hidden beneath computation in MoE layers. As shown in Figure 5, in DeepSeek-V4 series, each MoE layer can be decomposed mainly into four stages: two communication-bound stages, *Dispatch* and *Combine*, and two computation-bound stages, *Linear-1* and *Linear-2*. Our profiling reveals that within a single MoE layer, the total time of communication is less than that of the computation. Therefore, after fusing communication and computation into a unified pipeline, computation remains the dominant bottleneck, implying that the system can tolerate lower interconnect bandwidth without degrading end-to-end performance.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x5.png)

Figure 5: Illustration of our EP scheme with related works. Comet \[ comet \] overlaps Dispatch with Linear-1, and Linear-2 with Combine, separately. Our EP scheme achieves a finer-grained overlapping by splitting and scheduling experts into waves. The theoretical speedup is evaluated in the configuration of the DeepSeek-V4-Flash architecture.

##### Fine-Grained EP Scheme.

To further lower the interconnect bandwidth requirement and amplify the benefits of overlapping, we introduce a finer-grained expert partitioning scheme. Inspired by many related works \[aimuyo2025FlashMoE, comet\], we split and schedule the experts into *waves*. Each wave consists of a small portion of experts. As soon as all experts within the wave have completed their communication, computation can commence immediately without waiting for other experts. In steady state, computation of current wave, token transfer for the next wave, and result sending of completed experts all proceed concurrently, as demonstrated in Figure 5. This forms a fine-grained pipeline among experts, keeping both computation and communication continuous throughout the wave. The wave-based scheduling speeds up the performance on extreme cases such as Reinforcement Learning (RL) rollout, which usually encounters long-tail small batches.

##### Performance and Open-Sourced Mega-Kernel.

We validated the fine-grained EP scheme on both NVIDIA GPUs and HUAWEI Ascend NPUs platforms. Compared against strong non-fused baselines, it achieves $1.50\sim 1.73\times$ speedup for general inference workloads, and up to $1.96\times$ for latency-sensitive scenarios such as RL rollouts and high-speed agent serving. We have open-sourced the CUDA-based mega-kernel implementation named MegaMoE <sup>2</sup> as a component of DeepGEMM.

##### Observations and Proposals.

- Computation-Communication Ratio. Full communication-computation overlap hinges on the computation-communication ratio, rather than the bandwidth solely. Denoting peak compute throughput as $C$ and interconnect bandwidth as $B$, communication can be fully hidden when $C/B\leqslant V_{\text{comp}}/V_{\text{comm}}$, where $V_{\text{comp}}$ denotes the computation volume and $V_{\text{comm}}$ refers to the communication volume. For DeepSeek-V4-Pro, where each token-expert pair requires $6hd$ FLOPs (SwiGLU gate, up, and down projections) but only $3h$ bytes of communication (FP8 Dispatch + BF16 Combine), this simplifies to:
	$$
	\frac{C}{B}\leqslant 2d=6144\;\mathrm{FLOPs/Byte}.
	$$
	That is, each GBps of interconnect bandwidth suffices to hide the communication for 6.1 TFLOP/s of compute. Once bandwidth meets this threshold, it ceases to be the bottleneck, and devoting additional silicon area to further bandwidth brings diminishing returns. We encourage future hardware designs to target such balance points rather than scale bandwidth unconditionally.
- Power Budget. Extreme kernel fusion drives compute, memory, and network to high load simultaneously, making power throttling a key performance limiter. We suggest that future hardware designs provide sufficient power headroom for such fully concurrent workloads.
- Communication Primitives. In the dispatch stage, we adopt a pull-based approach where each GPU actively reads activations from remote GPUs, avoiding the high notification latency that fine-grained push entails. Future hardware with lower-latency cross-GPU signaling would make push viable and enable more natural communication patterns.
- Activation Function. We propose replacing SwiGLU with a low-cost element-wise activation that involves no exponential or division operations. This directly reduces the overhead of post-GEMM processing, preventing the GEMM pipeline from being stalled by activation function computation, thereby enhancing overall computational throughput and resource utilization.

### 3.2 Flexible and Efficient Kernel Development with TileLang

In practice, our elaborate model architecture would have resulted in hundreds of fine-grained Torch ATen operators. We adopt TileLang \[wangtilelang\] to develop a set of fused kernels to replace the vast majority of them, delivering optimal performance with minimal effort. It also allows us to quickly prototype operators like attention variants during validation. These kernels play critical roles in model architecture development, large‑scale training, and ultimately production deployment of inference services. As a Domain-Specific Language (DSL), TileLang balances development productivity with runtime efficiency, enabling rapid development while supporting deep, iterative optimizations within the same codebase. Additionally, we collaborate closely with the TileLang community to foster a more agile, efficient, and stable kernel development workflow.

##### Reducing Invocation Overhead with Host Codegen.

As accelerators continue to grow in performance, CPU-side orchestration overhead becomes increasingly prominent. For small, highly optimized kernels, such fixed host overhead can easily cap utilization and throughput. A common source of this overhead is that host-side logic, such as runtime contract checks, is typically written in Python for flexibility and thus incurs a fixed per-invocation cost.

We mitigate this overhead with Host Codegen, which moves most host-side logic into generated host code. Specifically, we first co-generate the device kernel and a lightweight host launcher at the IR (Intermediate Representation) level, embedding the necessary metadata—such as data types, rank/shape constraints, and stride/layout assumptions—parsed from the language frontend. The launcher is then lowered to the host source code built on top of the TVM-FFI \[tvm\] framework, whose compact calling convention and zero-copy tensor interop together minimize host-side overhead. At runtime, this generated host code performs validation and argument marshaling, shifting all per‑invocation checks out of the Python execution path. Our measurements show that CPU-side validation overhead drops from tens or hundreds of microseconds to less than one microsecond per invocation.

##### SMT-Solver-Assisted Formal Integer Analysis.

TileLang kernels involve complex tensor index arithmetic that requires strong formal integer analysis. During compilation passes such as layout inference, memory hazard detection, and bound analysis, the compiler must verify whether integer expressions satisfy specific properties to enable the corresponding optimizations. Therefore, stronger formal analysis capabilities can unlock more advanced and complex optimization opportunities.

To this end, we integrate the Z3 SMT solver \[demoura2008z3\] into TileLang’s algebraic system, providing formal analysis capability for most integer expressions in tensor programs. We strike a balance between computational overhead and formal expressiveness by translating TileLang’s integer expressions into Z3’s quantifier-free non-linear integer arithmetic (QF\_NIA). Based on Integer Linear Programming (ILP) solvers, QF\_NIA seamlessly resolves standard linear integer expressions common in kernels. Furthermore, its inherent non-linear reasoning capacity effectively addresses advanced challenges like vectorization over variable tensor shapes. Under reasonable resource limits, Z3 elevates overall optimization performance while restricting compilation time overhead to just a few seconds. The impact is substantial across multiple passes, including vectorization, barrier insertion, and code simplification.

##### Numerical Precision and Bitwise Reproducibility.

In production settings, numerical correctness and reproducibility are as critical as raw throughput. We therefore prioritize accuracy by default: fast‑math optimizations are disabled at the compiler level, and precision‑affecting approximations are provided only as explicit, opt-in frontend operators (e.g., T.\_\_exp, T.\_\_log, and T.\_\_sin). Conversely, when strict IEEE-754 semantics are required, TileLang provides IEEE-compliant intrinsics with explicit rounding modes (e.g., T.ieee\_fsqrt, T.ieee\_fdiv, and T.ieee\_add), enabling developers to precisely specify numerical behavior.

We also target bitwise reproducibility for validating kernels against hand‑written CUDA baselines. We align TileLang’s algebraic simplification and lowering rules with mainstream CUDA toolchains (e.g., NVCC) to avoid transformations that introduce unintended bit‑level differences. Layout annotations (e.g., T.annotate\_layout) further allow users to pin down layout-dependent lowering decisions, keeping evaluation and accumulation order consistent with the reference CUDA implementation and thus enabling bit-identical outputs when desired.

Our evaluation shows that these accuracy- and reproducibility-oriented design choices do not sacrifice performance: under conservative defaults, TileLang kernels remain competitive, while exposing knobs to selectively relax numerical constraints for higher speed.

### 3.3 High-Performance Batch-Invariant and Deterministic Kernel Libraries

To enable efficient training and inference, we develop a comprehensive set of high-performance computational kernels. Beyond basic functionalities and maximizing hardware utilization, another pivotal design goal is to ensure training reproducibility and bitwise alignment among pre-training, post-training, and inference pipelines. Therefore, we implement end-to-end, bitwise batch-invariant, and deterministic kernels with minimal performance overhead. These kernels are helpful for debugging, stability analysis, and consistent post-training behavior.

##### Batch Invariance.

Batch invariance ensures that the output of any given token remains bitwise identical, regardless of its position within a batch. To implement batch invariance, the primary challenges are listed as follows:

- Attention. To achieve batch invariance, we cannot use the split-KV method \[flash\_decoding\], which distributes the attention computation for a single sequence across multiple Stream Multiprocessors (SMs) to balance the load of SMs. However, abandoning this technique will lead to severe wave-quantization problems <sup>3</sup>, which can adversely affect GPU utilization. To address this, we develop a dual-kernel strategy for batch-invariant decoding. The first kernel computes the attention output for an entire sequence within a single SM, ensuring high throughput for fully occupied waves. The second kernel, to minimize the latency of the final partially-filled wave and thus alleviate wave-quantization, uses multiple SMs for a single sequence. For the bitwise identity of these two kernels, we carefully design the calculation path of the second kernel to ensure its accumulation order is the same as that of the first kernel. Additionally, the second kernel utilizes distributed shared memory <sup>4</sup> within thread-block clusters, enabling high-speed data exchange across SMs. This dual-kernel method effectively confines the overhead of batch-invariant decoding to be negligible.
- Matrix Multiplication. Traditional cuBLAS library \[cublas\] cannot achieve batch invariance. Therefore, we replace it end-to-end with DeepGEMM \[deepgemm2025\]. Furthermore, for very small batch sizes, conventional implementation usually employs split-k \[stream\_k\] techniques to improve performance. Unfortunately, split-k techniques cannot guarantee batch invariance, a pivotal feature in DeepSeek-V4. Therefore, we abandon split-k in most scenarios, which, however, may cause performance degradation. To address this, we introduce a set of optimizations that enable our implementation of matrix multiplication to match or even surpass the performance of standard split-k in most major scenarios.

##### Determinism.

Deterministic training is highly beneficial for debugging hardware or software issues. Moreover, when training exhibits anomalies such as loss spikes, determinism enables researchers to more easily pinpoint numerical causes and further refine the model design. Non-determinism in training typically stems from non-deterministic accumulation order, often due to the use of atomic addition instructions. This issue primarily occurs during the backward pass, notably at the following parts:

- Attention Backward. In conventional implementations of backward propagation for sparse attention, we use atomicAdd to accumulate gradients for the KV tokens. This introduces non-determinism due to the non-associativity of floating-point addition. To address this problem, we allocate separate accumulation buffers for each SM, followed by a global deterministic summation across all buffers.
- MoE Backward. When multiple SMs from different ranks concurrently write data to the same buffer on a receiving rank, negotiating writing positions also introduces non-determinism. To resolve this, we design a token order pre-processing mechanism within each single rank, combined with buffer isolation across multiple ranks. This strategy ensures determinism of both the send results of expert parallelism and the accumulation order in the MoE backward pass.
- Matrix Multiplication in mHC. mHC involves a matrix multiplication with an output dimension of only 24. For very small batch sizes, we are compelled to use the split-k \[stream\_k\] algorithm, whose naive implementation will cause non-determinism. To overcome this, we output each split part separately and perform a deterministic reduction in a subsequent kernel, thereby preserving both performance and determinism.

### 3.4 Training Framework

Our training framework is built upon the scalable and efficient infrastructure developed for DeepSeek-V3 \[dsv3\]. In training DeepSeek-V4, we inherit this robust foundation while introducing several key innovations to accommodate its novel architectural components — specifically the Muon optimizer, mHC, and the hybrid attention mechanism — while maintaining high training efficiency and stability.

#### 3.4.1 Efficient Implementation of Muon

The Muon optimizer requires the full gradient matrix to compute parameter updates, which presents a challenge when combined with the Zero Redundancy Optimizer (ZeRO) \[zero\]. Traditional ZeRO is designed for element-wise optimizers like AdamW, where a single parameter matrix can be partitioned and updated across multiple ranks. To address this conflict, we design a hybrid strategy of ZeRO bucket assignment for Muon.

For dense parameters, we limit the maximum size of ZeRO parallelism and employ a knapsack algorithm to assign parameter matrices to these ranks, ensuring each rank manages a roughly balanced load. The bucket on each rank is padded to match the size of the largest bucket across ranks, facilitating efficient reduce-scatter operations. This padding typically incurs less than 10% memory overhead in our setup, where each rank manages no more than five parameter matrices. When the overall size of data parallelism exceeds the limit for ZeRO, we compute the Muon update redundantly across the extra data-parallel groups, trading computation for reduced total bucket memory.

For MoE parameters, we optimize each expert independently. We first flatten all down projection matrices in SwiGLU \[shazeer2020glu\] of all experts across all layers, followed by flattened up projection matrices and gate matrices. Then, we pad the flattened vector to ensure we can evenly distribute this vector across all ranks without splitting any logically independent matrix. Given the large number of experts, we do not impose a limit of ZeRO parallelism for MoE parameters, and the padding overhead is also negligible.

Additionally, on each rank, consecutive parameters of identical shape will be automatically merged, enabling batched execution of the Newton-Schulz iterations for better hardware utilization. Furthermore, we observe that the Newton-Schulz iterations in Muon remain stable when computed with BF16 matrix multiplications. Leveraging this, we further quantize, in a stochastic rounding manner, the MoE gradients to be synchronized across data-parallel ranks to the BF16 precision, halving the communication volume. To avoid accumulation errors introduced by low-precision adders, we replace conventional tree- or ring-based reduce-scatter collectives with a two-phase approach. First, an all-to-all operation exchanges local gradients across ranks, and then each rank performs a local sum in FP32. This design maintains numerical robustness.

#### 3.4.2 Cost-Effective and Memory-Efficient Implementation of mHC

The introduction of mHC increases both activation memory consumption and communication volume between pipeline stages, compared with conventional residual connections. To mitigate these costs, we implement several optimization strategies.

Firstly, we carefully design and implement fused kernels of mHC for both training and inference. Secondly, we introduce a recomputation strategy that selectively checkpoints intermediate tensors. Specifically, we recompute most hidden states between layers and all normalized layer inputs, while avoiding recomputation of compute-intensive operations. This achieves a balance between memory saving and computational overhead. Thirdly, we adjust the DualPipe 1F1B overlapping scheme to accommodate the increased pipeline communication and enable concurrent execution of some operations in mHC.

Collectively, these optimizations constrain the wall-time overhead of mHC to only 6.7% of the overlapped 1F1B pipeline stage. More details of the engineering optimization can be found in the dedicated mHC paper \[mhc\].

#### 3.4.3 Contextual Parallelism for Long-Context Attention

Conventional Context Parallelism (CP) partitions the sequence dimension, with each rank maintaining contiguous $s$ tokens. This introduces two challenges to our compressed attention mechanisms (i.e., CSA and HCA). On the one hand, training samples are packed from multiple sequences, and each sequence is compressed independently by a factor of $m$ (or $m^{\prime}$), with any trailing tokens fewer than $m$ being discarded. Consequently, the compressed KV lengths are typically less than $\frac{s}{m}$ and vary across ranks. On the other hand, the compression requires $m$ consecutive KV entries, which may straddle the boundary between two neighboring CP ranks.

To address these challenges, we design a two-stage communication approach. In the first stage, each rank $i$ sends its last $m$ uncompressed KV entries to rank $i+1$. Then, rank $i+1$ compresses some of these received entries together with its local $s$ uncompressed KV entries, producing a fixed length of $\frac{s}{m}+1$ compressed entries, in which exist some padding entries. In the second stage, an all-gather operation across all CP ranks collects the locally compressed KV entries. Then, a fused select-and-pad operator reorganizes them into the full set of compressed KV entries with a total length of $\texttt{cp\_size}\cdot\frac{s}{m}$. Any padding entries are placed at the tail. For HCA and the indexer in CSA, the visible range of compressed KV entries for each query token can be precomputed by rules. For the sparse attention in CSA, the top- $k$ selector explicitly specifies the indices of visible compressed KV entries for each query.

#### 3.4.4 Extended Automatic Differentiation for Flexible Activation Checkpointing

Conventional activation checkpointing implementations operate at the granularity of an entire module, deciding whether to retain or recompute its output activations during the backward pass. This coarse granularity often leads to suboptimal trade-offs between recomputation cost and activation memory footprint. An alternative approach is to manually implement the forward and backward logic of an entire layer, explicitly managing tensor checkpointing states. While enabling fine-grained control, this method loses the convenience of the automatic differentiation framework, substantially increasing development complexity.

To achieve fine-grained control without sacrificing programming efficiency, we implement a tensor-level activation checkpointing mechanism with automatic differentiation support. With this mechanism, developers only need to implement the forward pass and selectively annotate individual tensors for automatic checkpointing and recomputation. Our framework leverages TorchFX \[torchfx\] to trace the full computation graph. For each annotated tensor, it performs a backward traversal to identify the minimal subgraph required for its recomputation. We define these minimal subgraphs as recomputation graphs and insert them into the backward logic just before the corresponding gradient computation.

Compared with the manual implementation, this design introduces no additional overhead during training. Recomputation in this framework is implemented by directly freeing the GPU memory of the annotated tensor and reusing the storage pointer from the recomputed tensor, without any GPU memory copy. Furthermore, since graph tracing executes the model concretely, we can track the underlying storage pointer of each tensor, which enables automatic deduplication of recomputation for tensors that share storage (e.g., the input and output of a reshape operation). This relieves developers from reasoning about low-level memory details when annotating recomputation.

### 3.5 Inference Framework

Our inference framework largely inherits from that of DeepSeek-V3, with some differences in KV Cache management.

#### 3.5.1 KV Cache Structure and Management

To efficiently manage the heterogeneous KV caches arising from the hybrid attention mechanism in DeepSeek-V4, we design a customized KV cache layout. The layout is illustrated in Figure 6, and we will elaborate on it in detail as follows.

##### Heterogeneous KV Entries in DeepSeek-V4.

The hybrid attention mechanism in DeepSeek-V4 series introduces multiple types of KV entries with different Key-Value (KV) cache sizes and update rules. The lightning indexer for sparse selection introduces additional dimensions into the KV cache that possess embedding sizes distinct from those in the primary attention. The compression techniques employed in CSA and HCA reduce the sequence length by factors of $\frac{1}{m}$ and $\frac{1}{m^{\prime}}$, respectively, thereby decreasing the overall KV cache size. As a result, KV cache sizes vary across different layers. Furthermore, Sliding Window Attention (SWA) layers also operate with distinct KV cache sizes, as well as separate cache hit and eviction policies. In the compression branch, one KV entry is generated for every $m$ tokens. When the number of remaining tokens is insufficient for compression, all pending tokens and their associated hidden states must be retained in a buffer until the compression operation can be executed. These buffered tokens represent a sequence state determined by positional context and are also managed within the KV cache framework.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x6.png)

Figure 6: Illustration of the KV cache Layout for DeepSeek-V4. The KV cache is organized into two primary components: a classical KV cache for CSA/HCA, and a state cache for SWA and unready-for-compression tokens in CSA/HCA. In the state cache, each request is assigned a fixed-size cache block. Within this block, the SWA segment stores the KV entries corresponding to the most recent n win n\_{\\text{win}} tokens, while the CSA/HCA segment stores uncompressed tail states that are not yet ready for compression. In the classical KV cache, we allocate multiple blocks per request. Each cache block covers lcm ( m, ′ ) \\mathrm{lcm}(m,m^{\\prime}) original tokens, producing k 1 = k\_{1}=\\frac{\\mathrm{lcm}(m,m^{\\prime})}{m} CSA compressed tokens and 2 k\_{2}=\\frac{\\mathrm{lcm}(m,m^{\\prime})}{m^{\\prime}} HCA compressed tokens.

##### Challenges in Managing Hybrid Attention KV Cache.

The hybrid attention mechanism violates fundamental assumptions behind PagedAttention and its variants. Although recent hybrid KV cache managing algorithms (e.g., Jenga \[jenga\], Hymba \[dong2025hymba\]) target general hybrid attention models or specific structures, two principal obstacles prevent consolidating KV caches across all layers under the PagedAttention framework:

- Diverse cache policies, such as those used in Sliding Window Attention.
- Constraints imposed by high-performance attention kernels, including alignment requirements.

For efficient KV cache management of DeepSeek-V4, we design corresponding strategies to overcome these two challenges.

##### State Cache for SWA and Uncompressed Tail Tokens.

To address the first obstacle, we adopt an alternative cache management mechanism. Since SWA is designed to enhance performance under a limited KV cache size, it is reasonable to treat it, along with the uncompressed tail tokens from the compression branch, as a state-space model. The corresponding KV cache can thus be regarded as a sequence-specific state that depends solely on the current position. Accordingly, we pre-allocate a fixed- and limited-size pool of state caches, and dynamically assign it to each sequence.

##### Sparse Attention Kernel Co-Design.

Regarding the second obstacle, conventional high-performance attention kernels typically assume a fixed number $B$ of tokens per block to optimize performance, corresponding to $B\cdot m$ original tokens in CSA and $B\cdot m^{\prime}$ in HCA. Through employing a high-performance sparse-attention kernel, different layers can accommodate variable tokens per block without performance degradation. Achieving this requires co-designing the KV cache layout and the sparse attention kernel. For instance, padding blocks to align with cache lines can improve performance. Thus, for CSA with compression ratio $m$ and HCA with ratio $m^{\prime}$, the number of original tokens per block can be any multiple of $\operatorname{lcm}(m,m^{\prime})$, the least common multiple of these two compression ratios.

#### 3.5.2 On-Disk KV Cache Storage

When serving DeepSeek-V4, we leverage an on-disk KV cache storage mechanism to eliminate repeated prefilling for shared-prefix requests. For the compressed KV entries in CSA/HCA and the uncompressed KV entries in Sliding Window Attention (SWA), we design separate solutions for storage management.

For CSA and HCA, we simply store all of the compressed KV entries to the disk. When a request hits a stored prefix, we read and reuse the compressed KV entries corresponding to the prefix, until the last complete compression block. Specially, for prefix tokens in the tail incomplete block, we still need to recompute them to restore the uncompressed KV entries, as uncompressed KV entries in CSA and HCA are not stored.

For the SWA KV entries, since they are not compressed and exist in every layer, their volume is approximately 8 times larger than the compressed CSA and HCA KV entries. To handle these large SWA KV entries efficiently, we propose and implement three distinct strategies for managing on-disk SWA KV entries, each offering a different trade-off between storage overhead and computational redundancy:

- Full SWA Caching. This strategy stores the complete SWA KV entries for all tokens, ensuring computational zero-redundancy. Under this strategy, the SWA KV entries of the hitting prefix can be reconstructed by just reading the on-disk cache of the last $n_{\text{win}}$ tokens within that prefix. Despite computational zero-redundancy, this strategy is inefficient for modern SSD-based storage systems — only a small subset of the stored SWA KV cache will be accessed for each hitting request, which leads to an unbalanced write-intensive access pattern.
- Periodic Checkpointing. This strategy checkpoints SWA KV entries of the last $n_{\text{win}}$ tokens within every $p$ tokens, where $p$ is a tunable parameter. For a hitting prefix, we load the most recent checkpointed state, and then recompute the remaining tail tokens. Through tuning $p$, this strategy enables an on-demand trade-off between storage and computation.
- Zero SWA Caching. This strategy does not store any SWA KV entries. For a hitting prefix, we need to perform more recomputation to restore the SWA KV entries. To be specific, in each attention layer, the SWA KV entry of each token depends on the SWA KV entries of only the most recent $n_{\text{win}}$ tokens from the previous layer. Therefore, leveraging cached CSA and HCA KV entries, recomputing the last $n_{\text{win}}\cdot L$ tokens is enough to restore the last $n_{\text{win}}$ SWA KV entries for an $L$ -layer model.

Depending on specific deployment scenarios, we select the most suitable strategy to achieve the desired trade-off between storage and computation.

## 4 Pre-Training

### 4.1 Data Construction

On top of the pre-training data of DeepSeek-V3, we endeavor to construct a more diverse and higher-quality training corpus with longer effective contexts. We continually refine our data construction pipelines. For web-sourced data, we implement filtering strategies to remove batched auto-generated and templated content, thereby mitigating the risk of model collapse \[zhu2024synthesize\]. Mathematical and programming corpora still remain core components of our training data, and we further enhance the coding capabilities of DeepSeek-V4 series by incorporating agentic data during the mid-training phase. For multilingual data, we build a larger corpus for DeepSeek-V4, improving its capture of long-tail knowledge across different cultures. For DeepSeek-V4, we place a particular emphasis on long-document data curation, prioritizing scientific papers, technical reports, and other materials that reflect unique academic values. Combining all the above, our pre-training corpus comprises more than 32T tokens, containing mathematical contents, codes, web pages, long documents, and other high-quality categories.

For pre-training data, we largely follow the same pre-processing strategies of DeepSeek-V3. For tokenization, on top of the DeepSeek-V3 tokenizer, we introduce a few special tokens for context construction, and still remain the vocabulary size to be 128K. We also inherit the token-splitting \[dsv3\] and Fill-in-Middle (FIM) \[dscodervii\] strategies from DeepSeek-V3. Inspired by Ding2024FewerTI, we pack documents from different sources into appropriate sequences to minimize sample truncation. Different from DeepSeek-V3, we employ sample-level attention masking during pre-training.

### 4.2 Pre-Training Setups

#### 4.2.1 Model Setups

##### DeepSeek-V4-Flash.

We set the number of Transformer layers to 43 and the hidden dimension $d$ to 4096. For the first two layers, we use pure sliding window attention. For the subsequent layers, CSA and HCA are used in an interleaved manner. For CSA, we set the compression rate $m$ to 4, the number of indexer query heads $n_{h}^{I}$ to 64, the indexer head dimension $c^{I}$ to 128, and the number of KV entries selected for sparse attention (i.e., attention top-k) to 512. For HCA, we set the compression rate $m^{\prime}$ to 128. For both CSA and HCA, we set the number of query heads $n_{h}$ to 64, the head dimension $c$ to 512, and the query compression dimension $d_{c}$ to 1024. The number of output projection groups $g$ is set to 8, and the dimension of each intermediate attention output $d_{g}$ is set to 1024. For the additional branch of sliding window attention, the window size $n_{\text{win}}$ is set to 128. We employ MoE layers in all Transformer blocks, but use the Hash routing strategy for the first 3 MoE layers. Each MoE layer consists of 1 shared expert and 256 routed experts, where the intermediate hidden dimension of each expert is 2048. Among the routed experts, 6 experts will be activated for each token. The multi-token prediction depth is set to 1. As for mHC, the expansion factor $n_{\text{hc}}$ is set to 4, and the number of Sinkhorn-Knopp iterations $t_{\text{max}}$ is set to 20. Under this configuration, DeepSeek-V4-Flash comprises 284B total parameters, of which 13B are activated for each token.

##### DeepSeek-V4-Pro.

We set the number of Transformer layers to 61 and the hidden dimension $d$ to 7168. For the first two layers, we use HCA. For the subsequent layers, CSA and HCA are used in an interleaved manner. For CSA, we set the compression rate $m$ to 4, the number of indexer query heads $n_{h}^{I}$ to 64, the indexer head dimension $c^{I}$ to 128, and the number of KV entries selected for sparse attention (i.e., attention top-k) to 1024. For HCA, we set the compression rate $m^{\prime}$ to 128. For both CSA and HCA, we set the number of query heads $n_{h}$ to 128, the head dimension $c$ to 512, and the query compression dimension $d_{c}$ to 1536. The number of output projection groups $g$ is set to 16, and the dimension of each intermediate attention output $d_{g}$ is set to 1024. For the additional branch of sliding window attention, the window size $n_{\text{win}}$ is set to 128. We employ MoE layers in all Transformer blocks, but use the Hash routing strategy for the first 3 MoE layers. Each MoE layer consists of 1 shared expert and 384 routed experts, where the intermediate hidden dimension of each expert is 3072. Among the routed experts, 6 experts will be activated for each token. The multi-token prediction depth is set to 1. As for mHC, the expansion factor $n_{\text{hc}}$ is set to 4, and the number of Sinkhorn-Knopp iterations $t_{\text{max}}$ is set to 20. Under this configuration, DeepSeek-V4-Pro comprises 1.6T total parameters, of which 49B are activated for each token.

#### 4.2.2 Training Setups

##### DeepSeek-V4-Flash.

We employ the Muon optimizer \[muon, muon\_kimi\] for the majority of parameters, but use the AdamW optimizer \[adamW\] for the embedding module, the prediction head module, and the weights of all RMSNorm modules. For AdamW, we set its hyper-parameters to $\beta_{1}=0.9$, $\beta_{2}=0.95$, $\varepsilon=10^{-20}$, and $\mathrm{weight\_decay}=0.1$. For Muon, we set the momentum to 0.95 and the weight decay to 0.1, and rescale the RMS of each update matrix to 0.18 for reutilization of the AdamW learning rate. We train DeepSeek-V4-Flash on 32T tokens, and as in DeepSeek-V3, we also employ a batch size scheduling strategy that increases the batch size (in tokens) from a small size to 75.5M and then keeps it at 75.5M during most of the training. The learning rate is linearly warmed up in the first 2000 steps, maintained at $2.7\times 10^{-4}$ for most of the training. Near the end of the training, we finally decay the learning rate to $2.7\times 10^{-5}$ following a cosine schedule. The training starts with a sequence length of 4K, and we gradually extend the training sequence length to 16K, 64K, and 1M. As for the setups of sparse attention, we first warmup the model with dense attention for the first 1T tokens, and introduce sparse attention at the sequence length of 64K and keep sparse attention during the rest of the training. When introducing attention sparsity, we first set a short stage to warm up the lightning indexer in CSA, and then train the model with sparse attention for most of the training. For auxiliary-loss-free load balancing, we set the bias update speed to 0.001. For the balance loss, we set its loss weight to 0.0001 to avoid extreme imbalance within single sequences. The MTP loss weight is set to 0.3 for most of the training, and to 0.1 upon the start of learning rate decay.

##### DeepSeek-V4-Pro.

Except for specific values of hyper-parameters, the training setup of DeepSeek-V4-Pro is largely consistent with that of DeepSeek-V4-Flash. We employ the Muon optimizer for the majority of parameters, but use the AdamW optimizer for the embedding module, the prediction head module, and the weights of all RMSNorm modules. The hyper-parameters of AdamW and Muon are the same as those of DeepSeek-V4-Flash. We train DeepSeek-V4-Pro on 33T tokens, and also employ a batch size scheduling strategy, with the maximum batch size being 94.4M tokens. The learning rate scheduling strategy is largely the same as that of DeepSeek-V4-Flash, but the peak learning rate is set to $2.0\times 10^{-4}$ and the end learning rate is set to $2.0\times 10^{-5}$. The training also starts with a sequence length of 4K, and the length is gradually extended to 16K, 64K, and 1M. Compared with DeepSeek-V4-Flash, DeepSeek-V4-Pro starts with a longer stage of dense attention, and the strategy of introducing sparse attention is the same as DeepSeek-V4-Flash, following a two-stage training method. For auxiliary-loss-free load balancing, we set the bias update speed to 0.001. For the balance loss, we set its loss weight to 0.0001 to avoid extreme imbalance within single sequences. The MTP loss weight is set to 0.3 for most of the training, and to 0.1 upon the start of learning rate decay.

#### 4.2.3 Mitigating Training Instability

Training trillion-parameter MoE models presents significant stability challenges, and DeepSeek-V4 series are no exception. We encountered notable instability challenges during training. While simple rollbacks could temporarily restore the training state, they proved inadequate as a long-term solution because they do not prevent the recurrence of loss spikes. Empirically, we identified that the occurrence of spikes is consistently tied to outliers in the MoE layers, and the routing mechanism itself appears to exacerbate the emergence of these outliers. Therefore, we sought to tackle this issue from two dimensions: breaking the vicious cycle induced by routing, and directly suppressing anomalous values. Fortunately, we discovered two practical techniques that effectively maintain training stability. Although a comprehensive theoretical understanding of their underlying mechanisms remains an open question for now, we are sharing them openly to foster further exploration by the community.

##### Anticipatory Routing.

We found that decoupling the synchronous updates of the backbone network and the routing network significantly improves training stability. Consequently, at step $t$, we use the current network parameters $\theta_{t}$ for feature computation, but the routing indices are computed and applied using the historical network parameters $\theta_{t-\Delta t}$. In practice, to circumvent the overhead of loading model parameters twice, we fetch the data for step $t$ in advance at step $t-\Delta t$. We "anticipatorily" compute and cache the routing indices to be used later at step $t$, which is why we name this approach Anticipatory Routing. We also heavily optimized this at the infrastructure level. First, given that pre-computing the routing indices only requires a single forward pass over the data, we carefully orchestrated the pipeline execution and the overlapping of computation with Expert Parallelism (EP) communication, successfully bounding the additional wall-clock time overhead of Anticipatory Routing to approximately 20%. Second, we introduced an automatic detection mechanism that triggers a short rollback and activates Anticipatory Routing exclusively when a loss spike occurs; after operating in this mode for a certain period, the system reverts to standard training. Ultimately, this dynamic application allows us to avert loss spikes with negligible overall additional training overhead, all without compromising model performance.

##### SwiGLU Clamping.

In previous literature \[bello\*2017neural, gemma2\], clamping has been explicitly utilized to constrain numerical ranges, thereby enhancing training stability. In our actual training runs, we empirically found that applying SwiGLU clamping \[gpt\_oss\] effectively eliminates outliers and substantially aids in stabilizing the training process, without compromising performance. Throughout the training of both DeepSeek-V4-Flash and DeepSeek-V4-Pro, we clamped the linear component of SwiGLU to the range of $[-10,10]$, while capping the upper bound of the gate component at $10$.

### 4.3 Evaluations

#### 4.3.1 Evaluation Benchmarks

For the evaluation of the base models, we consider benchmarks spanning four key dimensions: world knowledge, language understanding and reasoning, coding and mathematics, and long-context processing.

World knowledge benchmarks include AGIEval \[agieval\], C-Eval \[ceval\], CMMLU \[cmmlu\] MMLU \[mmlu\], MMLU-Redux \[mmlu\_redux\], MMLU-Pro \[mmlu\_pro\], MMMLU \[mmmlu\], MultiLoKo \[multiloko\], Simple-QA verified \[haas2025simpleqa\], SuperGPQA \[du2025supergpqa\], FACTS Parametric \[cheng2025facts\], and TriviaQA \[joshi-etal-2017-triviaqa\].

Language understanding and reasoning benchmarks include BigBench Hard (BBH) \[bbh\], DROP \[drop\], HellaSwag \[hellaswag\], CLUEWSC \[clue\], and WinoGrande \[sakaguchi2019winogrande\].

Coding and mathematical benchmarks include BigCodeBench \[big\_code\_bench\], HumanEval \[codex\], GSM8K \[gsm8k\], MATH \[hendrycks2021measuring\], MGSM \[mgsm\], and CMath \[wei2023cmath\].

Long context benchmarks include LongBench-V2 \[bai2025longbench\].

Table 1: Comparison among DeepSeek-V3.2-Base, DeepSeek-V4-Flash-Base, and DeepSeek-V4-Pro-Base. All models are evaluated in our internal framework and share the same evaluation setting. Scores with a gap not exceeding 0.3 are considered to be at the same level. The highest score in each row is in bold font, and the second is underlined.

<table><tbody><tr><th></th><th rowspan="2">Benchmark (Metric)</th><th rowspan="2"># Shots</th><td>DeepSeek-V3.2</td><td>DeepSeek-V4-Flash</td><td>DeepSeek-V4-Pro</td></tr><tr><th></th><td>Base</td><td>Base</td><td>Base</td></tr><tr><th></th><th>Architecture</th><th>-</th><td>MoE</td><td>MoE</td><td>MoE</td></tr><tr><th></th><th># Activated Params</th><th>-</th><td>37B</td><td>13B</td><td>49B</td></tr><tr><th></th><th># Total Params</th><th>-</th><td>671B</td><td>284B</td><td>1.6T</td></tr><tr><th rowspan="12">World Knowl.</th><th>AGIEval (EM)</th><th>0-shot</th><td>80.1</td><td>82.6</td><td>83.1</td></tr><tr><th>MMLU (EM)</th><th>5-shot</th><td>87.8</td><td>88.7</td><td>90.1</td></tr><tr><th>MMLU-Redux (EM)</th><th>5-shot</th><td>87.5</td><td>89.4</td><td>90.8</td></tr><tr><th>MMLU-Pro (EM)</th><th>5-shot</th><td>65.5</td><td>68.3</td><td>73.5</td></tr><tr><th>MMMLU (EM)</th><th>5-shot</th><td>87.9</td><td>88.8</td><td>90.3</td></tr><tr><th>C-Eval (EM)</th><th>5-shot</th><td>90.4</td><td>92.1</td><td>93.1</td></tr><tr><th>CMMLU (EM)</th><th>5-shot</th><td>88.9</td><td>90.4</td><td>90.8</td></tr><tr><th>MultiLoKo (EM)</th><th>5-shot</th><td>38.7</td><td>42.2</td><td>51.1</td></tr><tr><th>Simple-QA verified (EM)</th><th>25-shot</th><td>28.3</td><td>30.1</td><td>55.2</td></tr><tr><th>SuperGPQA (EM)</th><th>5-shot</th><td>45.0</td><td>46.5</td><td>53.9</td></tr><tr><th>FACTS Parametric (EM)</th><th>25-shot</th><td>27.1</td><td>33.9</td><td>62.6</td></tr><tr><th>TriviaQA (EM)</th><th>5-shot</th><td>83.3</td><td>82.8</td><td>85.6</td></tr><tr><th rowspan="5">Lang. & Reas.</th><th>BBH (EM)</th><th>3-shot</th><td>87.6</td><td>86.9</td><td>87.5</td></tr><tr><th>DROP (F1)</th><th>1-shot</th><td>88.2</td><td>88.6</td><td>88.7</td></tr><tr><th>HellaSwag (EM)</th><th>0-shot</th><td>86.4</td><td>85.7</td><td>88.0</td></tr><tr><th>WinoGrande (EM)</th><th>0-shot</th><td>78.9</td><td>79.5</td><td>81.5</td></tr><tr><th>CLUEWSC (EM)</th><th>5-shot</th><td>83.5</td><td>82.2</td><td>85.2</td></tr><tr><th rowspan="6">Code & Math</th><th>BigCodeBench (Pass@1)</th><th>3-shot</th><td>63.9</td><td>56.8</td><td>59.2</td></tr><tr><th>HumanEval (Pass@1)</th><th>0-shot</th><td>62.8</td><td>69.5</td><td>76.8</td></tr><tr><th>GSM8K (EM)</th><th>8-shot</th><td>91.1</td><td>90.8</td><td>92.6</td></tr><tr><th>MATH (EM)</th><th>4-shot</th><td>60.5</td><td>57.4</td><td>64.5</td></tr><tr><th>MGSM (EM)</th><th>8-shot</th><td>81.3</td><td>85.7</td><td>84.4</td></tr><tr><th>CMath (EM)</th><th>3-shot</th><td>92.6</td><td>93.6</td><td>90.9</td></tr><tr><th>Long Context</th><th>LongBench-V2 (EM)</th><th>1-shot</th><td>40.2</td><td>44.7</td><td>51.5</td></tr></tbody></table>

#### 4.3.2 Evaluation Results

In Table 1, we provide a detailed comparison of the base models for DeepSeek-V3.2, DeepSeek-V4-Flash, and DeepSeek-V4-Pro, all evaluated under a unified internal framework with strictly consistent settings.

Comparing DeepSeek-V4-Flash-Base with DeepSeek-V3.2-Base reveals a compelling efficiency story. Despite utilizing a substantially smaller number of both activated and total parameters, DeepSeek-V4-Flash-Base outperforms DeepSeek-V3.2-Base across a wide array of benchmarks. This advantage is especially evident in world knowledge tasks and challenging long-context scenarios. These results underscore that architectural improvements, refined data quality, and training optimizations in DeepSeek-V4-Flash-Base yield superior performance even with a more compact parameter budget, effectively surpassing the larger DeepSeek-V3.2-Base on the majority of evaluations.

Furthermore, DeepSeek-V4-Pro-Base demonstrates a further, decisive leap in capability, establishing near-universal dominance over both DeepSeek-V3.2-Base and DeepSeek-V4-Flash-Base. With improvements across almost all categories, DeepSeek-V4-Pro-Base reaches new performance highs among DeepSeek base models on the most demanding benchmarks. On knowledge-intensive evaluations, it delivers dramatic gains, while also substantially advancing long-context understanding. On most reasoning and code benchmarks, DeepSeek-V4-Pro-Base also exceeds both previous models. This comprehensive uplift confirms DeepSeek-V4-Pro-Base as the strongest foundation model in the DeepSeek series, outperforming its predecessors across the spectrum of knowledge, reasoning, coding, and long-context capabilities.

## 5 Post-Training

### 5.1 Post-Training Pipeline

Following pre-training, we conducted a post-training phase to yield the final models of DeepSeek-V4 series. Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD; lu2025onpolicydistillation, minillm).

#### 5.1.1 Specialist Training

The development of domain specialists was conducted by adapting the DeepSeek-V3.2 training pipeline. Specifically, each model was sequentially optimized through an initial fine-tuning phase and subsequent Reinforcement Learning (RL) guided by domain-specific prompts and reward signals. For the RL stage, we implemented the Group Relative Policy Optimization (GRPO) algorithm, maintaining hyper-parameters closely aligned with our prior research \[dsr1, dsv32\].

##### Reasoning Efforts.

It is widely recognized that a model’s performance on reasoning tasks is fundamentally governed by the computational effort expended. Consequently, we trained distinct specialist models under divergent RL configurations to facilitate the development of models optimized for varying reasoning capacities. As detailed in Table 2, DeepSeek-V4-Pro and DeepSeek-V4-Flash both support three specific reasoning effort modes. For each mode, we apply distinct length penalties and context windows during RL training, which results in varying output token lengths for reasoning. To integrate these distinct reasoning modes, we utilize specialized response formats demarcated by the <think\> and </think\> tokens. Furthermore, for the "Think Max" mode, we prepend a specific instruction to the beginning of the system prompt to guide the model’s reasoning process, as shown in Table 3.

Table 2: Comparison of three reasoning modes

| Reasoning Mode | Characteristics | Typical Use Cases | Response Format |
| --- | --- | --- | --- |
| Non-think | Fast, intuitive responses based on habits or simple rules. | Routine daily tasks, emergency reactions, low-risk decisions. | </think\> summary |
| Think High | Conscious logical analysis, slower but more accurate. | Complex problem-solving, planning, medium-risk decisions. | <think\> thinking tokens </think\> summary |
| Think Max | Push reasoning to its fullest extent. Slow but powerful. | Exploring the boundary of model reasoning capability. | 1\. A special system prompt at the beginning.   2\. <think\> thinking tokens </think\> summary |

Table 3: Instruction injected into the system prompt for the "Think Max" mode.

<svg id="S5.T3.pic1" height="158.94" overflow="visible" version="1.1" viewBox="0 0 600 158.94" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,158.94) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#404040;" fill="#404040" fill-opacity="1.0"><path style="stroke:none" d="M 0 5.91 L 0 153.03 C 0 156.29 2.64 158.94 5.91 158.94 L 594.09 158.94 C 597.36 158.94 600 156.29 600 153.03 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" /></g><g style="--ltx-fill-color:#F2F2F2;" fill="#F2F2F2" fill-opacity="1.0"><path style="stroke:none" d="M 1.97 5.91 L 1.97 134.83 L 598.03 134.83 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 143.42)"><foreignobject style="--ltx-fg-color:#FFFFFF;--ltx-fo-width:40.23em;--ltx-fo-height:0.69em;--ltx-fo-depth:0.19em;" width="556.69" height="12.3" transform="matrix(1 0 0 -1 0 9.61)" overflow="visible" color="#FFFFFF"><span id="S5.T3.pic1.1.1.1.1.1" style="width:40.23em;"><span id="S5.T3.pic1.1.1.1.1.1.1">Injected Instruction</span> </span></foreignobject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 13.78)"><foreignobject style="--ltx-fg-color:#000000;--ltx-fo-width:40.23em;--ltx-fo-height:7.89em;--ltx-fo-depth:0em;" width="556.69" height="109.24" transform="matrix(1 0 0 -1 0 109.24)" overflow="visible" color="#000000"><span id="S5.T3.pic1.2.2.2.1.1" style="width:40.23em;"><span id="S5.T3.pic1.2.2.2.1.1.1">Reasoning Effort: Absolute maximum with no shortcuts permitted.</span> <span id="S5.T3.pic1.2.2.2.1.1.2">You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.</span> <span id="S5.T3.pic1.2.2.2.1.1.3">Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.</span></span></foreignobject></g></g></svg>

##### Generative Reward Model.

Typically, easy-to-verify tasks can be effectively optimized using simple rule-based verifiers or test cases. In contrast, hard-to-verify tasks traditionally rely on Reinforcement Learning from Human Feedback (RLHF), which necessitates extensive human annotation to train a scalar reward model. In the post-training phase of DeepSeek-V4 series, however, we dispense with these conventional scalar-based reward models. Instead, to address hard-to-verify tasks, we curate rubric-guided RL data and employ a Generative Reward Model (GRM) to evaluate policy trajectories. Crucially, we apply RL optimization directly to the GRM itself. In this paradigm, the actor network natively functions as the GRM, enabling the joint optimization of the model’s evaluative (judging) proficiency alongside its standard generative capabilities. By unifying these roles, the model’s internal reasoning capabilities are inherently fused into its evaluative process, resulting in highly robust scoring. Furthermore, this approach achieves superior performance with only a minimal set of diverse human annotations, as the model leverages its own logic to generalize across complex tasks.

Table 4: Tool-call schema for DeepSeek-V4 series.

<svg id="S5.T4.pic1" height="4942.22" overflow="visible" version="1.1" viewBox="0 0 600 4942.22" width="600"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,4942.22) matrix(1 0 0 -1 0 0)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-fill-color:#404040;" fill="#404040" fill-opacity="1.0"><path style="stroke:none" d="M 0 5.91 L 0 4936.31 C 0 4939.57 2.64 4942.22 5.91 4942.22 L 594.09 4942.22 C 597.36 4942.22 600 4939.57 600 4936.31 L 600 5.91 C 600 2.64 597.36 0 594.09 0 L 5.91 0 C 2.64 0 0 2.64 0 5.91 Z" /></g><g style="--ltx-fill-color:#F2F2F2;" fill="#F2F2F2" fill-opacity="1.0"><path style="stroke:none" d="M 1.97 5.91 L 1.97 4922.72 L 598.03 4922.72 L 598.03 5.91 C 598.03 3.73 596.27 1.97 594.09 1.97 L 5.91 1.97 C 3.73 1.97 1.97 3.73 1.97 5.91 Z" /></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 4928.63)"><foreignobject style="--ltx-fg-color:#FFFFFF;--ltx-fo-width:47.33em;--ltx-fo-height:0.65em;--ltx-fo-depth:0em;" width="556.69" height="7.69" transform="matrix(1 0 0 -1 0 7.69)" overflow="visible" color="#FFFFFF"><span id="S5.T4.pic1.1.1.1.1.1" style="width:47.33em;"><span id="S5.T4.pic1.1.1.1.1.1.1"><span id="S5.T4.pic1.1.1.1.1.1.1.1" style="font-size:80%;">Tool Call Schema</span></span> </span></foreignobject></g><g fill-opacity="1.0" transform="matrix(1.0 0.0 0.0 1.0 21.65 4903.78)"><foreignobject style="--ltx-fg-color:#000000;--ltx-fo-width:47.33em;--ltx-fo-height:0.61em;--ltx-fo-depth:415.76em;" width="556.69" height="4897.13" transform="matrix(1 0 0 -1 0 7.13)" overflow="visible" color="#000000"><span id="S5.T4.pic1.2.2.2.1.1" style="width:47.33em;"><span id="S5.T4.pic1.2.2.2.1.1.1"><a download="">⬇</a> <span id="lstnumberx1"><span id="lstnumberx1.2" style="font-size:80%;">##</span> <span id="lstnumberx1.4" style="font-size:80%;">Tools</span> </span><span id="lstnumberx3"><span id="lstnumberx3.2" style="font-size:80%;">You</span> <span id="lstnumberx3.4" style="font-size:80%;">have</span> <span id="lstnumberx3.6" style="font-size:80%;">access</span> <span id="lstnumberx3.8" style="font-size:80%;">to</span> <span id="lstnumberx3.10" style="font-size:80%;">a</span> <span id="lstnumberx3.12" style="font-size:80%;">set</span> <span id="lstnumberx3.14" style="font-size:80%;">of</span> <span id="lstnumberx3.16" style="font-size:80%;">tools</span> <span id="lstnumberx3.18" style="font-size:80%;">to</span> <span id="lstnumberx3.20" style="font-size:80%;">help</span> <span id="lstnumberx3.22" style="font-size:80%;">answer</span> <span id="lstnumberx3.24" style="font-size:80%;">the</span> <span id="lstnumberx3.26" style="font-size:80%;">user</span> <span id="lstnumberx3.27" style="font-size:80%;">’</span> <span id="lstnumberx3.28" style="font-size:80%;">s</span> <span id="lstnumberx3.30" style="font-size:80%;">question</span><span id="lstnumberx3.31" style="font-size:80%;">.</span><span id="lstnumberx3.33" style="font-size:80%;">You</span> <span id="lstnumberx3.35" style="font-size:80%;">can</span> <span id="lstnumberx3.37" style="font-size:80%;">invoke</span> <span id="lstnumberx3.39" style="font-size:80%;">tools</span> <span id="lstnumberx3.41" style="font-size:80%;">by</span> <span id="lstnumberx3.43" style="font-size:80%;">writing</span> <span id="lstnumberx3.45" style="font-size:80%;">a</span> <span id="lstnumberx3.47" style="font-size:80%;">"&lt;|</span> <span id="lstnumberx3.48" style="font-size:80%;">DSML</span> <span id="lstnumberx3.49" style="font-size:80%;">|</span> <span id="lstnumberx3.50" style="font-size:80%;">tool_calls</span> <span id="lstnumberx3.51" style="font-size:80%;">&gt;"</span> <span id="lstnumberx3.53" style="font-size:80%;">block</span> <span id="lstnumberx3.55" style="font-size:80%;">like</span> <span id="lstnumberx3.57" style="font-size:80%;">the</span> <span id="lstnumberx3.59" style="font-size:80%;">following</span><span id="lstnumberx3.60" style="font-size:80%;">:</span></span> <span id="lstnumberx5"><span id="lstnumberx5.2" style="font-size:80%;">&lt;|</span> <span id="lstnumberx5.3" style="font-size:80%;">DSML</span> <span id="lstnumberx5.4" style="font-size:80%;">|</span> <span id="lstnumberx5.5" style="font-size:80%;">tool_calls</span> <span id="lstnumberx5.6" style="font-size:80%;">&gt;</span> </span><span id="lstnumberx6"><span id="lstnumberx6.2" style="font-size:80%;">&lt;|</span> <span id="lstnumberx6.3" style="font-size:80%;">DSML</span> <span id="lstnumberx6.4" style="font-size:80%;">|</span> <span id="lstnumberx6.5" style="font-size:80%;">invoke</span> <span id="lstnumberx6.7" style="font-size:80%;">name</span> <span id="lstnumberx6.8" style="font-size:80%;">="</span> <span id="lstnumberx6.9" style="font-size:80%;">$TOOL_NAME</span> <span id="lstnumberx6.10" style="font-size:80%;">"&gt;</span> </span><span id="lstnumberx7"><span id="lstnumberx7.2" style="font-size:80%;">&lt;|</span> <span id="lstnumberx7.3" style="font-size:80%;">DSML</span> <span id="lstnumberx7.4" style="font-size:80%;">|</span> <span id="lstnumberx7.5" style="font-size:80%;">parameter</span> <span id="lstnumberx7.7" style="font-size:80%;">name</span> <span id="lstnumberx7.8" style="font-size:80%;">="</span> <span id="lstnumberx7.9" style="font-size:80%;">$PARAMETER_NAME</span> <span id="lstnumberx7.10" style="font-size:80%;">"</span> <span id="lstnumberx7.12" style="font-size:80%;">string</span> <span id="lstnumberx7.13" style="font-size:80%;">="</span> <span id="lstnumberx7.14" style="font-size:80%;">true</span> <span id="lstnumberx7.15" style="font-size:80%;">|</span> <span id="lstnumberx7.16" style="font-size:80%;">false</span> <span id="lstnumberx7.17" style="font-size:80%;">"&gt;</span> <span id="lstnumberx7.18" style="font-size:80%;">$PARAMETER_VALUE</span> <span id="lstnumberx7.19" style="font-size:80%;">&lt;/|</span> <span id="lstnumberx7.20" style="font-size:80%;">DSML</span> <span id="lstnumberx7.21" style="font-size:80%;">|</span> <span id="lstnumberx7.22" style="font-size:80%;">parameter</span> <span id="lstnumberx7.23" style="font-size:80%;">&gt;</span> </span><span id="lstnumberx8"><span id="lstnumberx8.2" style="font-size:80%;">...</span></span> <span id="lstnumberx9"><span id="lstnumberx9.2" style="font-size:80%;">&lt;/|</span> <span id="lstnumberx9.3" style="font-size:80%;">DSML</span> <span id="lstnumberx9.4" style="font-size:80%;">|</span> <span id="lstnumberx9.5" style="font-size:80%;">invoke</span> <span id="lstnumberx9.6" style="font-size:80%;">&gt;</span> </span><span id="lstnumberx10"><span id="lstnumberx10.2" style="font-size:80%;">&lt;|</span> <span id="lstnumberx10.3" style="font-size:80%;">DSML</span> <span id="lstnumberx10.4" style="font-size:80%;">|</span> <span id="lstnumberx10.5" style="font-size:80%;">invoke</span> <span id="lstnumberx10.7" style="font-size:80%;">name</span> <span id="lstnumberx10.8" style="font-size:80%;">="</span> <span id="lstnumberx10.9" style="font-size:80%;">$TOOL_NAME2</span> <span id="lstnumberx10.10" style="font-size:80%;">"&gt;</span> </span><span id="lstnumberx11"><span id="lstnumberx11.2" style="font-size:80%;">...</span></span> <span id="lstnumberx12"><span id="lstnumberx12.2" style="font-size:80%;">&lt;/|</span> <span id="lstnumberx12.3" style="font-size:80%;">DSML</span> <span id="lstnumberx12.4" style="font-size:80%;">|</span> <span id="lstnumberx12.5" style="font-size:80%;">invoke</span> <span id="lstnumberx12.6" style="font-size:80%;">&gt;</span> </span><span id="lstnumberx13"><span id="lstnumberx13.2" style="font-size:80%;">&lt;/|</span> <span id="lstnumberx13.3" style="font-size:80%;">DSML</span> <span id="lstnumberx13.4" style="font-size:80%;">|</span> <span id="lstnumberx13.5" style="font-size:80%;">tool_calls</span> <span id="lstnumberx13.6" style="font-size:80%;">&gt;</span> </span><span id="lstnumberx15"><span id="lstnumberx15.2" style="font-size:80%;">String</span> <span id="lstnumberx15.4" style="font-size:80%;">parameters</span> <span id="lstnumberx15.6" style="font-size:80%;">should</span> <span id="lstnumberx15.8" style="font-size:80%;">be</span> <span id="lstnumberx15.10" style="font-size:80%;">specified</span> <span id="lstnumberx15.12" style="font-size:80%;">as</span> <span id="lstnumberx15.14" style="font-size:80%;">is</span> <span id="lstnumberx15.16" style="font-size:80%;">and</span> <span id="lstnumberx15.18" style="font-size:80%;">set</span> <span id="lstnumberx15.20" style="font-size:80%;">‘</span> <span id="lstnumberx15.21" style="font-size:80%;">string</span> <span id="lstnumberx15.22" style="font-size:80%;">="</span> <span id="lstnumberx15.23" style="font-size:80%;">true</span> <span id="lstnumberx15.24" style="font-size:80%;">"‘.</span><span id="lstnumberx15.26" style="font-size:80%;">For</span> <span id="lstnumberx15.28" style="font-size:80%;">all</span> <span id="lstnumberx15.30" style="font-size:80%;">other</span> <span id="lstnumberx15.32" style="font-size:80%;">types</span> <span id="lstnumberx15.34" style="font-size:80%;">(</span><span id="lstnumberx15.35" style="font-size:80%;">numbers</span><span id="lstnumberx15.36" style="font-size:80%;">,</span><span id="lstnumberx15.38" style="font-size:80%;">booleans</span><span id="lstnumberx15.39" style="font-size:80%;">,</span><span id="lstnumberx15.41" style="font-size:80%;">arrays</span><span id="lstnumberx15.42" style="font-size:80%;">,</span><span id="lstnumberx15.44" style="font-size:80%;">objects</span><span id="lstnumberx15.45" style="font-size:80%;">),</span><span id="lstnumberx15.47" style="font-size:80%;">pass</span> <span id="lstnumberx15.49" style="font-size:80%;">the</span> <span id="lstnumberx15.51" style="font-size:80%;">value</span> <span id="lstnumberx15.53" style="font-size:80%;">in</span> <span id="lstnumberx15.55" style="font-size:80%;">JSON</span> <span id="lstnumberx15.57" style="font-size:80%;">format</span> <span id="lstnumberx15.59" style="font-size:80%;">and</span> <span id="lstnumberx15.61" style="font-size:80%;">set</span> <span id="lstnumberx15.63" style="font-size:80%;">‘</span> <span id="lstnumberx15.64" style="font-size:80%;">string</span> <span id="lstnumberx15.65" style="font-size:80%;">="</span> <span id="lstnumberx15.66" style="font-size:80%;">false</span> <span id="lstnumberx15.67" style="font-size:80%;">"‘.</span></span> <span id="lstnumberx17"><span id="lstnumberx17.2" style="font-size:80%;">If</span> <span id="lstnumberx17.4" style="font-size:80%;">thinking_mode</span> <span id="lstnumberx17.6" style="font-size:80%;">is</span> <span id="lstnumberx17.8" style="font-size:80%;">enabled</span> <span id="lstnumberx17.10" style="font-size:80%;">(</span><span id="lstnumberx17.11" style="font-size:80%;">triggered</span> <span id="lstnumberx17.13" style="font-size:80%;">by</span> <span id="lstnumberx17.15" style="font-size:80%;">&lt;</span> <span id="lstnumberx17.16" style="font-size:80%;">think</span> <span id="lstnumberx17.17" style="font-size:80%;">&gt;),</span><span id="lstnumberx17.19" style="font-size:80%;">you</span> <span id="lstnumberx17.21" style="font-size:80%;">MUST</span> <span id="lstnumberx17.23" style="font-size:80%;">output</span> <span id="lstnumberx17.25" style="font-size:80%;">your</span> <span id="lstnumberx17.27" style="font-size:80%;">complete</span> <span id="lstnumberx17.29" style="font-size:80%;">reasoning</span> <span id="lstnumberx17.31" style="font-size:80%;">inside</span> <span id="lstnumberx17.33" style="font-size:80%;">&lt;</span> <span id="lstnumberx17.34" style="font-size:80%;">think</span> <span id="lstnumberx17.35" style="font-size:80%;">&gt;...&lt;/</span> <span id="lstnumberx17.36" style="font-size:80%;">think</span> <span id="lstnumberx17.37" style="font-size:80%;">&gt;</span> <span id="lstnumberx17.39" style="font-size:80%;">BEFORE</span> <span id="lstnumberx17.41" style="font-size:80%;">any</span> <span id="lstnumberx17.43" style="font-size:80%;">tool</span> <span id="lstnumberx17.45" style="font-size:80%;">calls</span> <span id="lstnumberx17.47" style="font-size:80%;">or</span> <span id="lstnumberx17.49" style="font-size:80%;">final</span> <span id="lstnumberx17.51" style="font-size:80%;">response</span><span id="lstnumberx17.52" style="font-size:80%;">.</span></span> <span id="lstnumberx19"><span id="lstnumberx19.2" style="font-size:80%;">Otherwise</span><span id="lstnumberx19.3" style="font-size:80%;">,</span><span id="lstnumberx19.5" style="font-size:80%;">output</span> <span id="lstnumberx19.7" style="font-size:80%;">directly</span> <span id="lstnumberx19.9" style="font-size:80%;">after</span> <span id="lstnumberx19.11" style="font-size:80%;">&lt;/</span> <span id="lstnumberx19.12" style="font-size:80%;">think</span> <span id="lstnumberx19.13" style="font-size:80%;">&gt;</span> <span id="lstnumberx19.15" style="font-size:80%;">with</span> <span id="lstnumberx19.17" style="font-size:80%;">tool</span> <span id="lstnumberx19.19" style="font-size:80%;">calls</span> <span id="lstnumberx19.21" style="font-size:80%;">or</span> <span id="lstnumberx19.23" style="font-size:80%;">final</span> <span id="lstnumberx19.25" style="font-size:80%;">response</span><span id="lstnumberx19.26" style="font-size:80%;">.</span></span> <span id="lstnumberx21"><span id="lstnumberx21.2" style="font-size:80%;">###</span> <span id="lstnumberx21.4" style="font-size:80%;">Available</span> <span id="lstnumberx21.6" style="font-size:80%;">Tool</span> <span id="lstnumberx21.8" style="font-size:80%;">Schemas</span> </span><span id="lstnumberx23"><span id="lstnumberx23.2" style="font-size:80%;">{</span> <span id="lstnumberx23.3" style="font-size:80%;">Tool</span> <span id="lstnumberx23.5" style="font-size:80%;">Definition</span><span id="lstnumberx23.6" style="font-size:80%;">...}</span></span></span></span></foreignobject></g></g></svg>

##### Tool-Call Schema and Special Token.

Consistent with our previous version, we utilize a dedicated <think\></think\> tag to delineate the reasoning path. In DeepSeek-V4 series, we introduce a new tool-call schema that employs a special "|DSML|" token and utilizes an XML-based format for tool invocations, as demonstrated in Table 4. Our experiments demonstrate that the XML format effectively mitigates escaping failures and reduces tool-call errors, providing a more robust interface for model-tool interactions.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x7.png)

Figure 7: Thinking management of DeepSeek-V4 series.

##### Interleaved Thinking.

DeepSeek-V3.2 introduced a context management strategy that retains reasoning traces across tool-result rounds but discards them upon the arrival of new user messages. While effective, this still caused unnecessary token waste in complex agentic workflows — each new user turn would flush all accumulated reasoning content, forcing the model to reconstruct its problem-solving state from scratch. Leveraging the expanded 1M-token context window of DeepSeek-V4 series, we further refine this mechanism to maximize the effectiveness of interleaved thinking in agentic environments:

- Tool-Calling Scenarios. As illustrated in Figure 7(a), all reasoning content is fully preserved throughout the entire conversation. Unlike DeepSeek-V3.2, which discarded thinking traces upon each new user turn, DeepSeek-V4 series retain the complete reasoning history across all rounds, including across user message boundaries. This allows the model to maintain a coherent, cumulative chain of thought over long-horizon agent tasks.
- General Conversational Scenarios. As illustrated in Figure 7(b), the original strategy is preserved: reasoning content from previous turns is discarded when a new user message arrives, keeping the context concise for settings where persistent reasoning traces provide limited benefit.

As with DeepSeek-V3.2, agent frameworks that simulate tool interactions via user messages (e.g., Terminus) may not trigger the tool-calling context path and thus may not benefit from enhanced reasoning persistence. We continue to recommend non-think models for such architectures.

Table 5: Quick Instruction special tokens for auxiliary tasks.

| Special Token | Description | Format |
| --- | --- | --- |
| <\|action\|\> | Determines whether the user prompt requires a web search or can be answered directly. | ...<\|User\|\>{prompt}<\|Assistant\|\><think\><\|action\|\> |
| <\|title\|\> | Generates a concise conversation title after the first assistant response. | ...<\|Assistant\|\>{response}<\|end\_of\_sentence\|\><\|title\|\> |
| <\|query\|\> | Generates search queries for the user prompt. | ...<\|User\|\>{prompt}<\|query\|\> |
| <\|authority\|\> | Classifies the user prompt’s demand for source authoritativeness. | ...<\|User\|\>{prompt}<\|authority\|\> |
| <\|domain\|\> | Identifies the domain of the user prompt. | ...<\|User\|\>{prompt}<\|domain\|\> |
| <\|extracted\_url\|\> <\|read\_url\|\> | Determines whether each URL in the user prompt should be fetched and read. | ...<\|User\|\>{prompt}<\|extracted\_url\|\>{url}<\|read\_url\|\> |

##### Quick Instruction.

In chatbot scenarios, a number of auxiliary tasks (e.g., determining whether to trigger a web search, intent recognition, etc.) must be executed before generating the response. Conventionally, these tasks are handled by a separate small model, requiring redundant prefilling since it cannot reuse the existing KV cache. To overcome this limitation, we introduce Quick Instruction. We append a set of dedicated special tokens directly to the input sequence, where each token corresponds to a specific auxiliary task. By directly reusing the already-computed KV cache, this mechanism completely avoids redundant prefilling and allows certain tasks, such as generating search queries and determining authority and domain, to be executed in parallel. Consequently, this approach significantly reduces the user-perceived time-to-first-token (TTFT) and eliminates the engineering overhead of maintaining and iterating an extra small model. The supported Quick Instruction tokens are summarized in Table 5.

#### 5.1.2 On-Policy Distillation

After training multiple domain-specific experts via specialized fine-tuning and reinforcement learning, we employ multi-teacher On-Policy Distillation (OPD; lu2025onpolicydistillation, minillm) as the primary technique for merging expert capabilities into the final model. OPD has emerged as an effective post-training paradigm for efficiently transferring the knowledge and capabilities of domain experts to a single, unified model. This is achieved by having the student learn from the output distributions of teacher models on its own generated trajectories. Formally, given a set of $N$ expert models $\{\pi_{E_{1}},\pi_{E_{2}},\dots,\pi_{E_{N}}\}$, the OPD objective function is defined as:

$$
\mathcal{L}_{\text{OPD}}(\theta)=\sum_{i=1}^{N}w_{i}\cdot\text{D}_{\text{KL}}\left(\pi_{\theta}\parallel\pi_{E_{i}}\right).
$$

In this formulation, $w_{i}$ represents the assigned weight for each expert, typically determined by the relative importance of the expert. Computing the reverse KL loss $\text{D}_{\text{KL}}\left(\pi_{\theta}\parallel\pi_{E_{i}}\right)$ requires sampling training trajectories from the student $\pi_{\theta}$ to maintain on-policy learning. The underlying logic ensures that the unified policy $\pi_{\theta}$ selectively learns from the specialized expert relevant to the current task context (e.g., aligning with the mathematics expert for math reasoning tasks and the coding expert for programming tasks). Through this mechanism, the knowledge from physically distinct expert weights is consolidated into a unified parameter space via logits-level alignment, practically circumventing the performance degradation often encountered in traditional weight-merging or mixed RL techniques. In this stage, more than ten teacher models covering various domains are employed to distill a single student model.

In handling the above OPD objective, prior works usually simplify the full-vocabulary KL loss into a token-level KL estimate at each token position, and reuse RL framework by replacing $\texttt{sg}\big[\log\frac{\pi_{E_{i}}(y_{t}|x,y_{<t})}{\pi_{\theta}(y_{t}|x,y_{<t})}\big]$ (sg represents the stop gradient operation) as the per-token advantage estimate in the policy loss calculation. Although this approach is resource-efficient, it leads to high variance in gradient estimation and often causes training instability. Therefore, we adopt full-vocabulary logit distillation in our OPD. Preserving the complete logit distribution in calculating reverse KL loss yields more stable gradient estimates and ensures faithful distillation of the teachers’ knowledge. In the following subsection, we describe the engineering efforts that make full-vocabulary OPD feasible at scale.

### 5.2 Post-Training Infrastructures

Our post-training infrastructure is built upon the scalable framework developed for DeepSeek-V3.2. Specifically, we integrate the same distributed training stack described in Section 3.4 and the rollout engine introduced earlier for efficient auto-regressive sampling. Building on this foundation, we introduce the following principal enhancements in the present work. These designs enable efficient execution of ultra-long-context RL and OPD merging tasks involving over ten distinct teacher models, thereby substantially accelerating the iteration cycle for model releases.

#### 5.2.1 FP4 Quantization-Aware Training

To achieve inference acceleration and reducing memory traffic at deployment, we introduce Quantization-Aware Training (QAT) \[QAT\] during the post-training stage, enabling the model, including those of teacher and reference models, to adapt to the precision degradation introduced by quantization. We apply FP4 (MXFP4) quantization \[OCP\_MXFormat\] to two components: (1) MoE expert weights, which are a major source of GPU memory occupancy \[gpt\_oss\], and (2) the Query-Key (QK) path in the indexer of CSA, where QK activations are cached, loaded, and multiplied entirely in FP4, accelerating attention score computation in long-context scenarios. In addition, we further quantize the index scores $I_{:,:}$ from FP32 to BF16 during this QAT process. This optimization achieves a 2 $\times$ speedup for the top-k selector, while preserving a 99.7% recall rate of KV entries.

For MoE expert weights, following the common practice of QAT, the FP32 master weights maintained by the optimizer are first quantized to FP4, then dequantized back to FP8 for computation. Notably, our FP4-to-FP8 dequantization is lossless. This is because FP8 (E4M3) has 2 additional exponent bits compared with FP4 (E2M1), offering a larger dynamic range. Consequently, as long as the ratio between the maximum and minimum scale factors of the FP4 sub-blocks ($1\times 32$ tiles) within each FP8 quantization block ($128\times 128$ tiles) does not exceed a certain threshold, the fine-grained scale information can be fully absorbed by the extended dynamic range of FP8. We empirically verify that current weights satisfy this condition. This allows the entire QAT pipeline to fully reuse the existing FP8 training framework without any modification. In the backward pass, gradients are computed with respect to the same FP8 weights in the forward pass and directly propagated back to the FP32 master weights, equivalent to applying the Straight-Through Estimator (STE) through the quantization operation. This also avoids the need to re-quantize transposed weights.

During the inference and rollout phases of RL training, which do not involve backward passes, we directly use native FP4 quantized weights instead of simulated quantization. This ensures that model behavior during sampling is fully consistent with online deployment, while also reducing kernel memory loading for actual speedup and significantly lowering memory consumption. We process the QK path in the indexer of CSA similarly.

#### 5.2.2 Efficient Teacher Scheduling for Full-Vocabulary OPD

Our framework supports full-vocabulary On-Policy Distillation (OPD) with an effectively unbounded number of teachers, each potentially comprising trillions of parameters. To enable this, all teacher weights are offloaded to a centralized distributed storage and are loaded on demand during the teacher forward pass with ZeRO-like parameter sharding to alleviate both I/O and DRAM pressure. Furthermore, naively materializing logits for a vocabulary size $|V|>100\text{k}$ across all teachers is prohibitive, even when spooled to disk. We address this by caching only the last-layer teacher hidden states in a centralized buffer during the forward pass. At training time, these cached states are retrieved and passed through the corresponding prediction head module to reconstruct the full logits on the fly. This design incurs negligible recomputation overhead while completely circumventing the memory burden associated with explicit logits materialization. To mitigate the GPU memory footprint of the teacher prediction head, we order training samples by teacher index during data dispatching. This arrangement ensures that each distinct teacher head is loaded only once per mini-batch and that at most one teacher head resides in device memory at any given time. All parameters and hidden state loading/offloading operations proceed asynchronously in the background, without blocking computation on the critical path. Finally, the exact KL divergences between teacher and student logits are computed using a specialized TileLang kernel, which accelerates the computation and curtails dynamic memory allocation.

#### 5.2.3 Preemptible and Fault-Tolerant Rollout Service

To maximize GPU resource utilization while enabling rapid hardware provisioning for high-priority tasks, our GPU cluster employs a cluster-wide preemptive task scheduler, where any running task may be preempted at any time. Also, hardware failures are prevalent in large-scale GPU clusters. To this end, we implement a preemptible and fault-tolerant LLM generation service for RL/OPD rollout.

Specifically, we implement a token-granular Write-Ahead Log (WAL) for each generation request. Whenever a new token is generated for a request, we immediately append it to that request’s WAL. During preemption, we pause the inference engine and save the KV cache of unfinished requests. Upon resumption, we use the persisted WALs and saved KV cache to continue decoding. Even when a fatal hardware error occurs, we can re-run the prefill phase using the persisted tokens in WAL to reconstruct the KV cache.

Importantly, it is mathematically incorrect to regenerate unfinished requests from scratch, as this introduces length bias. Because shorter responses are more likely to survive interruption, regenerating from scratch makes the model more prone to producing shorter sequences whenever an interruption occurs. If the inference stack is batch-invariant and deterministic, this correctness issue could also be addressed by regenerating with a consistent seed for the pseudorandom number generator used in the sampler. However, this approach still incurs the extra cost of re-running the decoding phase, making it far less efficient than our token-granular WAL method.

#### 5.2.4 Scaling RL Framework for Million-Token Context

We introduce targeted optimizations for efficient RL and OPD on million-token sequences. During the rollout phase, we adopt a preemptible and fault-tolerant rollout service, detailed in Section 5.2.3. For the inference and training phase, we decompose the rollout data format into lightweight metadata and heavy per-token fields. During data dispatching, the metadata for the entire rollout data can be loaded to perform global shuffling and packing layout computation. Heavy per-token fields are loaded via a shared-memory data loader to eliminate intra-node data redundancy and are released immediately upon consumption at the mini-batch granularity, substantially reducing both CPU and GPU memory pressure. The number of on-device mini-batches is dynamically determined based on workload, allowing an efficient trade-off between computational throughput and I/O overlap.

#### 5.2.5 Sandbox Infrastructure for Agentic AI

To meet the diverse execution demands of agentic AI during post-training and evaluation, we build a production-grade sandbox platform, DeepSeek Elastic Compute (DSec). DSec comprises three Rust components — the API gateway (Apiserver), per-host agent (Edge), and the cluster monitor (Watcher) — that are interconnected by a custom RPC protocol and scale horizontally atop the 3FS distributed filesystem \[ds3fs\]. In production, a single DSec cluster manages hundreds of thousands of concurrent sandbox instances.

The design of DSec is motivated by four observations: (1) agentic workloads are highly heterogeneous, spanning lightweight function calls to full software-engineering pipelines with diverse OS and security requirements; (2) environment images are numerous and large, yet must load quickly and support iterative customization; (3) high-density deployment demands efficient CPU and memory utilization; (4) sandbox lifecycles must coordinate with GPU training schedules, including preemption and checkpoint-based resumption. Based on these observations, we elaborate on the four core designs of DSec individually in the following.

##### Four Execution Substrates Behind One Unified Interface.

DSec exposes a single Python SDK (libdsec) that abstracts four execution substrates. Function Call dispatches stateless invocations to a pre-warmed container pool, eliminating cold-start overhead. Container is fully Docker-compatible and leverages EROFS \[erofs\] on-demand loading for efficient image assembly. microVM, built on Firecracker \[firecracker\], adds VM-level isolation for security-sensitive, high-density deployments. fullVM, built on QEMU \[qemu\], supports arbitrary guest operating systems. All four share a common API surface — command execution, file transfer, and TTY access — and switching between them requires only a parameter change.

##### Fast Image Loading via Layered Storage.

DSec reconciles fast startup with a large and growing corpus of environment images through layered, on-demand loading. For containers, base images and filesystem commits are stored as 3FS-backed readonly EROFS layers mounted directly into overlay lowerdirs. We keep file metadata readily available on the local disk at mount time; meanwhile, data blocks are fetched from 3FS upon request. For microVMs, DSec uses the overlaybd \[overlaybd\] disk format: the read-only base layer resides on 3FS for cross-instance sharing, while writes go to a local copy-on-write layer. Such snapshots are chainable, facilitating efficient versioning and millisecond-scale resumption.

##### Density Optimizations Under Massive Concurrency.

To accommodate hundreds of thousands of sandboxes per cluster, DSec tackles two resource bottlenecks. First, it mitigates duplicate page-cache footprints in virtualized environments and applies memory reclamation to enable safe overcommitment. Second, it alleviates spinlock contention in the container runtime and therefore, reduces per-sandbox CPU overhead, significantly increasing per-host packing density.

##### Trajectory Logging and Preemption-Safe Resumption.

DSec maintains a globally ordered trajectory log for each sandbox, persistently recording every command invocation and its results. The trajectory serves three purposes: (1) client fast-forwarding — when a training task is preempted, sandbox resources are retained nonetheless; upon resumption, DSec replays cached results for previously completed commands, accelerating task recovery whilst also preventing errors from re-execution of non-idempotent operations; (2) fine-grained provenance — the origin and corresponding outcomes of each state change are traceable; (3) deterministic replay — any historical session can be faithfully reproduced from its trajectory.

### 5.3 Standard Benchmark Evaluation

#### 5.3.1 Evaluation Setup

##### Knowledge and Reasoning.

Knowledge and reasoning datasets include MMLU-Pro \[mmlu\_pro\], GPQA \[gpqa\], Human Last Exam \[hle\], Simple-QA Verified \[haas2025simpleqa\], Chinese-SimpleQA \[csimpleqa\], LiveCodeBench-v6 \[jain2024livecodebench\], CodeForces (Internal Benchmark), HMMT 2026 Feb, Apex \[balunovic2025matharena\], Apex Shortlist \[balunovic2025matharena\], IMOAnswerBench \[luong-etal-2025-towards\], and PutnamBench \[tsoukalas2024putnambenchevaluatingneuraltheoremprovers\].

For code, we evaluate DeepSeek-V4 series on LiveCodeBench-v6 and an internal Codeforces benchmark. For Codeforces, we collect 14 Codeforces Division 1 contests comprising 114 problems (May 2025 - November 2025). The Elo rating is computed as follows. For each contest, we generate 32 candidate solutions per problem. For each problem independently, we sample 10 of these solutions without replacement and arrange them in a random order to form the submission sequence. Each submission is judged against a test suite constructed by domain experts. The score for a solved problem follows the penalty scheme of OpenAI (2025): the model receives the median score of human participants who solved the same problem with the same number of prior failed attempts. This yields a total contest score for each sampled submission sequence, which is then converted into a contest rank and subsequently into an estimated rating via the standard Codeforces rating system. The contest-level expected rating is defined as the expectation of this estimated rating over all possible random selections and orderings of the 10 submissions per problem. The model’s overall rating is the average of these contest-level expected ratings across all 14 contests.

For reasoning and knowledge tasks, we set the temperature to 1.0 and the context window to 8K, 128K, and 384K tokens for the Non-think, High, and Max modes, respectively. For math tasks (e.g., HMMT, IMOAnswerBench, Apex, and HLE), we evaluate using the following template: "{question}\\nPlease reason step by step, and put your final answer within \\boxed{}." For DeepSeek-V4-Pro-Max on math tasks, we use the following template to elicit deeper reasoning: "Solve the following problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.\\n\\n{question}".

For formal math tasks, we evaluate in an agentic setting on Lean v4.28.0-rc1 \[moura2021lean\], with access to the Lean compiler and a semantic tactic search engine, running up to 500 tool calls with max reasoning effort. In addition, we evaluate a more compute-intensive pipeline in which candidate natural-language solutions are first generated and filtered by self-verification \[shao2025deepseekmathv2selfverifiablemathematicalreasoning\], and the retained solutions are then provided as guidance to a formal agent for proving the corresponding Lean statement. This design uses informal reasoning to improve exploration while preserving strict correctness through formal verification. A submission is counted as correct only if the strict verifier Comparator accepts it for both settings.

We have left some entries blank for K2.6 and GLM-5.1, as their APIs were too busy to return responses to our queries.

##### 1M-Token Context.

Since DeepSeek-V4 series supports 1M-token contexts, we evaluate model performance in a long context scenario by selecting OpenAI MRCR \[mrcr\] and CorpusQA \[lu2026corpusqa\] as the benchmarks. We re-evaluate Claude Opus 4.6 and Gemini 3.1 Pro on these tasks with the goal of standardizing the configuration across all models. We did not evaluate GPT-5.4 because its API failed to respond to a large portion of our queries.

##### Agent.

Agent datasets include Terminal Bench 2.0 \[merrill2026terminal\], SWE-Verified \[swe\_verified\], SWE Multilingual \[yang2025swesmith\], SWE-Pro \[deng2025swebenchproaiagents\], BrowseComp \[wei2025browsecomp\], the public evaluation set of MCPAtlas \[bandi2026mcp\], GDPval-AA \[patwardhan2025gdpval, gdpaa\], and Tool-Decathlon \[li2025tool\].

For code agent tasks (SWE-Verified, Terminal-Bench, SWE-Pro, SWE Multilingual), we evaluate DeepSeek-V4 series using an internally developed evaluation framework. This framework provides a minimal set of tools — a bash tool and a file-edit tool. The maximum number of interaction steps is set to 500, and the maximum context length is set to 512K tokens. Regarding Terminal-Bench 2.0, we acknowledge the environment-related issues noted by GLM-5.1. Nevertheless, we report our performance on the original Terminal-Bench 2.0 dataset for consistency. On the Terminal-Bench 2.0 Verified subset, DeepSeek-V4-Pro achieves a score of approximately 72.0.

For search agent tasks (BrowseComp, HLE w/ tool), we also use an in-house harness with websearch and Python tool, and set maximum interaction steps to 500 and the maximum context length to 512K tokens. For BrowseComp, we use the same discard-all context management strategy as DeepSeek-V3.2 \[dsv32\].

#### 5.3.2 Evaluation Results

Table 6: Comparison between DeepSeek-V4-Pro-Max and closed/open source models. "Max", "xHigh", and "High" denote reasoning effort. The best results are highlighted in bold; the second-best results are underlined.

<table><tbody><tr><td></td><td rowspan="2">Benchmark (Metric)</td><td>Opus-4.6</td><td>GPT-5.4</td><td>Gemini-3.1-Pro</td><td>K2.6</td><td>GLM-5.1</td><td>DS-V4-Pro</td><td></td></tr><tr><td></td><td>Max</td><td>xHigh</td><td>High</td><td>Thinking</td><td>Thinking</td><td>Max</td><td></td></tr><tr><td rowspan="11">Knowledge & Reasoning</td><td>MMLU-Pro (EM)</td><td>89.1</td><td>87.5</td><td>91.0</td><td>87.1</td><td>86.0</td><td>87.5</td><td></td></tr><tr><td>SimpleQA-Verified (Pass@1)</td><td>46.2</td><td>45.3</td><td>75.6</td><td>36.9</td><td>38.1</td><td>57.9</td><td></td></tr><tr><td>Chinese-SimpleQA (Pass@1)</td><td>76.4</td><td>76.8</td><td>85.9</td><td>75.9</td><td>75.0</td><td>84.4</td><td></td></tr><tr><td>GPQA Diamond (Pass@1)</td><td>91.3</td><td>93.0</td><td>94.3</td><td>90.5</td><td>86.2</td><td>90.1</td><td></td></tr><tr><td>HLE (Pass@1)</td><td>40.0</td><td>39.8</td><td>44.4</td><td>36.4</td><td>34.7</td><td>37.7</td><td></td></tr><tr><td>LiveCodeBench (Pass@1)</td><td>88.8</td><td>-</td><td>91.7</td><td>89.6</td><td>-</td><td>93.5</td><td></td></tr><tr><td>Codeforces (Rating)</td><td>-</td><td>3168</td><td>3052</td><td>-</td><td>-</td><td>3206</td><td></td></tr><tr><td>HMMT 2026 Feb (Pass@1)</td><td>96.2</td><td>97.7</td><td>94.7</td><td>92.7</td><td>89.4</td><td>95.2</td><td></td></tr><tr><td>IMOAnswerBench (Pass@1)</td><td>75.3</td><td>91.4</td><td>81.0</td><td>86.0</td><td>83.8</td><td>89.8</td><td></td></tr><tr><td>Apex (Pass@1)</td><td>34.5</td><td>54.1</td><td>60.9</td><td>24.0</td><td>11.5</td><td>38.3</td><td></td></tr><tr><td>Apex Shortlist (Pass@1)</td><td>85.9</td><td>78.1</td><td>89.1</td><td>75.5</td><td>72.4</td><td>90.2</td><td></td></tr><tr><td rowspan="2">Long</td><td>MRCR 1M (MMR)</td><td>92.9</td><td>-</td><td>76.3</td><td>-</td><td>-</td><td>83.5</td><td></td></tr><tr><td>CorpusQA 1M (ACC)</td><td>71.7</td><td>-</td><td>53.8</td><td>-</td><td>-</td><td>62.0</td><td></td></tr><tr><td rowspan="10">Agentic</td><td>Terminal Bench 2.0 (Acc)</td><td>65.4</td><td>75.1</td><td>68.5</td><td>66.7</td><td>63.5</td><td>67.9</td><td></td></tr><tr><td>SWE Verified (Resolved)</td><td>80.8</td><td>-</td><td>80.6</td><td>80.2</td><td>-</td><td>80.6</td><td></td></tr><tr><td>SWE Pro (Resolved)</td><td>57.3</td><td>57.7</td><td>54.2</td><td>58.6</td><td>58.4</td><td>55.4</td><td></td></tr><tr><td>SWE Multilingual (Resolved)</td><td>77.5</td><td>-</td><td>-</td><td>76.7</td><td>73.3</td><td>76.2</td><td></td></tr><tr><td>BrowseComp (Pass@1)</td><td>83.7</td><td>82.7</td><td>85.9</td><td>83.2</td><td>79.3</td><td>83.4</td><td></td></tr><tr><td>HLE w/ tools (Pass@1)</td><td>53.1</td><td>52.0</td><td>51.6</td><td>54.0</td><td>50.4</td><td>48.2</td><td></td></tr><tr><td>GDPval-AA (Elo)</td><td>1619</td><td>1674</td><td>1314</td><td>1482</td><td>1535</td><td>1554</td><td></td></tr><tr><td>MCPAtlas Public(Pass@1)</td><td>73.8</td><td>67.2</td><td>69.2</td><td>66.6</td><td>71.8</td><td>73.6</td><td></td></tr><tr><td>Toolathlon (Pass@1)</td><td>47.2</td><td>54.6</td><td>48.8</td><td>50.0</td><td>40.7</td><td>51.8</td><td></td></tr></tbody></table>

Table 7: Comparison among different sizes and modes of DeepSeek-V4 series. "Non-Think", "High", and "Max" denote reasoning effort.

<table><tbody><tr><td></td><td rowspan="2">Benchmark (Metric)</td><td colspan="3">DeepSeek-V4-Flash</td><td colspan="3">DeepSeek-V4-Pro</td></tr><tr><td></td><td>Non-Think</td><td>High</td><td>Max</td><td>Non-Think</td><td>High</td><td>Max</td></tr><tr><td rowspan="11">Knowledge & Reasoning</td><td>MMLU-Pro (EM)</td><td>83.0</td><td>86.4</td><td>86.2</td><td>82.9</td><td>87.1</td><td>87.5</td></tr><tr><td>SimpleQA-Verified (Pass@1)</td><td>23.1</td><td>28.9</td><td>34.1</td><td>45.0</td><td>46.2</td><td>57.9</td></tr><tr><td>Chinese-SimpleQA (Pass@1)</td><td>71.5</td><td>73.2</td><td>78.9</td><td>75.8</td><td>77.7</td><td>84.4</td></tr><tr><td>GPQA Diamond (Pass@1)</td><td>71.2</td><td>87.4</td><td>88.1</td><td>72.9</td><td>89.1</td><td>90.1</td></tr><tr><td>HLE (Pass@1)</td><td>8.1</td><td>29.4</td><td>34.8</td><td>7.7</td><td>34.5</td><td>37.7</td></tr><tr><td>LiveCodeBench (Pass@1-COT)</td><td>55.2</td><td>88.4</td><td>91.6</td><td>56.8</td><td>89.8</td><td>93.5</td></tr><tr><td>Codeforces (Rating)</td><td>-</td><td>2816</td><td>3052</td><td>-</td><td>2919</td><td>3206</td></tr><tr><td>HMMT 2026 Feb (Pass@1)</td><td>40.8</td><td>91.9</td><td>94.8</td><td>31.7</td><td>94.0</td><td>95.2</td></tr><tr><td>IMOAnswerBench (Pass@1)</td><td>41.9</td><td>85.1</td><td>88.4</td><td>35.3</td><td>88.0</td><td>89.8</td></tr><tr><td>Apex (Pass@1)</td><td>1.0</td><td>19.1</td><td>33.0</td><td>0.4</td><td>27.4</td><td>38.3</td></tr><tr><td>Apex Shortlist (Pass@1)</td><td>9.3</td><td>72.1</td><td>85.7</td><td>9.2</td><td>85.5</td><td>90.2</td></tr><tr><td rowspan="2">Long</td><td>MRCR 1M(MMR)</td><td>37.5</td><td>76.9</td><td>78.7</td><td>44.7</td><td>83.3</td><td>83.5</td></tr><tr><td>CorpusQA 1M(ACC)</td><td>15.5</td><td>59.3</td><td>60.5</td><td>35.6</td><td>56.5</td><td>62.0</td></tr><tr><td rowspan="10">Agentic</td><td>Terminal Bench 2.0 (Acc)</td><td>49.1</td><td>56.6</td><td>56.9</td><td>59.1</td><td>63.3</td><td>67.9</td></tr><tr><td>SWE Verified (Resolved)</td><td>73.7</td><td>78.6</td><td>79.0</td><td>73.6</td><td>79.4</td><td>80.6</td></tr><tr><td>SWE Pro (Resolved)</td><td>49.1</td><td>52.3</td><td>52.6</td><td>52.1</td><td>54.4</td><td>55.4</td></tr><tr><td>SWE Multilingual (Resolved)</td><td>69.7</td><td>70.2</td><td>73.3</td><td>69.8</td><td>74.1</td><td>76.2</td></tr><tr><td>BrowseComp (Pass@1)</td><td>-</td><td>53.5</td><td>73.2</td><td>-</td><td>80.4</td><td>83.4</td></tr><tr><td>HLE w/ tools (Pass@1)</td><td>-</td><td>40.3</td><td>45.1</td><td>-</td><td>44.7</td><td>48.2</td></tr><tr><td>MCPAtlas Public (Pass@1)</td><td>64.0</td><td>67.4</td><td>69.0</td><td>69.4</td><td>74.2</td><td>73.6</td></tr><tr><td>GDPval-AA (Elo)</td><td>-</td><td>-</td><td>1395</td><td>-</td><td>-</td><td>1554</td></tr><tr><td>Toolathlon (Pass@1)</td><td>40.7</td><td>43.5</td><td>47.8</td><td>46.3</td><td>49.0</td><td>51.8</td></tr></tbody></table>

The comparison of DeepSeek-V4-Pro-Max and other closed/open source models is presented in Table 6. Also, we evaluate different modes of DeepSeek-V4-Flash and DeepSeek-V4-Pro and show the results in Table 7.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x8.png)

Figure 8: Formal reasoning under practical and frontier regimes. Left: Putnam-200 Pass@8 evaluates a fixed random subset of PutnamBench \[ tsoukalas2024putnambenchevaluatingneuraltheoremprovers \] following the setup introduced by Seed-Prover; all models are tested on the same problem set. We follow the Seed-Prover protocol but replace proprietary search tools with the open-source LeanExplore Asher\_LeanExplore\_2025, yielding a lightweight setting with minimal agent tools and bounded sampling. Right: Putnam-2025 probes the frontier of mathematical reasoning in a scaled hybrid formal-informal regime, where informal reasoning is combined with formal verification to expose gaps and improve rigor; DeepSeek-V4 reaches a proof-perfect 120/120.

##### Knowledge.

In the evaluation of general world knowledge, DeepSeek-V4-Pro-Max, the maximum reasoning effort mode of DeepSeek-V4-Pro, establishes a new state-of-the-art among open-source large language models. As demonstrated by the SimpleQA-Verified, DeepSeek-V4-Pro-Max significantly outperforms all existing open-source baselines by a margin of 20 absolute percentage points. Despite these advances, it currently trails the leading proprietary model, Gemini-3.1-Pro. In the domain of educational knowledge and reasoning, DeepSeek-V4-Pro-Max marginally outperforms Kimi and GLM across the MMLU-Pro, GPQA, and HLE benchmarks, although it lags behind leading proprietary models. Broadly, DeepSeek-V4-Pro-Max marks a significant milestone in enhancing the world knowledge capabilities of open-source models.

In addition, a significant performance gap exists between DeepSeek-V4-Flash and DeepSeek-V4-Pro on knowledge-based tasks; this is anticipated, as larger parameter counts facilitate greater knowledge retention during pre-training. Notably, both models demonstrate improved results on knowledge benchmarks when allocated higher reasoning effort.

##### Reasoning.

DeepSeek-V4-Pro-Max outperforms all prior open models across reasoning benchmarks, and matches state-of-the-art closed models on many metrics, while the smaller DeepSeek-V4-Flash-Max also surpasses the previous best open-source model, K2.6-Thinking, on code and math reasoning tasks. Meanwhile, DeepSeek-V4-Pro and DeepSeek-V4-Flash excel in coding competitions. According to our evaluation, their performance is comparable to GPT-5.4, making this the first time an open model has matched a closed model on this task. On the Codeforces leaderboard, DeepSeek-V4-Pro-Max currently ranks 23rd among human candidates. DeepSeek-V4 also demonstrates strong performance on formal mathematical task under both agentic and compute-intensive settings. Under an agentic setup, it achieves state-of-the-art results, shown in Figure 8, outperforming prior models such as Seed Prover \[chen2025seedprover15masteringundergraduatelevel\]. With a more compute-intensive pipeline, performance further improves, surpassing systems including Aristotle \[achim2025aristotle\] and matching the best known results under this setting.

##### Agent.

The DeepSeek-V4 series demonstrates strong agent performance in evaluations. For code agent tasks, DeepSeek-V4-Pro achieves results comparable to K2.6 and GLM-5.1, though all these open models still lag behind their closed-source counterparts. DeepSeek-V4-Flash underperforms DeepSeek-V4-Pro on coding tasks, particularly on Terminal Bench 2.0. A similar trend is observed across other agent evaluations. It is worth noting that DeepSeek-V4-Pro performs well on MCPAtlas and Toolathlon—two evaluation test sets that include a wide range of tools and MCP services—indicating that our model has excellent generalization capability and does not perform well only on internal frameworks.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x10.png)

Figure 9: DeepSeek-V4 series performance on the MRCR task.

##### 1M-Token Context.

DeepSeek-V4-Pro outperforms Gemini-3.1-Pro on the MRCR task, which measures in-context retrieval, but remains behind Claude Opus 4.6. As illustrated in Figure 9, retrieval performance remains highly stable within a 128K context window. While a performance degradation becomes visible beyond the 128K mark, the model’s retrieval capabilities at 1M tokens remain remarkably strong compared to both proprietary and open-source counterparts. Unlike MRCR, CorpusQA is similar to real scenarios. The evaluation results also indicate that DeepSeek-V4-Pro is better than Gemini-3.1-Pro.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x11.png)

Figure 10: HLE and Terminal Bench 2.0 performance by reasoning effort. “None” indicates Non-think mode, and “Speciale” indicates DeepSeek-V3.2-Speciale model.

##### Reasoning Effort.

As shown in Table 7, the Max mode, which employs longer contexts and reduced length penalties in RL, outperforms the High mode on the most challenging tasks. Figure 10 presents a comparison of performance and cost among DeepSeek-V4-Pro, DeepSeek-V4-Flash, and DeepSeek-V3.2 on representative reasoning and agentic tasks. By scaling test-time compute, DeepSeek-V4 series achieve substantial improvements over the predecessor. Furthermore, on reasoning tasks like HLE, DeepSeek-V4-Pro demonstrates higher token efficiency than DeepSeek-V3.2.

### 5.4 Performance on Real-World Tasks

Standardized benchmarks often struggle to capture the complexities of diverse, real-world tasks, creating a gap between test results and actual user experience. To bridge this, we have developed proprietary internal metrics that prioritize real-world usage patterns over traditional benchmarks. This approach ensures that our optimizations translate into tangible benefits. Our evaluation framework specifically targets the primary use cases of the DeepSeek API and Chatbot, aligning model performance with practical demands.

#### 5.4.1 Chinese Writing

One of the primary use cases for DeepSeek is Chinese writing. We conducted a rigorous evaluation on functional writing and creative writing. Table 12 presents a pairwise comparison between DeepSeek-V4-Pro and Gemini-3.1-Pro on functional writing tasks. These tasks consist of common daily writing queries, where prompts are typically concise and straightforward. Gemini-3.1-Pro was selected as the baseline, as it stands as the top-performing external model for Chinese writing in our evaluations. The results indicate that DeepSeek-V4-Pro outperforms the baseline with an overall win rate of 62.7% versus 34.1%; this is primarily because Gemini occasionally allows its inherent stylistic preferences to override the user’s explicit requirements in Chinese writing scenarios.

Table 13 presents the creative writing comparison, which is evaluated along two axes: instruction following and writing quality. Compared with Gemini-3.1-Pro, DeepSeek-V4-Pro achieves a 60.0% win rate in instruction following and 77.5% in writing quality, demonstrating a marginal improvement in instruction following and a substantial gain in writing quality. Although DeepSeek-V4-Pro yields superior results in aggregate user case analysis, an evaluation restricted to the most challenging prompts — specifically those involving high-complexity constraints or multi-turn scenarios — reveals that Claude Opus 4.5 retains a performance advantage over DeepSeek-V4-Pro. As shown in Table 14, Claude Opus 4.5 achieves a 52.0% win rate versus 45.9%.

#### 5.4.2 Search

Search-augmented question answering is a core capability of the DeepSeek chatbot. On the DeepSeek web and app, the "non-think" mode employs Retrieval-Augmented Search (RAG), whereas the "thinking" mode utilizes agentic search.

##### Retrieval Augmented Search.

We conducted a pairwise evaluation comparing DeepSeek-V4-Pro and DeepSeek-V3.2 across both objective and subjective Q&A categories. As presented in Table 11, DeepSeek-V4-Pro outperforms DeepSeek-V3.2 by a substantial margin, demonstrating a consistent advantage across both categories. The most pronounced gains are observed in single-value search and planning & strategy tasks, suggesting that DeepSeek-V4-Pro excels at locating precise factual answers and synthesizing structured plans from retrieved context. However, DeepSeek-V3.2 remains relatively competitive on comparison and recommendation tasks, indicating potential room for improvement for DeepSeek-V4-Pro in scenarios requiring balanced, multi-perspective reasoning over search results.

##### Agentic Search.

Unlike standard RAG, agentic search empowers the model to iteratively invoke search and fetch tools per query, significantly enhancing overall search performance. For the thinking mode in DeepSeek-Chat, we optimized the agentic search function to maximize response accuracy within a predefined "thinking budget". As shown in Table 9, agentic search consistently outperforms RAG, particularly on complex tasks. Furthermore, its cost remains highly efficient, with agentic search being only marginally more expensive than standard RAG (see Table 10).

#### 5.4.3 White-Collar Task

To rigorously evaluate the model’s utility in sophisticated enterprise productivity scenarios, we constructed a comprehensive suite of 30 advanced Chinese professional tasks. These workflows deliberately encompass high-level cognitive demands, including in-depth information analysis, comprehensive document generation, and nuanced document editing, spanning a diverse spectrum of 13 critical industries (e.g., finance, education, law, and technology). The evaluation was conducted within an in-house agent harness equipped with basic tools, including Bash and web search.

Given the open-ended nature of these tasks, automated metrics usually fall short in capturing the nuances of a high-quality response. Therefore, we conducted human evaluations to compare the performance of DeepSeek-V4-Pro-Max against Opus-4.6-Max. Annotators blindly assessed the model outputs across four dimensions:

- Task Completion: Whether the core problem was successfully resolved.
- Instruction Following: Adherence to specific constraints and directives.
- Content Quality: Factual accuracy, logical coherence, and professional tone.
- Formatting Aesthetics: Layout readability and visual presentation.

As illustrated in Figure 12, DeepSeek-V4-Pro-Max outperforms Opus-4.6-Max on diverse Chinese white-collar tasks, achieving an impressive non-loss rate of 63%, and demonstrating consistent advantages across analysis, generation, and editing tasks. The detailed dimension scores shown in Figure 12 highlight the model’s primary strengths in Task Completion and Content Quality. Specifically, DeepSeek-V4-Pro-Max proactively anticipates implicit user intents by frequently providing supplementary insights and self-verification steps. It also excels in long-form generation, delivering in-depth, coherent narratives rather than relying on the overly simplistic bullet points frequently produced by Opus-4.6-Max. Additionally, the model strictly conforms to formal professional conventions, such as standardized Chinese hierarchical numbering. However, in terms of Instruction Following, it occasionally overlooks specific formatting constraints and slightly trails Opus. Furthermore, the model is less proficient at condensing extensive text inputs into succinct summaries. Finally, its Formatting Aesthetics still have substantial room for improvement regarding the overall visual design of presentation slides. Figure 13, 14, and 15 present several test cases; due to the extensive length of certain outputs, only partial pages are displayed.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x12.png)

Figure 11: Win-rate comparison across analysis, generation, editing tasks, and the overall performance.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x14.png)

Figure 13: Example output of a task which requires drafting a joint marketing proposal for a popular bubble tea brand and the Beijing Subway.

#### 5.4.4 Code Agent

To benchmark our coding agent capability, we curate tasks from real internal R&D workloads We collect ${\sim}$ 200 challenging tasks from 50+ internal engineers, spanning feature development, bug fixing, refactoring, and diagnostics across diverse technology stacks including PyTorch, CUDA, Rust, and C++. Each task is accompanied by its original repository, the corresponding execution environment, and human-annotated scoring rubrics; after rigorous quality filtering, 30 tasks are retained as the evaluation set. As shown in Table 8, DeepSeek-V4-Pro significantly outperforms Claude Sonnet 4.5 and approaches the level of Claude Opus 4.5.

Table 8: Comparison on R&D Coding Benchmark (external models included strictly for evaluation purposes).

| Model | Haiku 4.5 | Sonnet 4.5 | DeepSeek-V4-Pro-Max | Opus 4.5 | Opus 4.5 Thinking | Opus 4.6 Thinking |
| --- | --- | --- | --- | --- | --- | --- |
| Pass Rate (%) | 13 | 47 | 67 | 70 | 73 | 80 |

In a survey asking DeepSeek developers and researchers ($N=85$) — all with experience of using DeepSeek-V4-Pro for agentic coding in their daily work — whether DeepSeek-V4-Pro is ready to serve as their default and primary coding model compared to other frontier models, 52% said yes, 39% leaned toward yes, and fewer than 9% said no. Respondents find DeepSeek-V4-Pro to deliver satisfactory results across most tasks, but note trivial mistakes, misinterpretation of vague prompts, and occasional over-thinking.

## 6 Conclusion, Limitations, and Future Directions

In this work, we present a preview version of DeepSeek-V4 series, aiming at next-generation large language models that break the efficiency barrier of ultra-long-context processing. By combining a hybrid attention architecture that integrates CSA and HCA, DeepSeek-V4 series achieve a dramatic leap in long-sequence efficiency. The architectural innovations, together with extensive infrastructure optimization, enable efficient native support for million-token contexts and establish a necessary foundation for future test-time scaling, long-horizon tasks, and emerging paradigms such as online learning. Evaluation results demonstrate that DeepSeek-V4-Pro-Max, the maximum reasoning effort mode of DeepSeek-V4-Pro, redefines the state-of-the-art for open models. It substantially outperforms prior open-source models on knowledge benchmarks, achieves superior reasoning performance close to the frontier proprietary models, and delivers competitive agent capabilities. Meanwhile, DeepSeek-V4-Flash-Max attains comparable reasoning performance to leading closed models while maintaining a highly cost-efficient architecture. We believe DeepSeek-V4 series usher in a new era of million-length contexts for open models and pave the way toward better efficiency, scale, and intelligence.

In pursuit of extreme long-context efficiency, DeepSeek-V4 series adopted a bold architectural design. To minimize risk, we retained many preliminarily validated components and tricks, which, while effective, made the architecture relatively complex. In future iterations, we will carry out more comprehensive and principled investigations to distill the architecture down to its most essential designs, making it more elegant without sacrificing performance. Meanwhile, although Anticipatory Routing and SwiGLU Clamping have been proven effective in mitigating training instabilities, their underlying principles remain insufficiently understood. We will actively study foundational problems on training stability and strengthen internal metric monitoring, aiming for a more principled and predictive approach to stable large-scale training. In addition, beyond the MoE and sparse attention architecture, we will also proactively explore model sparsity along new dimensions — such as more sparse embedding modules \[engram\] — to further improve computational and memory efficiency without compromising capability. We will also continuously investigate low-latency architectures and system techniques to make long-context deployment and interaction more responsive. Furthermore, we recognize the importance and practical value of long-horizon, multi-round agentic tasks, and will continue to iterate and explore in this direction. We are also working on incorporating multimodal capabilities to our models. Finally, we are committed to developing better data curation and synthesis strategies to consistently enhance model intelligence, robustness, and practical usability across an increasingly broad range of scenarios and tasks.

## References

## Appendix

## Appendix A Author List and Acknowledgment

### A.1 Author List

Authors are listed alphabetically by their first name. Names marked with \* denote individuals who have departed from our team.

Research & Engineering: Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang\*, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chenze Shao, Chong Ruan\*, Conner Sun, Damai Dai, Daya Guo\*, Dejian Yang, Deli Chen, Donghao Li, Erhang Li, Fangyun Lin, Fangzhou Yuan, Feiyu Xia, Fucong Dai, Guangbo Hao, Guanting Chen, Guoai Cao, Guolai Meng, Guowei Li, Han Yu, Han Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoling Zhang, Haoming Luo, Haoran Wei\*, Haotian Yuan, Haowei Zhang\*, Haowen Luo, Haoyu Chen, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, J. Yang, J.Q. Zhu, Jia Yu, Jialiang Huang, Jiasheng Ye, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jin Yan, Jingchang Chen, Jingli Zhou, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jiping Yu, Joseph Sun, Jun Ran\*, Junguang Jiang, Junjie Qiu, Junlong Li\*, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexing Zhou, Kezhao Huang\*, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Li Zhang, Liang Zhao, Lihua Guo, Lingxiao Luo, Linwang Ma, Litong Wang, Liyu Cai, Liyue Zhang, Longhao Chen, M.S. Di, M.Y Xu, Max Mei, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Panpan Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Qiwei Jiang, Rui Tian, Ruifan Xu, Ruijie Lu, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqian Chen, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S.H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoheng Nie, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Shuying Yu, Songyang Zhou, Tao Ni, Tao Yun, Tian Jin, Tian Pei, Tian Ye, Tianle Lin, Tianran Ji, Tianyi Cui, Tianyuan Yue, Tingting Yu, Tun Wang, W. Zhang, Wangding Zeng, Weilin Zhao, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjing Yao, Wenjun Gao, Wenkai Yang, Wenlve Huang, Wentao Zhang, Wenting Ma, Xi Gao, Xiang He, Xiangwen Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingchen Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xu Chen, Xuanyu Wang, Xuecheng Su, Xuheng Lin, Xuwei Fu, Y.C. Yan, Y.Q. Wang\*, Y.W. Ma, Yanfeng Luo, Yang Zhang, Yanhong Xu, Yanru Ma, Yanwen Huang, Yao Li, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yijia Wu, Yiliang Xiong, Ying He, Ying Zhou, Yingjia Luo, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiang Zhang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuanhao Li, Yuduan Wang, Yuhan Wu, Yuhao Meng, Yuheng Zou, YuKun Li, Yunfan Xiong, Yupeng Chen, Yuqian Cao, Yuqian Wang, Yushun Zhang, Yutong Lin, Yuxian Gu, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuxuan Zhou, Yuyang Zhou, Yuzhen Huang, Z.F. Wu, Zehao Wang, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhixuan Chen, Zhiyu Wu, Zhizhou Ren, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang\*, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao.

Business & Compliance: Chenchen Ling, Chengyu Hou, Dongjie Ji, Fang Wei, Hengqing Zhang, Jia Luo, Jia Song, Jialu Cai, Jian Liang, Jiangting Zhou, Jieyu Yang, Jin Chen, Jingzi Zhou, Junmin Zheng, Leyi Xia, Linyan Zhu, Miaojun Wang, Mingming Li, Minmin Han, Ning Wang, Panpan Wang, Peng Zhang, Ruyi Chen, Shangmian Sun, Shaoqing Wu, W.L. Xiao, Wei An, Wenqing Hou, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Xinyu Zhang, Xueyin Chen, Yao Xu, Yi Shao, Yiling Ma, Ying Tang, Yuehan Yang, Yuer Xu, Yukun Zha, Yuping Lin, Yuting Yan, Zekai Zhang, Zhe Ju, Zheren Gao, Zhongyu Wu, Zihua Qu, Ziyi Wan.

### A.2 Acknowledgment

We would like to thank [Dolly Deng](https://www.zhihu.com/people/toyama) and other testers for their valuable suggestions and feedback regarding the capabilities of DeepSeek-V4 series models.

## Appendix B Evaluation Details

Table 9: Agentic Search vs. Retrieval Augmented Search for DeepSeek-V4-Pro.

<table><tbody><tr><td>Difficulty</td><td>Category</td><td>#</td><td>Agent Win</td><td>RAG Win</td><td>Tie</td><td>Agent%</td><td>RAG%</td><td>Tie%</td></tr><tr><td rowspan="2">Easy</td><td>Objective Q&A (客观问答)</td><td>196</td><td>110</td><td>43</td><td>43</td><td>56.1</td><td>21.9</td><td>21.9</td></tr><tr><td>Subjective Q&A (主观问答)</td><td>321</td><td>198</td><td>56</td><td>67</td><td>61.7</td><td>17.4</td><td>20.9</td></tr><tr><td rowspan="2">Hard</td><td>Objective Q&A (客观问答)</td><td>168</td><td>102</td><td>33</td><td>33</td><td>60.7</td><td>19.6</td><td>19.6</td></tr><tr><td>Subjective Q&A (主观问答)</td><td>184</td><td>126</td><td>27</td><td>31</td><td>68.5</td><td>14.7</td><td>16.8</td></tr><tr><td colspan="2">Total (总计)</td><td>869</td><td>536</td><td>159</td><td>174</td><td>61.7</td><td>18.3</td><td>20.0</td></tr></tbody></table>

Table 10: Cost Comparison:Agentic Search vs. Retrieval Augmented Search (Mean) for DeepSeek-V4-Pro. Most of the tool calls are parallel for Agentic Search.

| Version | Tool Calls | Prefill (tokens) | Output (tokens) |
| --- | --- | --- | --- |
| V4 Agentic Search | 16.2 | 13649 | 1526 |
| V4 Retrieval Augmented Search | — | 10453 | 1308 |

Table 11: Comparative Evaluation of DeepSeek-V4-Pro and DeepSeek-V3.2 on Search Q&A Tasks.

<table><tbody><tr><td></td><td></td><td></td><td colspan="6">Internal Evaluation (内部综合评估)</td></tr><tr><td>Category</td><td>Subcategory</td><td>#</td><td>V4 win</td><td>V3.2 win</td><td>tie</td><td>V4%</td><td>V3.2%</td><td>tie%</td></tr><tr><td rowspan="4">Objective Q&A (客观问答)</td><td>Single-value Search (单值信息查找)</td><td>95</td><td>36</td><td>10</td><td>49</td><td>37.9</td><td>10.5</td><td>51.6</td></tr><tr><td>Entity Search (实体信息查找)</td><td>99</td><td>24</td><td>7</td><td>68</td><td>24.2</td><td>7.1</td><td>68.7</td></tr><tr><td>Enumerative Search (枚举型信息查找)</td><td>95</td><td>19</td><td>8</td><td>68</td><td>20.0</td><td>8.4</td><td>71.6</td></tr><tr><td>Subtotal (小计)</td><td>289</td><td>79</td><td>25</td><td>185</td><td>27.3</td><td>8.7</td><td>64.0</td></tr><tr><td rowspan="8">Subjective Q&A (主观问答)</td><td>Causal Analysis (原因分析)</td><td>100</td><td>28</td><td>5</td><td>67</td><td>28.0</td><td>5.0</td><td>67.0</td></tr><tr><td>Comparison (对比)</td><td>96</td><td>28</td><td>20</td><td>48</td><td>29.2</td><td>20.8</td><td>50.0</td></tr><tr><td>Advice Seeking (寻求建议)</td><td>92</td><td>23</td><td>8</td><td>61</td><td>25.0</td><td>8.7</td><td>66.3</td></tr><tr><td>Recommendation (推荐)</td><td>95</td><td>26</td><td>19</td><td>50</td><td>27.4</td><td>20.0</td><td>52.6</td></tr><tr><td>Planning & Strategy (攻略计划)</td><td>92</td><td>32</td><td>11</td><td>49</td><td>34.8</td><td>12.0</td><td>53.3</td></tr><tr><td>Opinion & Evaluation (评价看法)</td><td>96</td><td>30</td><td>8</td><td>58</td><td>31.2</td><td>8.3</td><td>60.4</td></tr><tr><td>Trend Analysis (趋势分析)</td><td>96</td><td>23</td><td>3</td><td>70</td><td>24.0</td><td>3.1</td><td>72.9</td></tr><tr><td>Subtotal (小计)</td><td>667</td><td>190</td><td>74</td><td>403</td><td>28.5</td><td>11.1</td><td>60.4</td></tr><tr><td colspan="2">TOTAL (总计)</td><td>956</td><td>269</td><td>99</td><td>588</td><td>28.1</td><td>10.4</td><td>61.5</td></tr></tbody></table>

Table 12: Comparative Analysis of DeepSeek-V4-Pro and Gemini-3.1-Pro in Chinese Functional Writing.

<table><tbody><tr><td></td><td></td><td></td><td colspan="6">Internal Evaluation (内部综合评估)</td></tr><tr><td>Category</td><td>Subcategory</td><td>#</td><td>DS win</td><td>Gem win</td><td>Tie</td><td>DS%</td><td>Gem%</td><td>Tie%</td></tr><tr><td rowspan="10">Business Writing (办公文本)</td><td>Report (报告)</td><td>527</td><td>350</td><td>162</td><td>15</td><td>66.41</td><td>30.74</td><td>2.85</td></tr><tr><td>Proposal (方案策划)</td><td>291</td><td>181</td><td>103</td><td>7</td><td>62.20</td><td>35.40</td><td>2.41</td></tr><tr><td>Education (教育培训)</td><td>159</td><td>100</td><td>56</td><td>3</td><td>62.89</td><td>35.22</td><td>1.89</td></tr><tr><td>Email & Letter (邮件书信)</td><td>146</td><td>107</td><td>37</td><td>2</td><td>73.29</td><td>25.34</td><td>1.37</td></tr><tr><td>Notice (通知公告)</td><td>72</td><td>43</td><td>24</td><td>5</td><td>59.72</td><td>33.33</td><td>6.94</td></tr><tr><td>Professional (专业文本)</td><td>63</td><td>34</td><td>27</td><td>2</td><td>53.97</td><td>42.86</td><td>3.17</td></tr><tr><td>Recruitment (招聘求职)</td><td>42</td><td>27</td><td>15</td><td>0</td><td>64.29</td><td>35.71</td><td>0.00</td></tr><tr><td>Technical (技术文本)</td><td>29</td><td>22</td><td>7</td><td>0</td><td>75.86</td><td>24.14</td><td>0.00</td></tr><tr><td>Review (介绍评价)</td><td>20</td><td>15</td><td>5</td><td>0</td><td>75.00</td><td>25.00</td><td>0.00</td></tr><tr><td>Subtotal (小计)</td><td>1349</td><td>879</td><td>436</td><td>34</td><td>65.16</td><td>32.32</td><td>2.52</td></tr><tr><td rowspan="9">Media Writing (媒体文本)</td><td>Social Media (社交媒体文案)</td><td>267</td><td>156</td><td>101</td><td>10</td><td>58.43</td><td>37.83</td><td>3.75</td></tr><tr><td>Ad Copy (广告商品文案)</td><td>214</td><td>109</td><td>98</td><td>7</td><td>50.93</td><td>45.79</td><td>3.27</td></tr><tr><td>Long-form Content (内容平台长文)</td><td>99</td><td>71</td><td>25</td><td>3</td><td>71.72</td><td>25.25</td><td>3.03</td></tr><tr><td>News Report (新闻报道)</td><td>51</td><td>27</td><td>22</td><td>2</td><td>52.94</td><td>43.14</td><td>3.92</td></tr><tr><td>Advertorial (营销软文)</td><td>17</td><td>12</td><td>4</td><td>1</td><td>70.59</td><td>23.53</td><td>5.88</td></tr><tr><td>Headline (标题)</td><td>11</td><td>7</td><td>4</td><td>0</td><td>63.64</td><td>36.36</td><td>0.00</td></tr><tr><td>Narration Script (口播文案)</td><td>4</td><td>2</td><td>1</td><td>1</td><td>50.00</td><td>25.00</td><td>25.00</td></tr><tr><td>Comment (评论)</td><td>3</td><td>2</td><td>1</td><td>0</td><td>66.67</td><td>33.33</td><td>0.00</td></tr><tr><td>Subtotal (小计)</td><td>666</td><td>386</td><td>256</td><td>24</td><td>57.96</td><td>38.44</td><td>3.60</td></tr><tr><td rowspan="6">Everyday Writing (生活文本)</td><td>Congratulatory (祝贺文本)</td><td>101</td><td>54</td><td>41</td><td>6</td><td>53.47</td><td>40.59</td><td>5.94</td></tr><tr><td>Communication (沟通回复)</td><td>100</td><td>71</td><td>26</td><td>3</td><td>71.00</td><td>26.00</td><td>3.00</td></tr><tr><td>Reflection (心得感想)</td><td>90</td><td>68</td><td>17</td><td>5</td><td>75.56</td><td>18.89</td><td>5.56</td></tr><tr><td>Review (介绍评价)</td><td>55</td><td>44</td><td>9</td><td>2</td><td>80.00</td><td>16.36</td><td>3.64</td></tr><tr><td>Comment (评论)</td><td>44</td><td>34</td><td>8</td><td>2</td><td>77.27</td><td>18.18</td><td>4.55</td></tr><tr><td>Subtotal (小计)</td><td>390</td><td>271</td><td>101</td><td>18</td><td>69.49</td><td>25.90</td><td>4.62</td></tr><tr><td rowspan="6">Oral Writing (口头文本)</td><td>Speech (发言稿)</td><td>226</td><td>135</td><td>85</td><td>6</td><td>59.73</td><td>37.61</td><td>2.65</td></tr><tr><td>Narration Script (口播文案)</td><td>51</td><td>25</td><td>23</td><td>3</td><td>49.02</td><td>45.10</td><td>5.88</td></tr><tr><td>Sales Script (话术)</td><td>31</td><td>22</td><td>6</td><td>3</td><td>70.97</td><td>19.35</td><td>9.68</td></tr><tr><td>Dialogue (对话文本)</td><td>10</td><td>4</td><td>6</td><td>0</td><td>40.00</td><td>60.00</td><td>0.00</td></tr><tr><td>Congratulatory (祝贺文本)</td><td>1</td><td>1</td><td>0</td><td>0</td><td>100.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Subtotal (小计)</td><td>319</td><td>187</td><td>120</td><td>12</td><td>58.62</td><td>37.62</td><td>3.76</td></tr><tr><td rowspan="6">Official Document (公文文本)</td><td>Administrative Doc (事务文书)</td><td>117</td><td>60</td><td>53</td><td>4</td><td>51.28</td><td>45.30</td><td>3.42</td></tr><tr><td>Personal Doc (个人文书)</td><td>73</td><td>45</td><td>27</td><td>1</td><td>61.64</td><td>36.99</td><td>1.37</td></tr><tr><td>Government Doc (行政公文)</td><td>34</td><td>19</td><td>14</td><td>1</td><td>55.88</td><td>41.18</td><td>2.94</td></tr><tr><td>Speech (发言稿)</td><td>3</td><td>1</td><td>2</td><td>0</td><td>33.33</td><td>66.67</td><td>0.00</td></tr><tr><td>Essay Writing (申论写作)</td><td>3</td><td>1</td><td>1</td><td>1</td><td>33.33</td><td>33.33</td><td>33.33</td></tr><tr><td>Subtotal (小计)</td><td>230</td><td>126</td><td>97</td><td>7</td><td>54.78</td><td>42.17</td><td>3.04</td></tr><tr><td rowspan="5">Academic Writing (学术文本)</td><td>Research Paper (学术论文)</td><td>104</td><td>67</td><td>32</td><td>5</td><td>64.42</td><td>30.77</td><td>4.81</td></tr><tr><td>Coursework (课程作业)</td><td>90</td><td>53</td><td>35</td><td>2</td><td>58.89</td><td>38.89</td><td>2.22</td></tr><tr><td>Academic Support (学术辅助)</td><td>15</td><td>11</td><td>3</td><td>1</td><td>73.33</td><td>20.00</td><td>6.67</td></tr><tr><td>Science Outreach (专业科普)</td><td>7</td><td>6</td><td>1</td><td>0</td><td>85.71</td><td>14.29</td><td>0.00</td></tr><tr><td>Subtotal (小计)</td><td>216</td><td>137</td><td>71</td><td>8</td><td>63.43</td><td>32.87</td><td>3.70</td></tr><tr><td colspan="2">Total (总计)</td><td>3170</td><td>1986</td><td>1081</td><td>103</td><td>62.65</td><td>34.10</td><td>3.25</td></tr></tbody></table>

Table 13: Comparative Analysis of DeepSeek-V4-Pro and Gemini-3.1-Pro in Chinese Creative Writing.

<table><tbody><tr><td></td><td></td><td colspan="6">Instruction Following(指令遵循)</td><td colspan="6">Writing Quality (写作质量)</td></tr><tr><td>Subcategory (文体)</td><td>#</td><td>DS</td><td>Gem</td><td>Tie</td><td>DS%</td><td>Gem%</td><td>Tie%</td><td>DS</td><td>Gem</td><td>Tie</td><td>DS%</td><td>Gem%</td><td>Tie%</td></tr><tr><td>Fiction (小说故事)</td><td>836</td><td>504</td><td>323</td><td>5</td><td>60.58</td><td>38.82</td><td>0.60</td><td>672</td><td>157</td><td>3</td><td>80.77</td><td>18.87</td><td>0.36</td></tr><tr><td>General Fiction (泛小说故事)</td><td>662</td><td>368</td><td>290</td><td>3</td><td>55.67</td><td>43.87</td><td>0.45</td><td>467</td><td>194</td><td>0</td><td>70.65</td><td>29.35</td><td>0.00</td></tr><tr><td>Fan Fiction (同人文)</td><td>410</td><td>253</td><td>150</td><td>3</td><td>62.32</td><td>36.95</td><td>0.74</td><td>338</td><td>67</td><td>1</td><td>83.25</td><td>16.50</td><td>0.25</td></tr><tr><td>General Fan Fic. (泛同人文)</td><td>202</td><td>111</td><td>90</td><td>1</td><td>54.95</td><td>44.55</td><td>0.50</td><td>161</td><td>40</td><td>1</td><td>79.70</td><td>19.80</td><td>0.50</td></tr><tr><td>Narrative (记叙文)</td><td>171</td><td>115</td><td>54</td><td>2</td><td>67.25</td><td>31.58</td><td>1.17</td><td>141</td><td>30</td><td>0</td><td>82.46</td><td>17.54</td><td>0.00</td></tr><tr><td>General Prose (泛散文)</td><td>124</td><td>83</td><td>40</td><td>1</td><td>66.94</td><td>32.26</td><td>0.81</td><td>88</td><td>36</td><td>0</td><td>70.97</td><td>29.03</td><td>0.00</td></tr><tr><td>Prose (散文)</td><td>112</td><td>74</td><td>38</td><td>0</td><td>66.07</td><td>33.93</td><td>0.00</td><td>92</td><td>20</td><td>0</td><td>82.14</td><td>17.86</td><td>0.00</td></tr><tr><td>Writing Style (文笔)</td><td>112</td><td>81</td><td>31</td><td>0</td><td>72.32</td><td>27.68</td><td>0.00</td><td>86</td><td>26</td><td>0</td><td>76.79</td><td>23.21</td><td>0.00</td></tr><tr><td>Classical Poetry (古诗文)</td><td>48</td><td>24</td><td>24</td><td>0</td><td>50.00</td><td>50.00</td><td>0.00</td><td>39</td><td>9</td><td>0</td><td>81.25</td><td>18.75</td><td>0.00</td></tr><tr><td>Modern Poetry (现代诗)</td><td>43</td><td>23</td><td>20</td><td>0</td><td>53.49</td><td>46.51</td><td>0.00</td><td>32</td><td>11</td><td>0</td><td>74.42</td><td>25.58</td><td>0.00</td></tr><tr><td>Lyrics (歌词)</td><td>30</td><td>8</td><td>22</td><td>0</td><td>26.67</td><td>73.33</td><td>0.00</td><td>16</td><td>14</td><td>0</td><td>53.33</td><td>46.67</td><td>0.00</td></tr><tr><td>Literary Appreciation (赏析)</td><td>27</td><td>20</td><td>7</td><td>0</td><td>74.07</td><td>25.93</td><td>0.00</td><td>18</td><td>9</td><td>0</td><td>66.67</td><td>33.33</td><td>0.00</td></tr><tr><td>General Argument. (泛议论文)</td><td>24</td><td>15</td><td>9</td><td>0</td><td>62.50</td><td>37.50</td><td>0.00</td><td>17</td><td>7</td><td>0</td><td>70.83</td><td>29.17</td><td>0.00</td></tr><tr><td>General Narrative (泛记叙文)</td><td>23</td><td>11</td><td>12</td><td>0</td><td>47.83</td><td>52.17</td><td>0.00</td><td>15</td><td>8</td><td>0</td><td>65.22</td><td>34.78</td><td>0.00</td></tr><tr><td>General Classical (泛古文诗歌)</td><td>9</td><td>5</td><td>4</td><td>0</td><td>55.56</td><td>44.44</td><td>0.00</td><td>5</td><td>4</td><td>0</td><td>55.56</td><td>44.44</td><td>0.00</td></tr><tr><td>Creative Writing (创意写作)</td><td>6</td><td>2</td><td>4</td><td>0</td><td>33.33</td><td>66.67</td><td>0.00</td><td>4</td><td>2</td><td>0</td><td>66.67</td><td>33.33</td><td>0.00</td></tr><tr><td>Argumentative (议论文)</td><td>5</td><td>5</td><td>0</td><td>0</td><td>100.00</td><td>0.00</td><td>0.00</td><td>5</td><td>0</td><td>0</td><td>100.00</td><td>0.00</td><td>0.00</td></tr><tr><td>General Mod. Poetry (泛现代诗)</td><td>2</td><td>1</td><td>1</td><td>0</td><td>50.00</td><td>50.00</td><td>0.00</td><td>2</td><td>0</td><td>0</td><td>100.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Total (总计)</td><td>2837</td><td>1703</td><td>1119</td><td>15</td><td>60.03</td><td>39.44</td><td>0.53</td><td>2198</td><td>634</td><td>5</td><td>77.48</td><td>22.35</td><td>0.18</td></tr></tbody></table>

Table 14: DeepSeek-V4-Pro vs. Claude-Opus-4.5 on Complex Instruction Following and Multi-Turn Writing.

<table><thead><tr><th></th><th></th><th colspan="6">Internal Evaluation (内部综合评估)</th></tr><tr><th>Category</th><th>#</th><th>DS</th><th>Opus</th><th>Tie</th><th>DS%</th><th>Opus%</th><th>Tie%</th></tr></thead><tbody><tr><td>Complex Inst. Following (复杂指令跟随)</td><td>49</td><td>23</td><td>26</td><td>0</td><td>46.9%</td><td>53.1%</td><td>0.0%</td></tr><tr><td>Multi-Turn Writing (多轮写作)</td><td>147</td><td>67</td><td>76</td><td>4</td><td>45.6%</td><td>51.7%</td><td>2.7%</td></tr><tr><td>Total (总计)</td><td>196</td><td>90</td><td>102</td><td>4</td><td>45.9%</td><td>52.0%</td><td>2.0%</td></tr></tbody></table>

![Refer to caption](https://arxiv.org/html/2606.19348v1/x15.png)

Figure 14: Example output of a task that requires comparing two regular investment strategies for the NASDAQ.

![Refer to caption](https://arxiv.org/html/2606.19348v1/x16.png)

Figure 15: Example output of a task which requires researching 2020-2025 Nobel Science Prizes and generating an analytical PDF report.