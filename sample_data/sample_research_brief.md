# Research Brief: Sparse Windowed Attention for Efficient Long-Context Summarization

**Authors:** R. Menon, T. Okafor, S. Lindqvist
**Year:** 2025
**Venue:** Workshop on Efficient NLP Systems

## Research Analysis
**Problem Statement:** Standard transformer self-attention scales quadratically with sequence length, making summarization of long documents (papers, legal contracts) computationally expensive.

**Methodology:** Sparse Windowed Attention (SWA): each token attends to a local sliding window (w=256) plus a small set of k=32 learned 'anchor' tokens produced by pooling each 1,024-token block, giving linear-cost attention while preserving a pathway for global context.

**Hypothesis:** A learned (rather than fixed) global attention pathway can match full-attention summarization quality at a fraction of the compute cost.

**Key Experiments:**
- Benchmarked on three long-document summarization datasets: arXiv papers, GovReport, and PubMed
- Compared against full-attention, Longformer, and BigBird baselines at matched parameter count
- Ablation removing the anchor pathway to isolate its contribution

**Main Findings:**
- SWA cuts attention FLOPs by 71% and improves training throughput by 2.3x versus full attention
- ROUGE-L stays within 1-4 points of full attention across all three benchmarks
- SWA beats Longformer by 2.1 ROUGE-L points and BigBird by 3.4 points at equal sparsity budget
- Removing the anchor pathway (local window only) costs 8.7 ROUGE-L points, showing global context is necessary

## Executive Summary
Long documents such as research papers and legal contracts are expensive to summarize with standard transformers because self-attention cost grows quadratically with sequence length. This paper introduces Sparse Windowed Attention (SWA), which restricts each token to a local 256-token window plus a small set of 32 learned anchor tokens that compress distant context into compact summaries updated per 1,024-token block. This keeps attention cost linear while still letting information flow across an entire document. Evaluated on arXiv, GovReport, and PubMed summarization benchmarks, SWA cuts attention FLOPs by 71% and boosts training throughput by 2.3x relative to full attention, while staying within 1-4 ROUGE-L points of it. It also outperforms fixed-sparsity baselines Longformer and BigBird by 2.1 and 3.4 ROUGE-L points respectively. An ablation confirms the learned anchor pathway, not just the local window, is responsible for most of this quality, making SWA a practical option for efficient long-context summarization with a built-in interpretability signal from its anchor attention maps.

## Citations & References
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150. _(relevance: Fixed local+global sparse attention baseline compared against SWA)_
- Zaheer, M., et al. (2020). Big Bird: Transformers for Longer Sequences. NeurIPS 2020. _(relevance: Sparse attention baseline with theoretical expressiveness guarantees)_
- Kitaev, N., Kaiser, L., & Levskaya, A. (2020). Reformer: The Efficient Transformer. ICLR 2020. _(relevance: LSH-based attention approximation, related efficient-attention approach)_
- Jaegle, A., et al. (2021). Perceiver: General Perception with Iterative Attention. ICML 2021. _(relevance: Inspiration for the anchor/latent bottleneck pooling mechanism)_
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. _(relevance: Original transformer self-attention this work aims to make more efficient)_

## Key Insights
**Takeaways:**
- For long-document summarization pipelines, swapping full attention for a learned local+anchor scheme can cut compute substantially with modest quality loss
- Fixed-pattern sparse attention (Longformer/BigBird) leaves quality on the table versus a learned global pathway

**Implications:**
- Learned sparsity patterns generalize better across document types than hand-designed fixed patterns
- Anchor attention maps offer a cheap interpretability signal (which passages the model treats as globally salient) that dense attention does not provide

**Potential Applications:**
- Summarizing long legal contracts or research papers on commodity GPUs
- Enterprise document QA/search systems that need to process long inputs within tight latency budgets

## Quality Review Scores
- analysis: 9/10
- citations: 9/10
- summary: 9/10
- insights: 9/10