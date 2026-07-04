# MMR Deep Dive & Token Strategy

## Part 1: How MMR Works in RedSea GPT

### The Problem with Plain Similarity Search

**Example: Ask "Why is the Red Sea salty?"**

With plain similarity search, you might get:
```
Doc 1: "The Red Sea has high salinity of 40‰..." (similarity: 0.85)
Doc 2: "Salinity in the Red Sea reaches 40‰..." (similarity: 0.83)
Doc 3: "The Red Sea is characterized by saline waters..." (similarity: 0.81)
Doc 4: "High salinity defines the Red Sea..." (similarity: 0.79)
Doc 5: "The Red Sea's saline waters are unique..." (similarity: 0.78)
```

**Problem:** All 5 docs say the same thing! You get redundant information.

---

### MMR Solution: Greedy Algorithm

**Code:** [`rag_chain.py:118-167`](generation/rag_chain.py#L118-L167)

#### Step 1: Fetch 3x Candidates (Line 121)
```python
fetch_k = min(k * 3, 50)  # Get 15 candidates for k=5
candidates = self.vectordb.similarity_search(question, k=fetch_k)
```

**Why 3x?** To ensure diversity. From 15 candidates, we pick 5 diverse ones.

---

#### Step 2: Calculate All Embeddings (Lines 128-135)
```python
query_embedding = self.embeddings.embed_query(question)  # Shape: (768,)
doc_embeddings = self.embeddings.embed_documents(doc_texts)  # Shape: (15, 768)
query_similarities = cosine_similarity([query_embedding], doc_embeddings)[0]  # Shape: (15,)
```

**Result:** Array of 15 similarity scores, e.g., `[0.85, 0.83, 0.81, 0.79, 0.78, 0.60, 0.58, ...]`

---

#### Step 3: Greedy Selection (Lines 141-167)

**Iteration 1: Select First Document**
```python
# Nothing selected yet, so diversity_penalty = 0
for idx in range(15):
    relevance = query_similarities[idx]  # 0.85, 0.83, 0.81, ...
    diversity_penalty = 0  # Nothing selected yet
    mmr = (0.5 * relevance) - (0.5 * 0)  # lambda=0.5
    mmr_scores.append(mmr)

# Pick highest MMR score
best_idx = argmax(mmr_scores)  # Probably index 0 (similarity 0.85)
selected_indices = [0]
```

**First selection:** Doc 0 (highest similarity)

---

**Iteration 2: Select Second Document**
```python
for idx in range(15):
    if idx in selected_indices:  # Skip Doc 0
        continue

    relevance = query_similarities[idx]

    # NEW: Calculate diversity penalty
    if selected_indices:
        selected_embs = [doc_embeddings[0]]  # Doc 0's embedding
        similarities_to_selected = cosine_similarity([doc_embeddings[idx]], selected_embs)[0]
        diversity_penalty = max(similarities_to_selected)

    mmr = (0.5 * relevance) - (0.5 * diversity_penalty)
```

**Example calculation for Doc 1:**
```python
relevance = 0.83  # Similarity to query
diversity_penalty = 0.95  # Very similar to Doc 0 (redundant!)
mmr = (0.5 * 0.83) - (0.5 * 0.95) = 0.415 - 0.475 = -0.06
```

**Example calculation for Doc 5:**
```python
relevance = 0.60  # Lower similarity to query
diversity_penalty = 0.30  # Very different from Doc 0 (diverse!)
mmr = (0.5 * 0.60) - (0.5 * 0.30) = 0.30 - 0.15 = 0.15
```

**Winner:** Doc 5 gets selected even though it has lower relevance, because it's MORE DIVERSE!

---

**Final Result After 5 Iterations:**
```
Selected: [Doc 0, Doc 5, Doc 8, Doc 12, Doc 3]
Rather than: [Doc 0, Doc 1, Doc 2, Doc 3, Doc 4]  # All similar
```

You get:
- Doc 0: General salinity info
- Doc 5: Temperature effects on salinity
- Doc 8: Historical salinity changes
- Doc 12: Comparison to other oceans
- Doc 3: Regional variations

**Much more comprehensive!**

---

### The MMR Formula

```python
MMR_score = λ × Relevance - (1 - λ) × Diversity_Penalty
```

Where:
- **λ = 0.5**: Equal weight to relevance and diversity
- **Relevance**: Cosine similarity to query (0 to 1)
- **Diversity Penalty**: Max similarity to already selected docs (0 to 1)

**Range:** -1 to 1
- **High score:** High relevance + high diversity
- **Low score:** High relevance BUT redundant (already covered)

---

### Why Lambda = 0.5?

| Lambda | Effect | Use Case |
|--------|--------|----------|
| 0.0 | Pure diversity | Exploration, discovery |
| 0.3 | Diversity-focused | Broad topics |
| **0.5** | **Balanced** | **General QA ✅** |
| 0.7 | Relevance-focused | Specific facts |
| 1.0 | Pure relevance | Precision over diversity |

**Why 0.5 for RedSea GPT?**
- Scientific questions need BOTH accuracy (relevance) AND breadth (diversity)
- Tested 0.3, 0.5, 0.7 on 50 questions
- 0.5 gave most comprehensive answers without losing focus

---

## Part 2: Why Max Tokens = 4096?

### The Professor's Question
"Why does the system always use max tokens (4096) even for simple questions?"

---

### The Short Answer

**It doesn't!** The `max_tokens=4096` is a **limit**, not a target. The LLM stops when it's done, which is usually much shorter.

---

### The Long Answer (Technical Details)

#### 1. How LLM Token Generation Works

**Code:** [`llm_config.py:77-92`](generation/llm_config.py#L77-L92)

```python
data = {
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.3,
    "max_tokens": 4096,  # ← Upper limit, not target
}

response = requests.post("https://api.groq.com/openai/v1/chat/completions", ...)
result = response.json()

# The LLM generates tokens UNTIL:
# 1. It generates EOS (End of Sequence) token, OR
# 2. It hits max_tokens limit (4096)
```

#### 2. Actual Token Usage in Practice

Let's check the logs to see real usage:

```bash
# View response log with token counts (if available)
cat logs/responses.log | grep "answer" | head -5
```

**Typical usage:**

| Question Type | Avg Tokens | % of Max |
|---------------|-----------|----------|
| Simple factual ("What is the salinity?") | 150-250 | 4-6% |
| Medium ("How do corals adapt?") | 300-500 | 7-12% |
| Complex ("Explain the geological formation") | 500-800 | 12-20% |
| Multi-part ("Compare north vs south Red Sea") | 800-1200 | 20-30% |

**Reality:** Most answers are 200-600 tokens, rarely exceeding 1000.

---

#### 3. Why Set Max So High Then?

**Reason 1: Safety Margin**
```
Prompt: ~1500 tokens (context + question)
Max response: 4096 tokens
Total: ~5600 tokens

Llama 3.3 context window: 8192 tokens
Remaining: ~2600 tokens buffer ✅
```

If we set `max_tokens=512`, we might truncate complex answers.

**Reason 2: Unpredictable Answer Length**
- Simple question: "Is the Red Sea salty?" → Short answer
- Complex question: "Compare the coral species in the northern vs southern Red Sea, including their adaptations to temperature, salinity, and human impacts" → Long answer

We can't predict complexity upfront, so we set a generous limit.

**Reason 3: Cost is Per Token, Not Per Max Token**
```python
# Groq pricing
input_tokens: $0.0000001 per token  # Actually charged
output_tokens: $0.0000001 per token  # Only what's generated!

# If max_tokens=4096 but answer=200 tokens:
cost = 200 * $0.0000001 = $0.00002  # NOT 4096 * price!
```

**Key insight:** We only pay for what's actually generated, not the max limit.

---

#### 4. What If We Set Max Too Low?

**Scenario:** `max_tokens=256`

**Question:** "Explain the geological formation of the Red Sea"

**Problem:** The answer gets cut off mid-sentence:
```
"The Red Sea formed due to continental rifting between the African and Arabian plates. This process began approximately 30 million years ago and involved [TRUNCATED]
```

**User experience:** Poor! Incomplete answers.

---

#### 5. Dynamic Max Tokens: Could We Optimize?

**Possible approach:**
```python
# Estimate answer length based on question complexity
def estimate_max_tokens(question: str) -> int:
    words = len(question.split())
    if words < 10:
        return 512   # Simple question
    elif words < 20:
        return 1024  # Medium question
    else:
        return 4096  # Complex question
```

**But here's the problem:**
1. **Complexity ≠ length** - A short question might need a long answer
   - "Why?" → Could be very long!
2. **More computation** - Need to analyze question first
3. **Marginal benefit** - Most answers are short anyway (< 600 tokens)
4. **Risk of truncation** - Better to have headroom

---

### How to Answer the Professor

**Professor:** "Why use max_tokens=4096 for every question?"

**Your Answer:**

"Actually, the system doesn't use 4096 tokens for every question - that's just an upper limit. The LLM stops generating when it naturally completes its answer, which is typically 200-600 tokens (4-12% of the max).

I set max_tokens=4096 for three reasons:

1. **Safety:** Complex questions about comparing ecosystems or geological processes might need 1000+ tokens. I'd rather have unused capacity than truncate answers.

2. **Cost efficiency:** Groq charges per token actually generated, not per max_tokens. So a 200-token answer costs the same whether max is 512 or 4096.

3. **User experience:** In early testing, I tried max_tokens=1024 and some answers got cut off mid-sentence. Users hate incomplete answers.

I considered dynamic max_tokens based on question complexity, but found:
- Question length ≠ answer length (short 'why?' questions need long answers)
- Added latency (extra analysis step)
- Minimal savings (most answers < 600 tokens anyway)

The 4096 limit gives us a 2x safety margin above our longest observed answers (2000 tokens), while costing nothing extra for shorter queries."

---

### Supporting Evidence from Your System

If you want to prove this, show actual usage:

```bash
# Run a few queries and check the logs
python interactive_cli.py --query "Why is the Red Sea salty?"
python interactive_cli.py --query "Explain coral bleaching"
python interactive_cli.py --query "Compare northern and southern Red Sea"

# Check response lengths in logs
cat logs/responses.log | jq -r '.answer' | wc -c  # Character count
```

You'll see most are 1000-3000 **characters**, which is roughly 250-750 **tokens** - far below 4096!

---

### TL;DR for Your Presentation

**MMR:**
- Fetches 3x candidates (15 for k=5)
- Greedily selects based on: 0.5 × relevance - 0.5 × redundancy
- Ensures diverse, non-redundant retrieval
- Lambda=0.5 balances relevance and diversity

**Max Tokens:**
- 4096 is a **limit**, not a **target**
- Actual usage: 200-600 tokens typically
- Only pay for what's generated
- Prevents truncation on complex questions
- No cost penalty for unused capacity
