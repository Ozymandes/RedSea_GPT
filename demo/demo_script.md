# RedSea GPT - Live Demo Script

## Preparation (5 minutes before)

1. **Start the CLI:**
   ```bash
   cd /path/to/RedSea_GPT
   python interactive_cli.py
   ```

2. **Verify initialization:**
   - Should see "✅ Ready!"
   - Note the model info (Llama 70B via Groq)

3. **Clear screen if needed**

---

## Demo Flow (5-7 minutes)

### **Part 1: Introduction (1 minute)**
"Today I'm presenting RedSea GPT, an AI-powered naturalist for the Egyptian Red Sea. It's built using RAG - Retrieval Augmented Generation - which combines a powerful language model with a curated knowledge base of 15 scientific textbooks and papers."

**Key points to mention:**
- Uses Llama 3.3 70B via Groq API
- Ultra-fast responses (1-3 seconds)
- Specialized only on Red Sea topics
- Has guardrails and logging

---

### **Part 2: First Question (1 minute)**

**Question:** "Why is the Red Sea so saline?"

**What to show:**
- System responds quickly
- Note the confidence score
- Expand sources to show scientific papers

**Talking points:**
- "You can see it retrieved relevant documents about salinity"
- "Confidence is 85% - very high"
- "It cites specific papers with page numbers"

---

### **Part 3: Second Question - More Complex (1 minute)**

**Question:** "How are Red Sea corals adapted to heat?"

**What to show:**
- Retrieves about coral adaptations
- Shows detailed biological mechanisms
- Multiple sources used

**Talking points:**
- "This demonstrates retrieval from specialized coral research"
- "The system combines information from multiple papers"
- "Latency is still under 2 seconds"

---

### **Part 4: Refusal Behavior (1 minute)**

**Question:** "Who won the World Cup in 2022?"

**Expected result:** Polite refusal explaining it's off-topic

**Talking points:**
- "The system knows when it doesn't have relevant information"
- "It won't hallucinate answers outside its knowledge base"
- "This is a key safety feature"

---

### **Part 5: Technical Details (1-2 minutes)**

**Show architecture:**
```
User Question
    ↓
Vector Retrieval (MMR)
    ↓
LLM Generation (Llama 70B)
    ↓
Refusal & Hallucination Check
    ↓
Answer + Sources + Metadata
```

**Key technical points:**
- **MMR Retrieval:** Diverse, non-redundant results
- **Structured Citations:** [1], [2], [3] format
- **Hallucination Detection:** Checks grounding
- **Guardrails:** Rate limiting, content moderation
- **Logging:** All queries logged for analysis

---

### **Part 6: Performance & Cost (30 seconds)**

**Metrics to mention:**
- Average latency: 1-3 seconds
- Cost per query: ~$0.0001 (Groq API)
- Knowledge base: 15 textbooks/papers
- Refusal accuracy: ~95%

---

## Backup Plan

**If the API is slow:**
- Explain Groq is in beta
- Move to showing architecture instead
- Switch to pre-recorded demo

**If the system refuses:**
- Explain the threshold (20% confidence)
- Ask a different question
- Use backup question

---

## Closing (30 seconds)

"RedSea GPT demonstrates how RAG systems can provide accurate, domain-specific AI assistants. The combination of fast LLM inference, intelligent retrieval, and proper guardrails creates a reliable system for scientific knowledge access."

**Thank the audience**
**Open for questions**
