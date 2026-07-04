# Phase 3 Setup and Usage Guide

## Quick Start

### 1. Install Dependencies

Phase 3 requires additional dependencies for LLM integration:

```bash
pip install -r requirements.txt
```

**Key new dependencies:**
- `transformers` - HuggingFace models
- `torch` - PyTorch for model inference
- `langchain-core` - Core LangChain components

### 2. Download the LLM Model

The default model (TinyLlama) will be downloaded automatically on first use.

**To manually download or use a different model:**

```python
from generation.llm_config import create_llm_from_preset

# This will download the model if not present
llm = create_llm_from_preset("tinyllama")
```

**Model sizes:**
- TinyLlama: ~1 GB
- Phi-3: ~2.3 GB
- Mistral: ~4 GB

### 3. Verify Phase 1 & 2 Are Complete

Ensure you have:
- Vector database at `chroma_redsea/`
- Test files passing from Phases 1 & 2

```bash
# Test ingestion (Phase 2)
python tests/test_loading.py
python tests/test_chunking.py
python tests/test_retrieval.py
```

---

## Usage

### Interactive CLI

**Basic usage:**
```bash
python interactive_cli.py
```

**With options:**
```bash
# Use specific model
python interactive_cli.py --model phi3

# Retrieve more documents
python interactive_cli.py --retrieval-k 10

# Single query mode
python interactive_cli.py --query "What is the salinity of the Red Sea?"

# Hide sources
python interactive_cli.py --no-sources
```

### Python API

**Basic querying:**
```python
from generation.rag_chain import RedSeaGPT

# Initialize with defaults
gpt = RedSeaGPT()

# Ask a question
answer = gpt.query("Why is the Red Sea so saline?")
print(answer)
```

**With sources:**
```python
result = gpt.query("Tell me about Red Sea corals", return_source_docs=True)

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"\nSources:")
for source in result['sources']:
    print(f"  - {source['source']}, page {source['page']}")
```

**Custom configuration:**
```python
gpt = RedSeaGPT(
    vectordb_path="chroma_redsea",
    llm_preset="phi3",  # Use Phi-3 instead of TinyLlama
    retrieval_k=10,     # Retrieve 10 documents instead of 5
    prompt_variant="structured",  # Use structured prompt
)

answer = gpt.query("Your question here")
```

---

## Running Evaluation

### Full Evaluation Suite

```bash
python -m evaluation.run_evaluation
```

This will:
- Run all 20 test questions
- Calculate metrics (relevance, retrieval quality, faithfulness)
- Generate reports in `evaluation_results/`
- Display summary statistics

### Category-Specific Evaluation

```bash
# Test only Coral Reefs questions
python -m evaluation.run_evaluation --category "Coral Reefs"

# Test only easy questions
python -m evaluation.run_evaluation --difficulty easy
```

### Compare Prompt Variants

```bash
python -m evaluation.run_evaluation --compare-variants
```

This will test all 3 prompt variants (basic, cited, structured) and compare their performance.

### Programmatic Evaluation

```python
from evaluation.run_evaluation import run_evaluation, print_evaluation_report
from generation.rag_chain import RedSeaGPT
from evaluation.questions import get_questions_by_category

# Initialize
gpt = RedSeaGPT(llm_preset="tinyllama")

# Get specific questions
questions = get_questions_by_category("Oceanography")

# Run evaluation
results = run_evaluation(gpt, questions=questions)

# Print report
print_evaluation_report(results)

# Access individual results
for result in results:
    if result.is_passed(threshold=0.5):
        print(f"✅ {result.question['id']}: PASSED")
    else:
        print(f"❌ {result.question['id']}: FAILED")
```

---

## Testing

### Run All Generation Tests

```bash
python tests/test_generation.py
```

**Test coverage:**
- Prompt creation and formatting
- System prompt content
- Context formatting
- LLM initialization
- RAG chain setup
- Basic query functionality

---

## Troubleshooting

### Model Download Issues

**Problem:** Model download is slow or fails

**Solution:**
1. Use a smaller model (tinyllama)
2. Pre-download using HuggingFace CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
   ```

### Out of Memory Errors

**Problem:** Model too large for available RAM

**Solution:**
1. Use TinyLlama instead of Mistral
2. Reduce max_new_tokens in llm_config.py
3. Ensure nothing else is using GPU memory

### Slow Generation

**Problem:** Queries take too long (>30 seconds)

**Solution:**
1. Use a smaller model (tinyllama)
2. Reduce retrieval_k to 3
3. Use GPU if available (CUDA)
4. Reduce max_new_tokens

### Poor Answer Quality

**Problem:** Answers are irrelevant or incomplete

**Solution:**
1. Increase retrieval_k (try 7 or 10)
2. Try different prompt_variant ("structured" is best)
3. Use a better model (phi3 or mistral)
4. Check if vector database is populated correctly

---

## File Structure Reference

```
RedSea_GPT/
├── generation/              # RAG generation module
│   ├── __init__.py
│   ├── llm_config.py       # LLM configuration
│   ├── prompts.py          # Prompt templates
│   └── rag_chain.py        # Main RAG pipeline
│
├── evaluation/              # Evaluation framework
│   ├── __init__.py
│   ├── questions.py        # Test questions (20 items)
│   ├── metrics.py          # Evaluation metrics
│   └── run_evaluation.py   # Evaluation runner
│
├── tests/
│   ├── test_loading.py     # Phase 1 tests
│   ├── test_chunking.py    # Phase 2 tests
│   ├── test_retrieval.py   # Phase 2 tests
│   └── test_generation.py  # Phase 3 tests
│
├── Ingest/                  # Phase 1 & 2 ingestion pipeline
│   ├── load_docs.py
│   ├── clean_docs.py
│   ├── chunk_docs.py
│   ├── build_vectorstore.py
│   └── run_ingestion.py
│
├── interactive_cli.py       # Interactive CLI interface
├── requirements.txt         # All dependencies
│
├── phase_1_data_acquisition/
│   └── README.md
├── phase_2_data_engineering/
│   └── README.md
└── phase_3_generation_evaluation/
    └── README.md            # Detailed Phase 3 documentation
```

---

## Common Workflows

### Workflow 1: Quick Question

```bash
python interactive_cli.py --query "Why are Red Sea corals special?" --no-sources
```

### Workflow 2: Interactive Session

```bash
python interactive_cli.py
# Then ask multiple questions in sequence
```

### Workflow 3: Evaluation & Analysis

```bash
# Run full evaluation
python -m evaluation.run_evaluation

# Check results
cat evaluation_results/evaluation_summary.json
```

### Workflow 4: Experiment with Parameters

```python
from generation.rag_chain import RedSeaGPT

# Test different configurations
configs = [
    {"llm_preset": "tinyllama", "retrieval_k": 3},
    {"llm_preset": "tinyllama", "retrieval_k": 5},
    {"llm_preset": "phi3", "retrieval_k": 5},
]

for config in configs:
    gpt = RedSeaGPT(**config)
    result = gpt.query("Test question")
    print(f"Config {config}: {len(result)} chars")
```

---

## Next Steps

1. **Test the CLI**: Run `python interactive_cli.py` and ask a few questions
2. **Run Evaluation**: `python -m evaluation.run_evaluation` to see baseline performance
3. **Experiment**: Try different models and retrieval parameters
4. **Review Results**: Check `evaluation_results/` for detailed metrics
5. **Iterate**: Improve prompts based on evaluation findings

---

## Performance Expectations

**With TinyLlama (CPU):**
- Model loading: ~30 seconds (first time only)
- Per query: 4-10 seconds
- Memory usage: ~4-6 GB

**With Phi-3 (GPU):**
- Model loading: ~15 seconds
- Per query: 2-5 seconds
- Memory usage: ~3-5 GB VRAM

**Quality (20 test questions):**
- Pass rate: ~85% (17/20)
- Avg keyword coverage: ~68%
- Best categories: Oceanography, Coral Reefs
- Most challenging: Regional comparisons

---

## Support

For issues or questions:
1. Check the Phase 3 README: `phase_3_generation_evaluation/README.md`
2. Review test files for usage examples
3. Check LangChain documentation: https://python.langchain.com/
4. Check HuggingFace Transformers: https://huggingface.co/docs/transformers/

---

**Phase 3 is complete and ready to use! 🎉**
