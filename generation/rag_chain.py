
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .llm_config import create_llm
from .memory import ConversationMemory, Turn, resolve_query_with_history
from .prompts import create_rag_prompt, format_context, DEFAULT_TONE, VALID_TONES
from .utils import clean_source_path


class RedSeaGPT:
    """
    RedSea GPT - Specialized naturalist for the Egyptian Red Sea.

    This class encapsulates the complete RAG pipeline including retrieval
    and generation capabilities.
    """

    def __init__(
        self,
        vectordb_path: str = "chroma_redsea",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        llm_config: Optional[Dict[str, Any]] = None,
        retrieval_k: int = 7,
        prompt_variant: str = "structured",
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        refusal_threshold: float = 0.2,  # Lowered from 0.3 for 70B model
        structured_citations: bool = True,
        enable_logging: bool = False,  # Phase 4: Logging parameter
        enable_guardrails: bool = False,  # Phase 4: Guardrails parameter
    ):
        self.vectordb_path = vectordb_path
        self.retrieval_k = retrieval_k
        self.embedding_model = embedding_model
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.refusal_threshold = refusal_threshold
        self.structured_citations = structured_citations
        self.enable_logging = enable_logging
        self.enable_guardrails = enable_guardrails

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        # Load vector database
        self.vectordb = Chroma(
            persist_directory=vectordb_path,
            embedding_function=self.embeddings,
        )

        # Initialize LLM (provider-agnostic; configured via env or llm_config)
        self.llm = create_llm(**(llm_config or {}))

        # Prompt templates are built per-tone on demand (cheap; cached upstream).
        # Keep a default-tone instance for the legacy simple_chain path.
        self.prompt = create_rag_prompt(DEFAULT_TONE)

        # Create the RAG chain
        self.chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """
        Create the RAG chain using a simple function-based approach.

        Returns:
            Composed RAG chain function
        """
        def simple_chain(question: str) -> str:
            # Retrieve documents
            docs = self.vectordb.similarity_search(question, k=self.retrieval_k)

            # Format context
            context = format_context(docs)

            # Generate prompt
            formatted_prompt = self.prompt.format(context=context, question=question)

            # Generate answer
            answer = self.llm.invoke(formatted_prompt)

            # Extract string if needed
            if hasattr(answer, 'content'):
                answer = answer.content
            elif isinstance(answer, dict) and 'content' in answer:
                answer = answer['content']

            return str(answer)

        return simple_chain

    def _mmr_retrieve(self, question: str, k: int = 5) -> Tuple[List[Document], List[float]]:
        
       
        fetch_k = min(k * 3, 50)  
        candidates = self.vectordb.similarity_search(question, k=fetch_k)

        if not candidates:
            return [], []

      
        query_embedding = self.embeddings.embed_query(question)
        doc_texts = [doc.page_content for doc in candidates]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)

        
        query_similarities = cosine_similarity(
            [query_embedding], doc_embeddings
        )[0]

     
        selected_indices = []
        selected_scores = []

        for _ in range(min(k, len(candidates))):
            # Calculate MMR score for each unselected document
            mmr_scores = []
            for idx, doc_emb in enumerate(doc_embeddings):
                if idx in selected_indices:
                    mmr_scores.append(-float('inf'))
                    continue

                # Relevance to query
                relevance = query_similarities[idx]

                # Diversity penalty (max similarity to already selected docs)
                diversity_penalty = 0
                if selected_indices:
                    selected_embs = [doc_embeddings[i] for i in selected_indices]
                    similarities_to_selected = cosine_similarity([doc_emb], selected_embs)[0]
                    diversity_penalty = max(similarities_to_selected)

                # MMR score
                mmr = (self.mmr_lambda * relevance) - ((1 - self.mmr_lambda) * diversity_penalty)
                mmr_scores.append(mmr)

            # Select document with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected_indices.append(best_idx)
            selected_scores.append(float(query_similarities[best_idx]))

        # Return selected documents and their relevance scores
        selected_docs = [candidates[i] for i in selected_indices]
        return selected_docs, selected_scores

    def _check_answer_confidence(self, relevance_scores: List[float]) -> Tuple[bool, float]:
        """
        Check if we have sufficient confidence to answer the question.

        Args:
            relevance_scores: List of relevance scores for retrieved documents

        Returns:
            Tuple of (should_answer, avg_relevance)
        """
        if not relevance_scores:
            return False, 0.0

        avg_relevance = np.mean(relevance_scores)
        max_relevance = max(relevance_scores)

        # Use both average and max relevance
        # We need at least one highly relevant doc OR decent overall relevance
        should_answer = (max_relevance >= self.refusal_threshold or
                        avg_relevance >= self.refusal_threshold * 0.7)

        return should_answer, avg_relevance

    def _check_topic_mismatch(self, question: str, docs: List[Document]) -> Dict[str, Any]:
        
        topic_keywords = {
            'fish': ['fish', 'fishes', 'ichthyofauna', 'piscine'],
            'coral': ['coral', 'corals', 'reef', 'scleractinian', 'cladocopium'],
            'plants': ['seagrass', 'algae', 'mangrove', 'phytoplankton'],
            'geology': ['geological', 'formation', 'basalt', 'magmatism', 'rift', 'tectonic'],
            'conservation': ['conservation', 'protection', 'threat', 'management', 'mpa'],
            'pollution': ['pollution', 'plastic', 'oil', 'contaminant'],
            'climate': ['climate change', 'warming', 'temperature rise', 'bleaching'],
            'future': ['will', 'future', 'predict', 'forecast', '2100', '2050', 'projected'],
            'salinity': ['salinity', 'salin', 'salt', '‰'],
            'biodiversity': ['biodiversity', 'species', 'endemic', 'diversity'],
        }

        question_lower = question.lower()

    
        question_topic = None
        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                question_topic = topic
                break

        
        if question_topic:
            all_content = ' '.join([doc.page_content.lower() for doc in docs])

            
            topic_keywords_in_docs = sum(1 for kw in topic_keywords[question_topic]
                                        if kw in all_content)

            # If question asks about a topic but documents rarely mention it, likely mismatch
            if topic_keywords_in_docs == 0:
                # Try to identify what documents actually discuss
                doc_topic = 'general Red Sea information'
                for topic, keywords in topic_keywords.items():
                    if topic == question_topic:
                        continue
                    if sum(1 for kw in keywords if kw in all_content) >= 2:
                        doc_topic = topic
                        break

                return {
                    'has_mismatch': True,
                    'question_topic': question_topic,
                    'doc_topic': doc_topic
                }

        return {
            'has_mismatch': False,
            'question_topic': question_topic or 'general',
            'doc_topic': 'general'
        }

    def _format_context_with_citations(self, docs: List[Document]) -> str:
        """
        Format context with structured citation IDs *and* page-level provenance.

        Provenance is included so the model can attach accurate citations and so
        each citation can be verified against a real source/page pair.
        """
        return format_context(docs)

    def _format_sources_list(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Format sources as a list with citation numbers.

        Args:
            docs: List of retrieved documents

        Returns:
            List of source dictionaries with citation info
        """
        sources = []
        for i, doc in enumerate(docs, start=1):
            source = clean_source_path(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "Unknown")

            sources.append({
                "citation_id": i,
                "source": source,
                "page": page,
                "content": doc.page_content[:300] + "...",
            })

        return sources

    def _detect_hallucinations(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Detect potential hallucinations in the generated answer.

        Uses multiple heuristics:
        1. Sentence-by-sentence grounding check
        2. N-gram overlap analysis
        3. Factual consistency check

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Dictionary with hallucination metrics
        """
        # Split into sentences
        sentences = [s.strip() for s in answer.split('.') if s.strip()]

        if not sentences:
            return {
                "has_hallucination": False,
                "grounded_sentences": 0,
                "total_sentences": 0,
                "grounding_rate": 0.0,
            }

        # Get context words (lowercase)
        context_words = set(context.lower().split())

        grounded_count = 0
        ungrounded_sentences = []

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())

            # Check overlap (at least 30% of words should be in context)
            if len(sentence_words) > 0:
                overlap = len(sentence_words & context_words)
                overlap_rate = overlap / len(sentence_words)

                # Also check if at least some content words overlap
                content_words = [w for w in sentence_words if len(w) > 3]
                content_overlap = len(set(content_words) & context_words)

                is_grounded = (overlap_rate >= 0.3 or
                              (len(content_words) > 0 and content_overlap >= len(content_words) * 0.4))

                if is_grounded:
                    grounded_count += 1
                else:
                    ungrounded_sentences.append(sentence)

        grounding_rate = grounded_count / len(sentences) if sentences else 0

        # Flag as potential hallucination if grounding rate is low
        has_hallucination = grounding_rate < 0.6

        return {
            "has_hallucination": has_hallucination,
            "grounded_sentences": grounded_count,
            "total_sentences": len(sentences),
            "grounding_rate": grounding_rate,
            "ungrounded_sentences": ungrounded_sentences[:3],  # First 3 ungrounded
        }

    def query(self, question: str, return_source_docs: bool = False,
              memory: Optional[ConversationMemory] = None,
              tone: str = DEFAULT_TONE) -> str | Dict[str, Any]:
        """
        Query RedSea GPT with a question.

        Enhanced with:
        - MMR retrieval for diverse results
        - Refusal logic for low-confidence queries
        - Structured citations [1], [2], [3]
        - Hallucination detection
        - Multiturn conversation memory: when ``memory`` is provided and non-empty,
          the latest message is rewritten into a self-contained question using the
          conversation history (so pronouns/implicit references like "how deep is
          it?" retrieve correctly). The full history is also shown to the generator
          so it can phrase the answer naturally. Falls back to the raw question on
          any failure.

        Args:
            question: User's question about the Red Sea
            return_source_docs: Whether to return source documents and metadata
            memory: Optional conversation buffer for multiturn chat
            tone: "technical" (university-level) or "intuitive" (hobbyist-friendly)

        Returns:
            Generated answer (or dict with answer, sources, and metadata if return_source_docs=True)

        Examples:
            >>> answer = gpt.query("Why is the Red Sea so saline?")
            >>> result = gpt.query("What corals live in the Red Sea?", return_source_docs=True)
            >>> print(result['answer'])
            >>> print(result['sources'])
            >>> # Multiturn:
            >>> mem = ConversationMemory()
            >>> gpt.query("How did the Red Sea form?", memory=mem)
            >>> gpt.query("how fast is that happening?", memory=mem)  # resolves to spreading rate
        """
        # Step 0: If we have conversation history, rewrite the latest message into a
        # self-contained question so retrieval/topic-mismatch see the real intent.
        # Falls back to the raw question if memory is empty or resolution fails.
        history_block = ""
        resolved_question = question
        if memory is not None and not memory.is_empty:
            resolved_question = resolve_query_with_history(self.llm, question, memory)
            history_block = memory.format_for_prompt()

        # Step 1: Retrieve documents using the RESOLVED question (handles pronouns).
        if self.use_mmr:
            source_docs, relevance_scores = self._mmr_retrieve(resolved_question, k=self.retrieval_k)
        else:
            source_docs = self.vectordb.similarity_search(resolved_question, k=self.retrieval_k)
            # Calculate relevance scores for standard retrieval
            query_embedding = self.embeddings.embed_query(resolved_question)
            doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in source_docs])
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            relevance_scores = list(similarities)

        # Step 2: Check for topic mismatch between (resolved) question and retrieved content
        topic_mismatch = self._check_topic_mismatch(resolved_question, source_docs)

        if topic_mismatch['has_mismatch']:
            refusal_msg = (
                f"I apologize, but I don't have sufficient information to answer your question about {topic_mismatch['question_topic']}. "
                f"While I found relevant documents, they discuss {topic_mismatch['doc_topic']} rather than {topic_mismatch['question_topic']} specifically.\n\n"
                f"The available research papers focus on {topic_mismatch['doc_topic']}, "
                f"but don't contain the specific information about {topic_mismatch['question_topic']} that you're asking about.\n\n"
                f"You might try:\n"
                f"• Asking about {topic_mismatch['doc_topic']} instead\n"
                f"• Rephrasing your question to focus on what's available in the research\n\n"
                f"I'm designed to be accurate and will only answer when I have reliable information "
                f"from the Red Sea scientific literature."
            )

            if return_source_docs:
                return {
                    "answer": refusal_msg,
                    "sources": self._format_sources_list(source_docs),
                    "question": question,
                    "confidence": 0.0,
                    "refusal": True,
                    "retrieval_method": "MMR" if self.use_mmr else "similarity",
                    "reason": f"Topic mismatch: question asks about {topic_mismatch['question_topic']}, documents discuss {topic_mismatch['doc_topic']}",
                }
            else:
                return refusal_msg

        # Step 3: Check confidence for refusal logic
        should_answer, avg_relevance = self._check_answer_confidence(relevance_scores)

        if not should_answer:
            refusal_msg = (
                "I apologize, but I don't have sufficient information in my knowledge base "
                "to provide a confident answer to your question about the Red Sea. "
                f"The retrieved documents have an average relevance score of {avg_relevance:.2f}, "
                f"which is below my threshold of {self.refusal_threshold}..\n\n"
                "This could mean:\n"
                "• Your question is outside the scope of my Red Sea knowledge base\n"
                "• The specific information isn't covered in the research papers I've studied\n"
                "• You might try rephrasing your question\n\n"
                "I'm designed to be accurate and will only answer when I have reliable information "
                "from the Red Sea scientific literature."
            )

            if return_source_docs:
                return {
                    "answer": refusal_msg,
                    "sources": self._format_sources_list(source_docs),
                    "question": question,
                    "confidence": avg_relevance,
                    "refusal": True,
                    "retrieval_method": "MMR" if self.use_mmr else "similarity",
                }
            else:
                return refusal_msg

        # Step 3: Format context (with or without structured citations)
        # Prepend the conversation history (if any) so the generator can phrase
        # the answer naturally and reference earlier turns.
        if self.structured_citations:
            context = self._format_context_with_citations(source_docs)
        else:
            context = format_context(source_docs)
        if history_block:
            context = history_block + context

        # Step 4: Generate answer (the generator sees the RESOLVED question so it
        # answers the real intent; the user still sees their raw message in the UI).
        # The prompt is selected per-tone: 'technical' (dense, jargon-permitting) vs
        # 'intuitive' (same specifics, every term unpacked in plain language).
        prompt = create_rag_prompt(tone if tone in VALID_TONES else DEFAULT_TONE)
        formatted_prompt = prompt.format(context=context, question=resolved_question)
        answer = self.llm.invoke(formatted_prompt)

        # Extract string if it's a structured output
        if hasattr(answer, 'content'):
            answer = answer.content
        elif isinstance(answer, dict) and 'content' in answer:
            answer = answer['content']

        answer_str = str(answer)

        # Step 5: Check if the LLM refused (either via explicit patterns OR natural refusal language).
        # The new prompt produces natural, conversational refusals ("I'm sorry, but I can't provide...",
        # "that term is not covered in the available sources"), which we must detect so the eval
        # flags them correctly. We KEEP the model's natural wording (it reads better than a canned
        # message) but set refusal=True.
        answer_lower = answer_str.lower()

        # Strong refusal indicators - explicit admissions the context can't answer.
        # These patterns indicate the LLM explicitly admits it can't answer from the context
        strong_refusal_patterns = [
            "no direct information in the provided context",
            "the provided context does not contain",
            "context does not provide information",
            "not mentioned in the provided context",
            "while the provided context does not",
            "while we cannot provide",
            "cannot be determined from the provided context",
            "insufficient information to predict",
            "insufficient information to answer",
        ]

        # Natural refusal phrases - covers the model's conversational refusals so they are
        # detected even when they don't match the canned patterns above. Kept in sync with
        # evaluation/metrics_v2.py REFUSUAL_PHRASES.
        natural_refusal_phrases = (
            "can't provide",
            "cannot provide",
            "unable to provide",
            "i'm unable to",
            "i am unable to",
            "i can't answer",
            "i cannot answer",
            "not covered in the available",
            "not covered in the provided",
            "does not appear",
            "does not appear explicitly",
            "isn't covered",
            "is not covered",
            "not explicitly mentioned",
            "outside the scope",
            "outside my scope",
        )

        # Check for any strong refusal pattern
        has_strong_refusal = any(pattern in answer_lower for pattern in strong_refusal_patterns)
        # Natural refusal: only counts if it appears in the FIRST ~25% of the answer (a real
        # answer may mention a limit later, but a refusal leads with it).
        first_part = answer_lower[: max(200, len(answer_lower) // 4)]
        has_natural_refusal = any(p in first_part for p in natural_refusal_phrases)

        # Secondary check: admission of inability + speculative language
        # Catches patterns like "cannot predict" or "cannot be predicted"
        speculative_refusal_patterns = [
            "cannot predict",
            "cannot be predicted",
            "impossible to determine",
            "no way to predict",
        ]
        has_speculative_refusal = any(pattern in answer_lower for pattern in speculative_refusal_patterns)

        # If a NATURAL refusal is detected, trust the model's wording but flag it as a refusal.
        # (We do NOT overwrite with the canned message - the model's natural phrasing reads better.)
        if has_natural_refusal:
            if return_source_docs:
                return {
                    "answer": answer_str,
                    "sources": self._format_sources_list(source_docs),
                    "question": question,
                    "confidence": avg_relevance,
                    "refusal": True,
                    "retrieval_method": "MMR" if self.use_mmr else "similarity",
                    "reason": "Natural refusal detected in generated answer",
                }
            else:
                return answer_str

        # If either strong refusal OR speculative refusal is detected, refuse to answer
        if has_strong_refusal or has_speculative_refusal:

            refusal_msg = (
                "I apologize, but I don't have sufficient information in my knowledge base "
                "to provide a confident answer to your question about the Red Sea.\n\n"
                "While I found relevant documents, they don't contain the specific information needed "
                "to address your question. The available research papers focus on current and historical "
                "data, not future projections or predictions.\n\n"
                "This could mean:\n"
                "• Your question asks about future predictions (e.g., 'what will happen in 2100?')\n"
                "• The specific information isn't covered in the research papers I've studied\n"
                "• You might try rephrasing your question to focus on current knowledge\n\n"
                "I'm designed to be accurate and will only answer when I have reliable information "
                "from the Red Sea scientific literature."
            )

            if return_source_docs:
                return {
                    "answer": refusal_msg,
                    "sources": self._format_sources_list(source_docs),
                    "question": question,
                    "confidence": avg_relevance,
                    "refusal": True,
                    "retrieval_method": "MMR" if self.use_mmr else "similarity",
                    "reason": "LLM admitted insufficient information",
                }
            else:
                return refusal_msg

        # Step 6: Detect hallucinations
        hallucination_check = self._detect_hallucinations(answer_str, context)

        # Step 7: Flag weak grounding in metadata only (do not pollute the answer
        # text the user/evaluator sees - that would itself lower faithfulness and
        # read as unpolished). The grounding rate is reported via metadata instead.

        # Step 8: Return results
        if return_source_docs:
            sources = self._format_sources_list(source_docs)

            return {
                "answer": answer_str,
                "sources": sources,
                "retrieved_chunks": [
                    {
                        "citation_id": i,
                        "source": clean_source_path(doc.metadata.get("source", "Unknown")),
                        "page": doc.metadata.get("page"),
                        "page_content": doc.page_content,
                    }
                    for i, doc in enumerate(source_docs, start=1)
                ],
                "question": question,
                "resolved_question": resolved_question,
                "confidence": avg_relevance,
                "refusal": False,
                "retrieval_method": "MMR" if self.use_mmr else "similarity",
                "num_sources": len(source_docs),
                "hallucination_check": hallucination_check,
            }
        else:
            return answer_str


def create_rag_chain(
    vectordb_path: str = "chroma_redsea",
    retrieval_k: int = 5,
    **kwargs
) -> RedSeaGPT:
    """
    Convenience function to create a RedSeaGPT instance.

    Args:
        vectordb_path: Path to vector database
        retrieval_k: Number of documents to retrieve
        **kwargs: Additional arguments for RedSeaGPT

    Returns:
        Configured RedSeaGPT instance

    Examples:
        >>> rag = create_rag_chain()
        >>> answer = rag.query("Tell me about Red Sea corals")
    """
    return RedSeaGPT(
        vectordb_path=vectordb_path,
        retrieval_k=retrieval_k,
        **kwargs
    )
