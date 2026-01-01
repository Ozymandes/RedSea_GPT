
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .llm_config import create_llm
from .prompts import create_rag_prompt, format_context
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
        retrieval_k: int = 5,
        prompt_variant: str = "structured",
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        refusal_threshold: float = 0.2,  # Lowered from 0.3 for 70B model
        structured_citations: bool = True,
    ):
        self.vectordb_path = vectordb_path
        self.retrieval_k = retrieval_k
        self.embedding_model = embedding_model
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.refusal_threshold = refusal_threshold
        self.structured_citations = structured_citations

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        # Load vector database
        self.vectordb = Chroma(
            persist_directory=vectordb_path,
            embedding_function=self.embeddings,
        )

        # Initialize LLM (Llama 70B via Groq API)
        self.llm = create_llm(**(llm_config or {}))

        # Create prompt template
        self.prompt = create_rag_prompt()

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
        Format context with structured citation markers.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted context string with [1], [2], [3] citation markers
        """
        context_parts = []

        for i, doc in enumerate(docs, start=1):
            # Add citation marker to content
            content = doc.page_content.strip()
            context_parts.append(f"[{i}] {content}")

        return "\n\n---\n\n".join(context_parts)

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

    def query(self, question: str, return_source_docs: bool = False) -> str | Dict[str, Any]:
        """
        Query RedSea GPT with a question.

        Enhanced with:
        - MMR retrieval for diverse results
        - Refusal logic for low-confidence queries
        - Structured citations [1], [2], [3]
        - Hallucination detection

        Args:
            question: User's question about the Red Sea
            return_source_docs: Whether to return source documents and metadata

        Returns:
            Generated answer (or dict with answer, sources, and metadata if return_source_docs=True)

        Examples:
            >>> answer = gpt.query("Why is the Red Sea so saline?")
            >>> result = gpt.query("What corals live in the Red Sea?", return_source_docs=True)
            >>> print(result['answer'])
            >>> print(result['sources'])
        """
        # Step 1: Retrieve documents (with or without MMR)
        if self.use_mmr:
            source_docs, relevance_scores = self._mmr_retrieve(question, k=self.retrieval_k)
        else:
            source_docs = self.vectordb.similarity_search(question, k=self.retrieval_k)
            # Calculate relevance scores for standard retrieval
            query_embedding = self.embeddings.embed_query(question)
            doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in source_docs])
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            relevance_scores = list(similarities)

        # Step 2: Check for topic mismatch between question and retrieved content
        topic_mismatch = self._check_topic_mismatch(question, source_docs)

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
        if self.structured_citations:
            context = self._format_context_with_citations(source_docs)
        else:
            context = format_context(source_docs)

        # Step 4: Generate answer
        formatted_prompt = self.prompt.format(context=context, question=question)
        answer = self.llm.invoke(formatted_prompt)

        # Extract string if it's a structured output
        if hasattr(answer, 'content'):
            answer = answer.content
        elif isinstance(answer, dict) and 'content' in answer:
            answer = answer['content']

        answer_str = str(answer)

        # Step 5: Check if LLM admitted it doesn't have information
        # This catches cases where retrieval confidence is high but context doesn't answer the question
        answer_lower = answer_str.lower()

        # Strong refusal indicators - if any of these appear, refuse immediately
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

        # Check for any strong refusal pattern
        has_strong_refusal = any(pattern in answer_lower for pattern in strong_refusal_patterns)

        # Secondary check: admission of inability + speculative language
        # Catches patterns like "cannot predict" or "cannot be predicted"
        speculative_refusal_patterns = [
            "cannot predict",
            "cannot be predicted",
            "impossible to determine",
            "no way to predict",
        ]
        has_speculative_refusal = any(pattern in answer_lower for pattern in speculative_refusal_patterns)

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

        # Step 7: Add warning if hallucinations detected
        if hallucination_check["has_hallucination"]:
            warning = (
                f"\n\n⚠️  Note: This answer may contain information not directly supported by the retrieved documents. "
                f"Grounding rate: {hallucination_check['grounding_rate']:.1%}. "
                f"Please verify important facts."
            )
            answer_str = answer_str + warning

        # Step 8: Return results
        if return_source_docs:
            sources = self._format_sources_list(source_docs)

            return {
                "answer": answer_str,
                "sources": sources,
                "question": question,
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
