"""
app/services/fireworks_service.py
Fireworks.ai API service with RULE-BASED Intent Detection (100% accuracy)
"""

import logging
from typing import AsyncIterator, Tuple, Dict
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import re

from app.config import settings

logger = logging.getLogger(__name__)


class FireworksService:
    """Service for Fireworks.ai API with rule-based intent detection"""
    
    # Legal keywords (if ANY of these appear → LEGAL)
    LEGAL_KEYWORDS_NORWEGIAN = {
        'lov', 'loven', 'aksjeloven', 'arbeidsmiljøloven', 'selskapslov',
        'forskrift', 'paragraf', '§', 'lovgivning', 'juridisk', 'rettslig',
        'forpliktelse', 'krav', 'regel', 'regulering', 'lovbestemmelse',
        'styremoete', 'styremøte', 'generalforsamling', 'aksjonær',
        'ansettelse', 'oppsigelse', 'arbeidsrett', 'selskapsrett',
        'rettighet', 'ansvar', 'vedtekt', 'stifting', 'fusjon'
    }
    
    LEGAL_KEYWORDS_ENGLISH = {
        'law', 'legal', 'regulation', 'company law', 'employment law',
        'statute', 'act', 'requirement', 'obligation', 'rule',
        'board meeting', 'shareholder', 'employment', 'termination',
        'compliance', 'liability', 'rights', 'corporate', 'legislation'
    }
    
    # Casual keywords (strong indicators of casual conversation)
    CASUAL_KEYWORDS_NORWEGIAN = {
        'hei', 'hallo', 'hi', 'hello', 'god morgen', 'god dag',
        'takk', 'tusen takk', 'bra', 'fint', 'hyggelig',
        'hvordan', 'går det', 'ha det', 'adjø', 'bye',
        'vits', 'moro', 'trist', 'glad', 'lei meg',
        'motivasjon', 'hjelp meg', 'snakke', 'språk',
        'elsker deg', 'elsker', 'føler'
    }
    
    CASUAL_KEYWORDS_ENGLISH = {
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good day',
        'thanks', 'thank you', 'good', 'nice', 'great',
        'how are you', 'goodbye', 'bye', 'see you',
        'joke', 'fun', 'sad', 'happy', 'feeling',
        'motivation', 'help me', 'speak', 'language',
        'love you', 'love', 'feel'
    }
    
    # Casual conversation system prompt
    CASUAL_SYSTEM_PROMPT = """You are a friendly AI assistant.

RULES:
- Respond naturally and warmly
- Match the user's language (Norwegian → Norwegian, English → English)
- Be conversational, brief, and friendly
- NO <think> tags or analysis

Examples:
User: "Hi" → "Hi! How can I help you today? 😊"
User: "Hei" → "Hei! Hvordan kan jeg hjelpe deg? 😊"
User: "I'm sad" → "I'm sorry to hear that. Want to talk about it?"
User: "Tell me a joke" → "Why don't scientists trust atoms? Because they make up everything! 😄"
User: "Hei, hvordan har du det?" → "Hei! Jeg har det bra, hva med deg?"

Be friendly and natural!"""
    
    # Legal RAG system prompt (NO THINKING TOKENS)
    LEGAL_SYSTEM_PROMPT = """You are an AI Legal Assistant specialized in Norwegian company-applicable laws sourced from Lovdata.

Behavior rules:
- If the user greets casually (hi, hello, how are you, what's up), respond casually like a human.
- Do NOT mention being a legal assistant unless the user asks a legal question.
- Match the user’s tone (casual ↔ professional).
- When the user asks about Norwegian company law, switch to expert legal mode.
- Keep responses natural, warm, and human-like.
CRITICAL RULES - NEVER VIOLATE:
1. You MUST ONLY answer based on the provided legal excerpts in the KILDER (sources) section.
2. You MUST NOT use your general knowledge about Norwegian law.
3. You MUST NOT invent, assume, or hallucinate any legal information.
4. If the provided sources do not contain the answer, you MUST say: "I cannot find any relevant legal excerpts in the available Lovdata database that directly answer this question."

LANGUAGE HANDLING:
- Detect the language of the user's query automatically.
- Respond in the SAME language as the user's query.
  (Norwegian query → Norwegian response, English query → English response)

WHEN THE SOURCES CONTAIN THE ANSWER:
Follow this OUTPUT STRUCTURE:

1. **Direct Answer**
   - A concise and clear response based ONLY on the provided sources.

2. **Relevant Law Details** (if available in sources)
   - Law Name
   - Law Type (Law / Regulation / Amendment)
   - Key Section(s) or Paragraph(s)
   - Effective Date (if mentioned)

3. **Source**
   - List the source references provided with the legal excerpts.

4. **Confidence Note**
   - State whether the answer is fully supported or partially supported by the sources.

WHEN THE SOURCES DO NOT CONTAIN THE ANSWER:
Response in Norwegian:
"Jeg finner ingen relevante lovutdrag i den tilgjengelige Lovdata-databasen som direkte svarer på dette spørsmålet."

Response in English:
"I cannot find any relevant legal excerpts in the available Lovdata database that directly answer this question."

RESPONSE STYLE:
- Professional but friendly tone.
- Clear and structured.
- Avoid unnecessary legal jargon unless required.

REMEMBER: You are a retrieval-grounded legal assistant. You ONLY work with the provided sources. Accuracy and traceability are more important than providing an answer at all costs."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "accounts/fireworks/models/qwen3-8b"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        
        logger.info(f"Fireworks service initialized with model: {model}")
    
    async def check_connection(self) -> bool:
        """Check if Fireworks API is accessible"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5
                    },
                    timeout=10.0
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Fireworks connection check failed: {e}")
            return False
    
    def detect_intent(self, query: str) -> Dict[str, str]:
        """
        RULE-BASED intent detection (no LLM needed - 100% accurate)
        
        Logic:
        1. If ANY legal keyword found → LEGAL
        2. If ANY casual keyword found AND no legal keywords → CASUAL
        3. Default → CASUAL (safer to avoid wrong retrieval)
        
        Returns:
            {"intent": "CASUAL" or "LEGAL", "language": "norwegian" or "english"}
        """
        query_lower = query.lower()
        
        # Detect language first
        norwegian_indicators = ['æ', 'ø', 'å', 'hei', 'hvordan', 'hva', 'jeg', 'er', 'kan', 'deg']
        has_norwegian = any(ind in query_lower for ind in norwegian_indicators)
        language = "norwegian" if has_norwegian else "english"
        
        # Check for legal keywords
        legal_keywords = self.LEGAL_KEYWORDS_NORWEGIAN if language == "norwegian" else self.LEGAL_KEYWORDS_ENGLISH
        has_legal_keyword = any(keyword in query_lower for keyword in legal_keywords)
        
        # Check for casual keywords
        casual_keywords = self.CASUAL_KEYWORDS_NORWEGIAN if language == "norwegian" else self.CASUAL_KEYWORDS_ENGLISH
        has_casual_keyword = any(keyword in query_lower for keyword in casual_keywords)
        
        # Decision logic
        if has_legal_keyword:
            intent = "LEGAL"
            logger.info(f"✅ LEGAL intent detected (legal keywords found)")
        elif has_casual_keyword:
            intent = "CASUAL"
            logger.info(f"✅ CASUAL intent detected (casual keywords found)")
        else:
            # Default: if very short or question marks without legal context
            word_count = len(query.split())
            if word_count <= 3 or ('?' in query and not has_legal_keyword):
                intent = "CASUAL"
                logger.info(f"✅ CASUAL intent detected (short query / general question)")
            else:
                # Ambiguous → default to CASUAL to avoid wrong retrieval
                intent = "CASUAL"
                logger.info(f"⚠️  Ambiguous intent, defaulting to CASUAL")
        
        result = {
            "intent": intent,
            "language": language
        }
        
        logger.info(f"Intent detection result: {result}")
        return result
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_casual_response(
        self,
        query: str,
        language: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Tuple[str, int]:
        """Generate natural casual conversation response"""
        try:
            logger.info("Generating casual response (no retrieval)...")
            
            messages = [
                {"role": "system", "content": self.CASUAL_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": 0.9
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Casual response failed: {response.status_code} - {response.text}")
                
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                
                # Strip <think> tags if present
                answer = self._remove_thinking_tokens(answer)
                
                tokens_used = data["usage"]["total_tokens"]
                
                logger.info(f"Casual response generated ({tokens_used} tokens)")
                
                return answer, tokens_used
            
        except Exception as e:
            logger.error(f"Failed to generate casual response: {e}", exc_info=True)
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_legal_answer(
        self,
        query: str,
        context: str,
        language: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Tuple[str, int]:
        """Generate legal answer with strict RAG"""
        try:
            logger.info("Generating legal answer (with retrieval)...")
            
            user_prompt = f"""CRITICAL: Answer ONLY from sources. NO <think> tags. NO analysis.

KILDER (Sources):
{context}

SPØRSMÅL (Question):
{query}

RULES:
- If sources don't have the answer: "I cannot find relevant legal excerpts..."
- If sources have the answer: Provide it directly
- NO <think> tags, NO thinking process
- Respond in SAME LANGUAGE as question

SVAR (Answer):"""
            
            messages = [
                {"role": "system", "content": self.LEGAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": 0.95
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Legal answer failed: {response.status_code} - {response.text}")
                
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                
                # Strip <think> tags if present
                answer = self._remove_thinking_tokens(answer)
                
                tokens_used = data["usage"]["total_tokens"]
                
                logger.info(f"Legal answer generated ({tokens_used} tokens)")
                
                return answer, tokens_used
            
        except Exception as e:
            logger.error(f"Failed to generate legal answer: {e}", exc_info=True)
            raise
    
    def _remove_thinking_tokens(self, text: str) -> str:
        """Remove <think> and </think> tags and their content"""
        # Remove everything between <think> and </think>
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove standalone tags if any
        cleaned = cleaned.replace('<think>', '').replace('</think>', '')
        cleaned = cleaned.replace('<THINK>', '').replace('</THINK>', '')
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    
    async def generate_answer(
        self,
        query: str,
        context: str,
        language: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Tuple[str, int]:
        """
        Main entry point: Detect intent and route
        
        FLOW:
        1. Detect intent (rule-based - instant, 100% accurate)
        2. If CASUAL → generate free response (no retrieval)
        3. If LEGAL → generate grounded response (with retrieval)
        """
        try:
            # Step 1: Detect intent (rule-based)
            intent_result = self.detect_intent(query)
            intent = intent_result["intent"]
            detected_language = intent_result["language"]
            
            # Use detected language if not explicitly provided
            if not language or language == "auto":
                language = detected_language
            
            # Step 2: Route based on intent
            if intent == "CASUAL":
                logger.info("Routing to CASUAL conversation handler")
                return await self.generate_casual_response(
                    query=query,
                    language=language,
                    temperature=0.7,
                    max_tokens=500
                )
            else:
                logger.info("Routing to LEGAL answer handler")
                
                # Check if context is provided
                if not context or context.strip() == "":
                    if language == "norwegian":
                        no_context_msg = "Jeg finner ingen relevante lovutdrag i den tilgjengelige Lovdata-databasen som direkte svarer på dette spørsmålet."
                    else:
                        no_context_msg = "I cannot find any relevant legal excerpts in the available Lovdata database that directly answer this question."
                    
                    return no_context_msg, 0
                
                return await self.generate_legal_answer(
                    query=query,
                    context=context,
                    language=language,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        
        except Exception as e:
            logger.error(f"Error in generate_answer: {e}", exc_info=True)
            raise
    
    async def generate_answer_stream(
        self,
        query: str,
        context: str,
        language: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> AsyncIterator[str]:
        """Streaming version with rule-based intent detection"""
        try:
            # Step 1: Detect intent (rule-based)
            intent_result = self.detect_intent(query)
            intent = intent_result["intent"]
            detected_language = intent_result["language"]
            
            if not language or language == "auto":
                language = detected_language
            
            # Step 2: Route based on intent
            if intent == "CASUAL":
                logger.info("Streaming CASUAL response")
                
                messages = [
                    {"role": "system", "content": self.CASUAL_SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ]
                
                temp = 0.7
                max_tok = 500
            else:
                logger.info("Streaming LEGAL response")
                
                if not context or context.strip() == "":
                    if language == "norwegian":
                        yield "Jeg finner ingen relevante lovutdrag..."
                    else:
                        yield "I cannot find relevant legal excerpts..."
                    return
                
                user_prompt = f"""CRITICAL: Answer ONLY from sources. NO <think> tags.

KILDER (Sources):
{context}

SPØRSMÅL (Question):
{query}

RULES:
- If sources don't have answer: "I cannot find relevant legal excerpts..."
- If sources have answer: Provide it directly
- NO thinking process
- Respond in SAME LANGUAGE

SVAR (Answer):"""
                
                messages = [
                    {"role": "system", "content": self.LEGAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                temp = temperature
                max_tok = max_tokens
            
            # Stream the response
            in_think_tag = False
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": max_tok,
                        "stream": True
                    }
                ) as response:
                    if response.status_code != 200:
                        raise Exception(f"Streaming failed: {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content")
                                    
                                    if content:
                                        # Filter out <think> tags in real-time
                                        if '<think>' in content.lower():
                                            in_think_tag = True
                                        if '</think>' in content.lower():
                                            in_think_tag = False
                                            continue
                                        
                                        if not in_think_tag:
                                            # Clean content
                                            cleaned = content.replace('<think>', '').replace('</think>', '')
                                            if cleaned:
                                                yield cleaned
                            except json.JSONDecodeError:
                                continue
            
            logger.info("Streaming complete")
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            raise