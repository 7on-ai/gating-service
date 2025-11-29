#!/usr/bin/env python3
"""
Ethical Growth Gating Service - COMPLETE VERSION 6.0
✅ Personal Ollama Adapter Integration
✅ Multilingual Support (Thai/English)
✅ Full LoRA Training Integration
✅ Multi-Tenant SaaS Ready
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import httpx
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ethical Growth Gating Service")

# ✅ IMPORTANT: Use internal URL for cluster communication
# External URL is only for cross-project or public access
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"

logger.info(f"🔗 Ollama URL: {OLLAMA_URL}")

# ============================================================
# ✅ NEW: Personal Ollama Adapter Classification
# ============================================================

async def classify_with_ollama_adapter(text: str, user_id: str, lang: str) -> Dict:
    """
    Try to classify using user's personal trained adapter first.
    Falls back to LLM if adapter not found or fails.
    """
    try:
        # Model name format: ethical-{first_8_chars_of_user_id}-{version}
        model_name = f"ethical-{user_id[:8]}-v1"
        logger.info(f"🤖 Trying personal adapter: {model_name}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"""Classify this text into ONE category:

Text: "{text}"
Language: {lang.upper()}

Categories:
- growth_memory: Positive emotions, learning, achievement, gratitude
- challenge_memory: Negative emotions, stress, conflict, disappointment
- wisdom_moment: Deep insights, philosophical reflection
- needs_support: Crisis, severe distress
- neutral_interaction: Casual conversation

Respond with ONLY the category name (no explanation).""",
                    "stream": False,
                    "options": {"temperature": 0.2}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                classification = data.get("response", "").strip().lower().replace(' ', '_').replace('-', '_')
                
                valid = ['growth_memory', 'challenge_memory', 'wisdom_moment', 
                        'needs_support', 'neutral_interaction']
                
                if classification in valid:
                    logger.info(f"✅ Personal adapter classified as: {classification}")
                    scores = get_default_ethical_scores(classification)
                    return {
                        'classification': classification,
                        **scores,
                        'reasoning': f'Personal adapter: {model_name}'
                    }
                else:
                    logger.warning(f"⚠️ Invalid classification '{classification}' from adapter")
    
    except Exception as e:
        logger.warning(f"⚠️ Personal adapter failed: {e}")
    
    # Fallback to LLM
    logger.info(f"🔄 Fallback to generic LLM")
    return await classify_with_llm(text, lang)

def get_default_ethical_scores(classification: str) -> Dict:
    """Default ethical scores based on classification"""
    scores_map = {
        'growth_memory': {
            'self_awareness': 0.7, 'emotional_regulation': 0.7, 'compassion': 0.7,
            'integrity': 0.6, 'growth_mindset': 0.8, 'wisdom': 0.6, 'transcendence': 0.6
        },
        'challenge_memory': {
            'self_awareness': 0.4, 'emotional_regulation': 0.3, 'compassion': 0.5,
            'integrity': 0.5, 'growth_mindset': 0.4, 'wisdom': 0.4, 'transcendence': 0.3
        },
        'wisdom_moment': {
            'self_awareness': 0.8, 'emotional_regulation': 0.7, 'compassion': 0.7,
            'integrity': 0.7, 'growth_mindset': 0.7, 'wisdom': 0.9, 'transcendence': 0.8
        },
        'needs_support': {
            'self_awareness': 0.3, 'emotional_regulation': 0.2, 'compassion': 0.4,
            'integrity': 0.4, 'growth_mindset': 0.3, 'wisdom': 0.3, 'transcendence': 0.2
        },
        'neutral_interaction': {
            'self_awareness': 0.5, 'emotional_regulation': 0.5, 'compassion': 0.5,
            'integrity': 0.5, 'growth_mindset': 0.5, 'wisdom': 0.5, 'transcendence': 0.4
        }
    }
    return scores_map.get(classification, scores_map['neutral_interaction'])

# ============================================================
# LLM Classification (Fallback)
# ============================================================

async def classify_with_llm(text: str, lang: str) -> Dict:
    """Generic LLM classification when personal adapter not available"""
    
    examples = """
Examples:
- "I learned something new today" → growth_memory
- "Too much work, very stressed" → challenge_memory
- "Sometimes slowing down reveals what matters" → wisdom_moment
- "Feel very discouraged" → needs_support
- "The weather is nice" → neutral_interaction
"""
    
    prompt = f"""You are an ethical growth analyst. Classify this text.

Text: "{text}"
Language: {lang.upper()}

{examples}

Categories:
- growth_memory: Positive emotions, learning, achievement
- challenge_memory: Negative emotions, stress, conflict
- wisdom_moment: Deep insights, philosophical reflection
- needs_support: Crisis, severe distress
- neutral_interaction: Casual conversation

Respond ONLY with valid JSON (no markdown):
{{
  "classification": "category_name",
  "self_awareness": 0.7,
  "emotional_regulation": 0.6,
  "compassion": 0.7,
  "integrity": 0.6,
  "growth_mindset": 0.7,
  "wisdom": 0.6,
  "transcendence": 0.5,
  "reasoning": "brief explanation"
}}"""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2}
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"⚠️ LLM API failed with status {response.status_code}")
                return get_fallback_classification(text, lang)
            
            data = response.json()
            llm_response = data.get("response", "")
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response)
            if json_match:
                result = json.loads(json_match.group())
                
                valid_classifications = [
                    'growth_memory', 'challenge_memory', 'wisdom_moment',
                    'needs_support', 'neutral_interaction'
                ]
                
                if result.get('classification') not in valid_classifications:
                    logger.warning(f"⚠️ Invalid classification from LLM")
                    return get_fallback_classification(text, lang)
                
                # Normalize scores
                for key in ['self_awareness', 'emotional_regulation', 'compassion',
                           'integrity', 'growth_mindset', 'wisdom', 'transcendence']:
                    if key not in result:
                        result[key] = 0.5
                    result[key] = max(0.0, min(1.0, float(result[key])))
                
                logger.info(f"✅ LLM classified as: {result['classification']}")
                return result
            else:
                logger.warning(f"⚠️ Could not parse LLM JSON")
                return get_fallback_classification(text, lang)
                
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        return get_fallback_classification(text, lang)

# ============================================================
# Keyword-based Classification (Last Resort Fallback)
# ============================================================

def get_fallback_classification(text: str, lang: str) -> Dict:
    """Keyword-based classification as last resort"""
    text_lower = text.lower()
    
    # Thai keywords
    growth_keywords_th = [
        'รัก', 'ขอบคุณ', 'เรียนรู้', 'สำเร็จ', 'ดีใจ', 'สุข', 'ภูมิใจ',
        'พัฒนา', 'เติบโต', 'ก้าวหน้า', 'ทำได้', 'ดีงาม', 'ใจดี'
    ]
    
    challenge_keywords_th = [
        'โกรธ', 'เกลียด', 'เครียด', 'ผิดหวัง', 'เสียใจ', 'เหงา', 
        'ท้อแท้', 'ล้มเหลว', 'สอบตก', 'ทะเลาะ'
    ]
    
    wisdom_keywords_th = [
        'ปัญญา', 'สติ', 'สมาธิ', 'ธรรมะ', 'ยอมรับ', 'สงบ'
    ]
    
    # English keywords
    growth_keywords_en = [
        'love', 'thank', 'grateful', 'happy', 'joy', 'proud', 'success',
        'learn', 'improve', 'grow', 'achieve', 'kind', 'help'
    ]
    
    challenge_keywords_en = [
        'angry', 'hate', 'stress', 'sad', 'lonely', 'discourage',
        'fail', 'lost', 'argue', 'hurt'
    ]
    
    wisdom_keywords_en = [
        'wisdom', 'insight', 'enlightenment', 'meditation', 'truth',
        'understanding', 'awareness', 'accept', 'peaceful'
    ]
    
    # Select keywords based on language
    if lang == 'th':
        growth_kw = growth_keywords_th
        challenge_kw = challenge_keywords_th
        wisdom_kw = wisdom_keywords_th
    else:
        growth_kw = growth_keywords_en
        challenge_kw = challenge_keywords_en
        wisdom_kw = wisdom_keywords_en
    
    # Find matches
    growth_matches = [kw for kw in growth_kw if kw in text_lower]
    challenge_matches = [kw for kw in challenge_kw if kw in text_lower]
    wisdom_matches = [kw for kw in wisdom_kw if kw in text_lower]
    
    logger.info(f"🔍 Keyword matches: Growth={len(growth_matches)}, Challenge={len(challenge_matches)}, Wisdom={len(wisdom_matches)}")
    
    # Priority: Challenge > Wisdom > Growth > Neutral
    if challenge_matches:
        return {
            'classification': 'challenge_memory',
            **get_default_ethical_scores('challenge_memory'),
            'reasoning': f'Keywords: {", ".join(challenge_matches[:3])}'
        }
    
    if wisdom_matches:
        return {
            'classification': 'wisdom_moment',
            **get_default_ethical_scores('wisdom_moment'),
            'reasoning': f'Keywords: {", ".join(wisdom_matches[:3])}'
        }
    
    if growth_matches:
        return {
            'classification': 'growth_memory',
            **get_default_ethical_scores('growth_memory'),
            'reasoning': f'Keywords: {", ".join(growth_matches[:3])}'
        }
    
    # Default
    return {
        'classification': 'neutral_interaction',
        **get_default_ethical_scores('neutral_interaction'),
        'reasoning': 'No keywords matched'
    }

# ============================================================
# Helper Functions
# ============================================================

def detect_language(text: str) -> str:
    """Detect language from text"""
    if re.search(r'[\u0E00-\u0E7F]', text):
        return 'th'
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return 'zh'
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return 'ja'
    else:
        return 'en'

async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate text embedding"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text}
            )
            if response.status_code == 200:
                return response.json().get("embedding")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
    return None

def detect_moments(ethical_scores: Dict, classification: str) -> List[Dict]:
    """Detect significant moments"""
    moments = []
    if classification == 'needs_support':
        moments.append({
            'type': 'crisis',
            'severity': 'critical',
            'description': 'User needs support'
        })
    elif classification in ['growth_memory', 'wisdom_moment']:
        moments.append({
            'type': 'growth',
            'severity': 'positive',
            'description': 'Positive development'
        })
    return moments

def determine_growth_stage(ethical_scores: Dict[str, float]) -> int:
    """Determine growth stage 1-5"""
    avg = sum(ethical_scores.values()) / len(ethical_scores)
    if avg < 0.3: return 1
    elif avg < 0.5: return 2
    elif avg < 0.7: return 3
    elif avg < 0.85: return 4
    else: return 5

def get_guidance(classification: str, ethical_scores: Dict, lang: str) -> str:
    """Get personalized guidance message"""
    guidance_map = {
        'growth_memory': {
            'en': "Wonderful progress! Keep nurturing this growth.",
            'th': "ยอดเยี่ยมมาก! เดินหน้าต่อไปนะคะ"
        },
        'challenge_memory': {
            'en': "I understand. Take it one step at a time.",
            'th': "ฉันเข้าใจ ทีละขั้นตอนนะคะ"
        },
        'wisdom_moment': {
            'en': "Beautiful insight. This wisdom will guide you.",
            'th': "ข้อคิดที่สวยงาม ปัญญานี้จะนำทางคุณ"
        },
        'needs_support': {
            'en': "I'm here for you. Please reach out to someone you trust.",
            'th': "ฉันอยู่ตรงนี้ กรุณาติดต่อคนที่คุณไว้ใจ"
        },
        'neutral_interaction': {
            'en': "Thank you for sharing.",
            'th': "ขอบคุณที่แบ่งปัน"
        }
    }
    
    guidance = guidance_map.get(classification, guidance_map['neutral_interaction'])
    return guidance.get(lang, guidance['en'])

def get_reflection_prompt(classification: str, stage: int, lang: str) -> str:
    """Get reflection prompt"""
    prompts = {
        'growth_memory': {
            'en': "What did you learn from this experience?",
            'th': "คุณเรียนรู้อะไรจากประสบการณ์นี้บ้าง?"
        },
        'challenge_memory': {
            'en': "How can you grow from this challenge?",
            'th': "คุณจะเติบโตจากความท้าทายนี้ได้อย่างไร?"
        },
        'wisdom_moment': {
            'en': "How will this insight change your perspective?",
            'th': "ข้อคิดนี้จะเปลี่ยนมุมมองของคุณอย่างไร?"
        },
        'needs_support': {
            'en': "What support do you need right now?",
            'th': "คุณต้องการการสนับสนุนอะไรตอนนี้?"
        },
        'neutral_interaction': {
            'en': "What are you thinking about?",
            'th': "คุณกำลังคิดอะไรอยู่?"
        }
    }
    
    prompt = prompts.get(classification, prompts['neutral_interaction'])
    return prompt.get(lang, prompt['en'])

def save_interaction_memory(
    user_id: str, text: str, classification: str, ethical_scores: Dict,
    moments: List[Dict], reflection_prompt: str, gentle_guidance: str,
    memory_embedding_id: str, db_conn
):
    """Save interaction memory with training metadata"""
    cursor = db_conn.cursor()
    
    # Training weight based on classification
    weight_map = {
        'growth_memory': 1.5,
        'challenge_memory': 2.0,
        'wisdom_moment': 2.5,
        'neutral_interaction': 0.8,
        'needs_support': 1.0
    }
    training_weight = weight_map.get(classification, 1.0)
    
    cursor.execute("""
        INSERT INTO user_data_schema.interaction_memories
        (user_id, text, classification, ethical_scores, moments, reflection_prompt,
         gentle_guidance, training_weight, approved_for_training, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id
    """, (
        user_id, text, classification, json.dumps(ethical_scores), json.dumps(moments),
        reflection_prompt, gentle_guidance, training_weight,
        classification != 'needs_support',  # Auto-approve except needs_support
        json.dumps({'source': 'gating', 'memory_embedding_id': memory_embedding_id})
    ))
    
    memory_id = cursor.fetchone()[0]
    db_conn.commit()
    cursor.close()
    
    logger.info(f"✅ Memory saved (ID: {memory_id}, weight: {training_weight})")
    return memory_id

# ============================================================
# API Models & Endpoints
# ============================================================

class GatingRequest(BaseModel):
    user_id: str
    text: str
    database_url: str

class GatingResponse(BaseModel):
    status: str
    routing: str
    ethical_scores: Dict[str, float]
    growth_stage: int
    moments: List[Dict]
    reflection_prompt: str
    gentle_guidance: str
    detected_language: str
    memory_id: Optional[str] = None
    reasoning: str

@app.post("/gating/ethical-route", response_model=GatingResponse)
async def ethical_routing(request: GatingRequest):
    """Main ethical routing endpoint with personal adapter support"""
    
    logger.info(f"📝 Processing text for user {request.user_id[:8]}...")
    
    db_conn = psycopg2.connect(request.database_url)
    
    try:
        # 1. Detect language
        lang = detect_language(request.text)
        logger.info(f"🌍 Language: {lang}")
        
        # 2. Generate embedding
        embedding = await generate_embedding(request.text)
        if not embedding:
            raise HTTPException(500, "Embedding generation failed")
        
        # 3. Classify (try personal adapter first, fallback to LLM)
        result = await classify_with_ollama_adapter(request.text, request.user_id, lang)
        
        classification = result['classification']
        ethical_scores = {k: result[k] for k in [
            'self_awareness', 'emotional_regulation', 'compassion',
            'integrity', 'growth_mindset', 'wisdom', 'transcendence'
        ]}
        
        logger.info(f"✅ Classification: {classification}")
        logger.info(f"📊 Reasoning: {result.get('reasoning')}")
        
        # 4. Generate guidance and prompts
        growth_stage = determine_growth_stage(ethical_scores)
        moments = detect_moments(ethical_scores, classification)
        reflection_prompt = get_reflection_prompt(classification, growth_stage, lang)
        gentle_guidance = get_guidance(classification, ethical_scores, lang)
        
        # 5. Save memory embedding
        cursor = db_conn.cursor()
        vector_str = f"[{','.join(map(str, embedding))}]"
        cursor.execute("""
            INSERT INTO user_data_schema.memory_embeddings
            (user_id, content, embedding, metadata, created_at)
            VALUES (%s, %s, %s::vector, %s, NOW())
            RETURNING id
        """, (
            request.user_id, request.text, vector_str,
            json.dumps({'classification': classification, 'language': lang})
        ))
        memory_id = str(cursor.fetchone()[0])
        db_conn.commit()
        cursor.close()
        
        # 6. Save interaction memory
        save_interaction_memory(
            request.user_id, request.text, classification, ethical_scores,
            moments, reflection_prompt, gentle_guidance, memory_id, db_conn
        )
        
        # 7. Update ethical profile
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO user_data_schema.ethical_profiles
            (user_id, self_awareness, emotional_regulation, compassion, integrity,
             growth_mindset, wisdom, transcendence, growth_stage, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                self_awareness = EXCLUDED.self_awareness,
                emotional_regulation = EXCLUDED.emotional_regulation,
                compassion = EXCLUDED.compassion,
                integrity = EXCLUDED.integrity,
                growth_mindset = EXCLUDED.growth_mindset,
                wisdom = EXCLUDED.wisdom,
                transcendence = EXCLUDED.transcendence,
                growth_stage = EXCLUDED.growth_stage,
                total_interactions = ethical_profiles.total_interactions + 1,
                updated_at = NOW()
        """, (
            request.user_id, ethical_scores['self_awareness'],
            ethical_scores['emotional_regulation'], ethical_scores['compassion'],
            ethical_scores['integrity'], ethical_scores['growth_mindset'],
            ethical_scores['wisdom'], ethical_scores['transcendence'], growth_stage
        ))
        db_conn.commit()
        cursor.close()
        
        logger.info(f"✅ Complete: {classification}")
        
        return GatingResponse(
            status='success',
            routing=classification,
            ethical_scores=ethical_scores,
            growth_stage=growth_stage,
            moments=moments,
            reflection_prompt=reflection_prompt,
            gentle_guidance=gentle_guidance,
            detected_language=lang,
            memory_id=memory_id,
            reasoning=result.get('reasoning', 'No reasoning')
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        db_conn.close()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "6.0-complete",
        "features": [
            "✅ Personal Ollama adapter support",
            "✅ LLM fallback classification",
            "✅ Keyword fallback (last resort)",
            "✅ Multilingual (Thai/English)",
            "✅ Full LoRA training integration",
            "✅ Multi-tenant SaaS ready"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
