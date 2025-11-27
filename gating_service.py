#!/usr/bin/env python3
"""
🌍 FIXED Multilingual Ethical Growth Gating Service
✅ Fixed keyword matching (text_lower)
✅ Expanded Thai keywords
✅ Better fallback logic
✅ Full LoRA training integration
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

app = FastAPI(title="Ethical Growth Gating Service - FIXED")

OLLAMA_URL = os.getenv("OLLAMA_EXTERNAL_URL", "http://ollama.ollama.svc.cluster.local:11434")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"

# ============================================================
# ✅ FIXED: EXPANDED MULTILINGUAL KEYWORDS
# ============================================================

def get_fallback_classification(text: str, lang: str) -> Dict:
    """FIXED: Proper keyword matching with expanded Thai support"""
    text_lower = text.lower()
    
    # ✅ EXPANDED Thai keywords
    growth_keywords_th = [
        # Emotions
        'รัก', 'ขอบคุณ', 'กตัญญู', 'ดีใจ', 'มีความสุข', 'สุข', 'ภูมิใจ', 'ชื่นชม', 'ประทับใจ',
        # Learning & Growth
        'เรียนรู้', 'พัฒนา', 'เติบโต', 'สำเร็จ', 'ก้าวหน้า', 'พยายาม', 'ฝึกฝน', 'ทำได้',
        # Spiritual
        'พระพุทธเจ้า', 'พระ', 'ธรรม', 'บูชา', 'สวดมนต์', 'ทำบุญ', 'บุญ', 'กุศล',
        # Nature & Beauty
        'ธรรมชาติ', 'ต้นไม้', 'ภูเขา', 'ทะเล', 'สวยงาม', 'งดงาม', 'ซาบซึ้ง',
        # Virtue
        'ดีงาม', 'ใจดี', 'เมตตา', 'กรุณา', 'เอื้อเฟื้อ', 'ช่วยเหลือ', 'แบ่งปัน'
    ]
    
    challenge_keywords_th = [
        'ฆ่า', 'ทำร้าย', 'โกรธ', 'เกลียด', 'ทำลาย', 'ร้าย', 'แก้แค้น', 'รุนแรง', 
        'ต่อสู้', 'โกง', 'หลอกลวง', 'เครียด', 'ทะเลาะ', 'ผิดหวัง', 'เสียใจ', 
        'โดดเดี่ยว', 'เหงา', 'ท้อแท้', 'ล้มเหลว', 'สอบตก'
    ]
    
    wisdom_keywords_th = [
        'ปัญญา', 'สติ', 'สมาธิ', 'ตรัสรู้', 'ไตร่ตรอง', 'ปรัชญา', 'ธรรมะ', 
        'วิปัสสนา', 'รู้แจ้ง', 'ความสุข', 'ยอมรับ', 'ช้าลง', 'สงบ'
    ]
    
    # English keywords
    growth_keywords_en = [
        'love', 'thank', 'grateful', 'happy', 'joy', 'proud', 'success', 'achieve',
        'learn', 'improve', 'grow', 'develop', 'appreciate', 'god', 'buddha', 
        'jesus', 'allah', 'prayer', 'worship', 'nature', 'beautiful', 'tree', 
        'mountain', 'sea', 'kind', 'help', 'compassion', 'share', 'care'
    ]
    
    challenge_keywords_en = [
        'kill', 'murder', 'hurt', 'harm', 'attack', 'hate', 'destroy', 'revenge',
        'violent', 'angry', 'rage', 'fight', 'stress', 'argue', 'disappoint',
        'sad', 'lonely', 'discourage', 'fail', 'lost'
    ]
    
    wisdom_keywords_en = [
        'wisdom', 'insight', 'enlightenment', 'meditation', 'contemplation',
        'reflection', 'philosophy', 'truth', 'understanding', 'awareness',
        'accept', 'slow', 'peaceful', 'mindful'
    ]
    
    # ✅ FIX: Use text_lower for comparison
    if lang == 'th':
        growth_kw = growth_keywords_th
        challenge_kw = challenge_keywords_th
        wisdom_kw = wisdom_keywords_th
    else:
        growth_kw = growth_keywords_en
        challenge_kw = challenge_keywords_en
        wisdom_kw = wisdom_keywords_en
    
    # ✅ FIXED: Check against text_lower
    growth_matches = [kw for kw in growth_kw if kw in text_lower]
    challenge_matches = [kw for kw in challenge_kw if kw in text_lower]
    wisdom_matches = [kw for kw in wisdom_kw if kw in text_lower]
    
    logger.info(f"🔍 Keyword matches:")
    logger.info(f"   Growth: {growth_matches}")
    logger.info(f"   Challenge: {challenge_matches}")
    logger.info(f"   Wisdom: {wisdom_matches}")
    
    # Priority: Challenge > Wisdom > Growth > Neutral
    if challenge_matches:
        return {
            'classification': 'challenge_memory',
            'self_awareness': 0.4,
            'emotional_regulation': 0.3,
            'compassion': 0.5,
            'integrity': 0.5,
            'growth_mindset': 0.4,
            'wisdom': 0.4,
            'transcendence': 0.3,
            'reasoning': f'Challenge keywords detected: {", ".join(challenge_matches[:3])}'
        }
    
    if wisdom_matches:
        return {
            'classification': 'wisdom_moment',
            'self_awareness': 0.8,
            'emotional_regulation': 0.7,
            'compassion': 0.7,
            'integrity': 0.7,
            'growth_mindset': 0.7,
            'wisdom': 0.9,
            'transcendence': 0.8,
            'reasoning': f'Wisdom keywords detected: {", ".join(wisdom_matches[:3])}'
        }
    
    if growth_matches:
        return {
            'classification': 'growth_memory',
            'self_awareness': 0.7,
            'emotional_regulation': 0.7,
            'compassion': 0.7,
            'integrity': 0.6,
            'growth_mindset': 0.8,
            'wisdom': 0.6,
            'transcendence': 0.6,
            'reasoning': f'Growth keywords detected: {", ".join(growth_matches[:3])}'
        }
    
    # Default neutral
    return {
        'classification': 'neutral_interaction',
        'self_awareness': 0.5,
        'emotional_regulation': 0.5,
        'compassion': 0.5,
        'integrity': 0.5,
        'growth_mindset': 0.5,
        'wisdom': 0.5,
        'transcendence': 0.4,
        'reasoning': f'No significant keywords detected'
    }

# ============================================================
# ✅ IMPROVED: LLM Classification with Examples
# ============================================================

async def classify_with_llm(text: str, lang: str) -> Dict:
    """Improved LLM classification with examples"""
    
    # ✅ Add examples for better classification
    examples = """
Examples:
- "I cooked a new dish today and succeeded, feeling proud" → growth_memory
- "Started exercising 3 times a week, feeling stronger" → growth_memory
- "Too much work, stressed and can't sleep" → challenge_memory
- "Fought with my best friend, feel very sad" → challenge_memory
- "Sometimes slowing down helps see what's important" → wisdom_moment
- "Learned that happiness comes from within, not outside" → wisdom_moment
- "The weather is nice today" → neutral_interaction
- "Feel very discouraged today" → needs_support
"""
    
    prompt = f"""You are an ethical growth analyst. Classify this text into ONE category.

Text: "{text}"
Language: {lang.upper()}

{examples}

Categories:
- growth_memory: Positive emotions, learning, achievement, gratitude, spiritual growth
- challenge_memory: Negative emotions, stress, conflict, disappointment, failure
- wisdom_moment: Deep insights, philosophical reflection, self-discovery
- needs_support: Crisis, severe distress, hopelessness
- neutral_interaction: Casual conversation, factual statements

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
                logger.warning(f"⚠️ LLM failed, using fallback")
                return get_fallback_classification(text, lang)
            
            data = response.json()
            llm_response = data.get("response", "")
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response)
            if json_match:
                result = json.loads(json_match.group())
                
                valid_classifications = [
                    'growth_memory', 'challenge_memory', 'wisdom_moment',
                    'needs_support', 'neutral_interaction'
                ]
                
                if result.get('classification') not in valid_classifications:
                    logger.warning(f"⚠️ Invalid classification from LLM, using fallback")
                    return get_fallback_classification(text, lang)
                
                for key in ['self_awareness', 'emotional_regulation', 'compassion',
                           'integrity', 'growth_mindset', 'wisdom', 'transcendence']:
                    if key not in result:
                        result[key] = 0.5
                    result[key] = max(0.0, min(1.0, float(result[key])))
                
                logger.info(f"✅ LLM classified as: {result['classification']}")
                return result
            else:
                logger.warning(f"⚠️ Could not parse LLM JSON, using fallback")
                return get_fallback_classification(text, lang)
                
    except Exception as e:
        logger.error(f"❌ LLM error: {e}, using fallback")
        return get_fallback_classification(text, lang)

# ============================================================
# ✅ FIXED: Complete guidance for all classifications
# ============================================================

def get_guidance(classification: str, ethical_scores: Dict, lang: str) -> str:
    """Always return guidance (never None)"""
    
    guidance_map = {
        'growth_memory': {
            'en': "Wonderful progress! Keep nurturing this positive growth.",
            'th': "ยอดเยี่ยมมาก! เดินหน้าต่อไปนะคะ"
        },
        'challenge_memory': {
            'en': "I understand this is challenging. Take it one step at a time.",
            'th': "ฉันเข้าใจว่ามันยาก ทีละขั้นตอนนะคะ"
        },
        'wisdom_moment': {
            'en': "Beautiful insight. This wisdom will guide you forward.",
            'th': "ข้อคิดที่สวยงามมาก ปัญญานี้จะนำทางคุณต่อไป"
        },
        'needs_support': {
            'en': "I'm here for you. Please reach out to someone you trust.",
            'th': "ฉันอยู่ตรงนี้นะ กรุณาติดต่อคนที่คุณไว้ใจ"
        },
        'neutral_interaction': {
            'en': "Thank you for sharing.",
            'th': "ขอบคุณที่แบ่งปันนะคะ"
        }
    }
    
    guidance = guidance_map.get(classification, guidance_map['neutral_interaction'])
    return guidance.get(lang, guidance['en'])

def get_reflection_prompt(classification: str, stage: int, lang: str) -> str:
    """Always return reflection prompt (never None)"""
    
    prompts = {
        'growth_memory': {
            'en': "What did you learn from this experience?",
            'th': "คุณได้เรียนรู้อะไรจากประสบการณ์นี้บ้าง?"
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

# ============================================================
# ✅ FIXED: Save to interaction_memories with ALL required fields
# ============================================================

def save_interaction_memory(
    user_id: str,
    text: str,
    classification: str,
    ethical_scores: Dict,
    moments: List[Dict],
    reflection_prompt: str,
    gentle_guidance: str,
    memory_embedding_id: str,
    db_conn
):
    """Save with ALL required fields for LoRA training"""
    cursor = db_conn.cursor()
    
    # ✅ Calculate training_weight based on classification
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
        (user_id, text, classification, ethical_scores, moments,
         reflection_prompt, gentle_guidance, training_weight,
         approved_for_training, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id
    """, (
        user_id,
        text,
        classification,
        json.dumps(ethical_scores),
        json.dumps(moments),
        reflection_prompt,
        gentle_guidance,
        training_weight,
        classification != 'needs_support',  # Auto-approve except needs_support
        json.dumps({
            'source': 'gating_service',
            'memory_embedding_id': memory_embedding_id
        })
    ))
    
    memory_id = cursor.fetchone()[0]
    db_conn.commit()
    cursor.close()
    
    logger.info(f"✅ Interaction memory saved (ID: {memory_id}, weight: {training_weight})")
    return memory_id

# ============================================================
# Other functions remain the same...
# ============================================================

def detect_language(text: str) -> str:
    """Language detection"""
    if re.search(r'[\u0E00-\u0E7F]', text):
        return 'th'
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return 'zh'
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return 'ja'
    else:
        return 'en'

async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("embedding")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
    return None

def detect_moments(ethical_scores: Dict, classification: str) -> List[Dict]:
    """Detect moments"""
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
            'description': 'Positive development detected'
        })
    return moments

def determine_growth_stage(ethical_scores: Dict[str, float]) -> int:
    """Determine stage"""
    avg = sum(ethical_scores.values()) / len(ethical_scores)
    if avg < 0.3: return 1
    elif avg < 0.5: return 2
    elif avg < 0.7: return 3
    elif avg < 0.85: return 4
    else: return 5

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
    """Fixed ethical routing with proper classification"""
    
    logger.info(f"📝 Processing: {request.text[:100]}...")
    
    db_conn = psycopg2.connect(request.database_url)
    
    try:
        # 1. Detect language
        lang = detect_language(request.text)
        logger.info(f"🌍 Language: {lang}")
        
        # 2. Generate embedding
        embedding = await generate_embedding(request.text)
        if not embedding:
            raise HTTPException(500, "Embedding generation failed")
        
        # 3. Classify with LLM (falls back to keywords if LLM fails)
        logger.info(f"🤖 Classifying with LLM...")
        result = await classify_with_llm(request.text, lang)
        
        classification = result['classification']
        ethical_scores = {k: result[k] for k in [
            'self_awareness', 'emotional_regulation', 'compassion',
            'integrity', 'growth_mindset', 'wisdom', 'transcendence'
        ]}
        
        logger.info(f"✅ Classification: {classification}")
        logger.info(f"📊 Reasoning: {result.get('reasoning')}")
        
        # 4. Get guidance and prompts (always non-None)
        growth_stage = determine_growth_stage(ethical_scores)
        moments = detect_moments(ethical_scores, classification)
        reflection_prompt = get_reflection_prompt(classification, growth_stage, lang)
        gentle_guidance = get_guidance(classification, ethical_scores, lang)
        
        # 5. Save memory with embedding
        cursor = db_conn.cursor()
        vector_str = f"[{','.join(map(str, embedding))}]"
        cursor.execute("""
            INSERT INTO user_data_schema.memory_embeddings
            (user_id, content, embedding, metadata, created_at)
            VALUES (%s, %s, %s::vector, %s, NOW())
            RETURNING id
        """, (
            request.user_id,
            request.text,
            vector_str,
            json.dumps({'classification': classification, 'language': lang})
        ))
        memory_id = str(cursor.fetchone()[0])
        db_conn.commit()
        cursor.close()
        
        # 6. Save interaction memory with ALL fields
        save_interaction_memory(
            request.user_id, request.text, classification,
            ethical_scores, moments, reflection_prompt,
            gentle_guidance, memory_id, db_conn
        )
        
        # 7. Update ethical profile
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO user_data_schema.ethical_profiles
            (user_id, self_awareness, emotional_regulation, compassion,
             integrity, growth_mindset, wisdom, transcendence, growth_stage, updated_at)
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
            request.user_id,
            ethical_scores['self_awareness'],
            ethical_scores['emotional_regulation'],
            ethical_scores['compassion'],
            ethical_scores['integrity'],
            ethical_scores['growth_mindset'],
            ethical_scores['wisdom'],
            ethical_scores['transcendence'],
            growth_stage
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
            reasoning=result.get('reasoning', 'No reasoning provided')
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
        "version": "5.0-fixed",
        "fixes": [
            "✅ Fixed keyword matching (text_lower)",
            "✅ Expanded Thai keywords",
            "✅ Better LLM prompts with examples",
            "✅ Always return guidance/prompts (never None)",
            "✅ Added training_weight field",
            "✅ Auto-approve for training (except needs_support)",
            "✅ Full LoRA training integration"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
