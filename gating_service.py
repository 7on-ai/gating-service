#!/usr/bin/env python3
"""
🌍 Multilingual Ethical Growth Gating Service - LLM CLASSIFICATION
✅ Uses Ollama LLM for intelligent classification
✅ Creates embeddings and stores in memory_embeddings table
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ethical Growth Gating Service")

# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = os.getenv("OLLAMA_EXTERNAL_URL", "http://ollama.ollama.svc.cluster.local:11434")
EMBEDDING_MODEL = "nomic-embed-text"  # 768 dimensions
LLM_MODEL = "tinyllama"  # For classification

# ============================================================
# LLM-BASED CLASSIFICATION (NEW!)
# ============================================================

async def classify_with_llm(text: str, lang: str) -> Dict:
    """Use Ollama LLM to classify memory with ethical analysis"""
    
    # Build multilingual prompt
    prompts = {
        'en': f"""You are an ethical growth analyst. Analyze this text and respond ONLY with valid JSON.

Text: "{text}"

Classify into ONE category and provide ethical scores (0.0-1.0):

Categories:
- growth_memory: Positive learning, gratitude, spiritual growth, faith, love
- challenge_memory: Negative emotions, aggression, conflict, anger
- wisdom_moment: Deep reflection, philosophical insights, breakthrough
- needs_support: Crisis, despair, self-harm thoughts, severe distress
- neutral_interaction: Everyday conversation, factual statements

JSON format (respond with ONLY this, no other text):
{{
  "classification": "category_name",
  "self_awareness": 0.0-1.0,
  "emotional_regulation": 0.0-1.0,
  "compassion": 0.0-1.0,
  "integrity": 0.0-1.0,
  "growth_mindset": 0.0-1.0,
  "wisdom": 0.0-1.0,
  "transcendence": 0.0-1.0,
  "reasoning": "brief explanation"
}}""",
        
        'th': f"""คุณคือนักวิเคราะห์การเติบโตทางจริยธรรม วิเคราะห์ข้อความนี้และตอบเป็น JSON เท่านั้น

ข้อความ: "{text}"

จัดประเภทเป็น 1 ประเภท:
- growth_memory: การเรียนรู้เชิงบวก ความกตัญญู การเติบโตทางจิตวิญญาณ
- challenge_memory: อารมณ์เชิงลบ ความก้าวร้าว ความขัดแย้ง
- wisdom_moment: การไตร่ตรองลึกซึ้ง ปัญญา
- needs_support: วิกฤต ความสิ้นหวัง ต้องการความช่วยเหลือ
- neutral_interaction: การสนทนาทั่วไป

JSON format:
{{
  "classification": "ชื่อประเภท",
  "self_awareness": 0.0-1.0,
  "emotional_regulation": 0.0-1.0,
  "compassion": 0.0-1.0,
  "integrity": 0.0-1.0,
  "growth_mindset": 0.0-1.0,
  "wisdom": 0.0-1.0,
  "transcendence": 0.0-1.0,
  "reasoning": "คำอธิบายสั้นๆ"
}}"""
    }
    
    prompt = prompts.get(lang, prompts['en'])
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower = more consistent
                        "top_p": 0.9,
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"LLM classification error: {response.status_code}")
                return get_fallback_classification(text)
            
            data = response.json()
            llm_response = data.get("response", "")
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate classification
                valid_classifications = [
                    'growth_memory', 'challenge_memory', 'wisdom_moment', 
                    'needs_support', 'neutral_interaction'
                ]
                
                if result.get('classification') not in valid_classifications:
                    result['classification'] = 'neutral_interaction'
                
                # Ensure all scores are present and valid
                for key in ['self_awareness', 'emotional_regulation', 'compassion', 
                           'integrity', 'growth_mindset', 'wisdom', 'transcendence']:
                    if key not in result or not isinstance(result[key], (int, float)):
                        result[key] = 0.5
                    result[key] = max(0.0, min(1.0, float(result[key])))
                
                logger.info(f"✅ LLM classified as: {result['classification']}")
                return result
            else:
                logger.warning("⚠️ Could not parse LLM JSON response")
                return get_fallback_classification(text)
                
    except Exception as e:
        logger.error(f"❌ LLM classification error: {e}")
        return get_fallback_classification(text)

def get_fallback_classification(text: str) -> Dict:
    """Fallback to simple rule-based classification"""
    text_lower = text.lower()
    
    # Crisis detection
    crisis_words = ['kill', 'die', 'suicide', 'end it', 'hopeless', 'worthless']
    if any(word in text_lower for word in crisis_words):
        return {
            'classification': 'challenge_memory',
            'self_awareness': 0.3,
            'emotional_regulation': 0.2,
            'compassion': 0.4,
            'integrity': 0.4,
            'growth_mindset': 0.3,
            'wisdom': 0.3,
            'transcendence': 0.2,
            'reasoning': 'Fallback: detected challenging content'
        }
    
    # Growth detection
    growth_words = ['learn', 'improve', 'grow', 'thank', 'grateful', 'love', 'god', 'buddha']
    if any(word in text_lower for word in growth_words):
        return {
            'classification': 'growth_memory',
            'self_awareness': 0.6,
            'emotional_regulation': 0.6,
            'compassion': 0.7,
            'integrity': 0.6,
            'growth_mindset': 0.7,
            'wisdom': 0.6,
            'transcendence': 0.5,
            'reasoning': 'Fallback: detected growth-oriented content'
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
        'transcendence': 0.3,
        'reasoning': 'Fallback: neutral classification'
    }

# ============================================================
# EMBEDDING GENERATION
# ============================================================

async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding using Ollama nomic-embed-text"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama error: {response.status_code}")
                return None
            
            data = response.json()
            embedding = data.get("embedding")
            
            if not embedding or len(embedding) != 768:
                logger.error(f"Invalid embedding dimension: {len(embedding) if embedding else 0}")
                return None
            
            return embedding
            
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_language(text: str) -> str:
    """Simple language detection based on character sets"""
    if re.search(r'[\u0E00-\u0E7F]', text):
        return 'th'
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return 'zh'
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return 'ja'
    elif re.search(r'[\uAC00-\uD7AF]', text):
        return 'ko'
    else:
        return 'en'

def detect_moments(ethical_scores: Dict, classification: str) -> List[Dict]:
    """Detect significant moments based on scores and classification"""
    moments = []
    
    if ethical_scores.get('self_awareness', 0) > 0.7:
        moments.append({
            'type': 'breakthrough',
            'severity': 'positive',
            'description': 'High self-awareness detected',
            'timestamp': datetime.now().isoformat()
        })
    
    if ethical_scores.get('emotional_regulation', 0) < 0.3:
        moments.append({
            'type': 'struggle',
            'severity': 'neutral',
            'description': 'Emotional difficulty detected',
            'timestamp': datetime.now().isoformat()
        })
    
    if classification == 'needs_support':
        moments.append({
            'type': 'crisis',
            'severity': 'critical',
            'description': 'User needs support',
            'timestamp': datetime.now().isoformat(),
            'requires_intervention': True
        })
    
    if classification in ['growth_memory', 'wisdom_moment']:
        moments.append({
            'type': 'growth',
            'severity': 'positive',
            'description': 'Growth or wisdom detected',
            'timestamp': datetime.now().isoformat()
        })
    
    return moments

def determine_growth_stage(ethical_scores: Dict[str, float]) -> int:
    """Determine growth stage from ethical scores"""
    avg_score = sum(ethical_scores.values()) / len(ethical_scores)
    
    if avg_score < 0.3:
        return 1
    elif avg_score < 0.5:
        return 2
    elif avg_score < 0.7:
        return 3
    elif avg_score < 0.85:
        return 4
    else:
        return 5

# ============================================================
# DATABASE OPERATIONS
# ============================================================

def save_ethical_profile(user_id: str, ethical_scores: Dict, stage: int, db_conn):
    cursor = db_conn.cursor()
    
    cursor.execute("""
        INSERT INTO user_data_schema.ethical_profiles 
        (user_id, self_awareness, emotional_regulation, compassion, 
         integrity, growth_mindset, wisdom, transcendence, growth_stage, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id) 
        DO UPDATE SET
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
        user_id,
        ethical_scores['self_awareness'],
        ethical_scores['emotional_regulation'],
        ethical_scores['compassion'],
        ethical_scores['integrity'],
        ethical_scores['growth_mindset'],
        ethical_scores['wisdom'],
        ethical_scores['transcendence'],
        stage
    ))
    
    db_conn.commit()
    cursor.close()

async def save_memory_with_embedding(
    user_id: str, 
    text: str,
    embedding: List[float],
    classification: str,
    lang: str,
    growth_stage: int,
    db_conn
) -> str:
    """Save to memory_embeddings with vector and metadata"""
    cursor = db_conn.cursor()
    
    vector_str = f"[{','.join(map(str, embedding))}]"
    
    metadata = {
        'classification': classification,
        'language': lang,
        'growth_stage': growth_stage,
        'source': 'gating_service',
        'created_at': datetime.now().isoformat()
    }
    
    cursor.execute("""
        INSERT INTO user_data_schema.memory_embeddings
        (user_id, content, embedding, metadata, created_at)
        VALUES (%s, %s, %s::vector, %s, NOW())
        RETURNING id
    """, (
        user_id,
        text,
        vector_str,
        json.dumps(metadata)
    ))
    
    memory_id = cursor.fetchone()[0]
    db_conn.commit()
    cursor.close()
    
    logger.info(f"✅ Memory saved with ID: {memory_id}")
    return str(memory_id)

def save_interaction_memory(
    user_id: str, 
    text: str, 
    classification: str,
    ethical_scores: Dict,
    moments: List[Dict],
    reflection_prompt: str,
    gentle_guidance: Optional[str],
    memory_embedding_id: str,
    db_conn
):
    """Save to interaction_memories with link to memory_embeddings"""
    cursor = db_conn.cursor()
    
    cursor.execute("""
        INSERT INTO user_data_schema.interaction_memories
        (user_id, text, classification, ethical_scores, moments, 
         reflection_prompt, gentle_guidance, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id
    """, (
        user_id,
        text,
        classification,
        json.dumps(ethical_scores),
        json.dumps(moments),
        reflection_prompt,
        gentle_guidance,
        json.dumps({
            'source': 'gating_service',
            'memory_embedding_id': memory_embedding_id
        })
    ))
    
    db_conn.commit()
    cursor.close()
    logger.info(f"✅ Interaction memory saved")

def get_user_ethical_history(user_id: str, db_conn) -> Dict:
    cursor = db_conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT * FROM user_data_schema.ethical_profiles
        WHERE user_id = %s
    """, (user_id,))
    
    profile = cursor.fetchone()
    cursor.close()
    
    if profile:
        return {
            'baseline_self_awareness': profile['self_awareness'],
            'baseline_regulation': profile['emotional_regulation'],
            'baseline_compassion': profile['compassion'],
            'baseline_integrity': profile['integrity'],
            'baseline_growth': profile['growth_mindset'],
            'baseline_wisdom': profile['wisdom'],
            'baseline_transcendence': profile['transcendence'],
            'current_stage': profile['growth_stage']
        }
    
    return {
        'baseline_self_awareness': 0.3,
        'baseline_regulation': 0.4,
        'baseline_compassion': 0.4,
        'baseline_integrity': 0.5,
        'baseline_growth': 0.4,
        'baseline_wisdom': 0.3,
        'baseline_transcendence': 0.2,
        'current_stage': 2
    }

# ============================================================
# GUIDANCE TEMPLATES
# ============================================================

GUIDANCE_TEMPLATES = {
    'crisis': {
        'en': "I'm concerned about you. Please reach out to a mental health professional.",
        'th': "ฉันเป็นห่วงคุณมาก โปรดติดต่อสายด่วนสุขภาพจิต 1323",
    },
    'emotional_dysregulation': {
        'en': "Take a deep breath. These feelings will pass.",
        'th': "ลองหายใจเข้าลึกๆ ความรู้สึกนี้จะผ่านไป",
    },
}

REFLECTION_PROMPTS = {
    1: {
        'en': "What are you feeling right now?",
        'th': "สิ่งที่คุณกำลังรู้สึกตอนนี้คืออะไร?",
    },
    2: {
        'en': "If someone else were in this situation, how would they feel?",
        'th': "ถ้าคนอื่นอยู่ในสถานการณ์นี้ เขาจะรู้สึกยังไง?",
    },
    3: {
        'en': "What values does this decision reflect?",
        'th': "การตัดสินใจนี้สะท้อนคุณค่าอะไร?",
    },
}

def get_guidance(classification: str, ethical_scores: Dict, lang: str) -> Optional[str]:
    if classification == 'needs_support':
        return GUIDANCE_TEMPLATES['crisis'].get(lang, GUIDANCE_TEMPLATES['crisis']['en'])
    
    if ethical_scores.get('emotional_regulation', 0.5) < 0.3:
        return GUIDANCE_TEMPLATES['emotional_dysregulation'].get(lang, GUIDANCE_TEMPLATES['emotional_dysregulation']['en'])
    
    return None

def get_reflection_prompt(stage: int, lang: str) -> str:
    prompts = REFLECTION_PROMPTS.get(stage, REFLECTION_PROMPTS[2])
    return prompts.get(lang, prompts.get('en', ''))

# ============================================================
# API MODELS
# ============================================================

class GatingRequest(BaseModel):
    user_id: str
    text: str
    database_url: str
    session_id: Optional[str] = None
    metadata: Optional[Dict] = {}

class GatingResponse(BaseModel):
    status: str
    routing: str
    ethical_scores: Dict[str, float]
    growth_stage: int
    moments: List[Dict]
    insights: Optional[Dict] = None
    reflection_prompt: Optional[str] = None
    gentle_guidance: Optional[str] = None
    growth_opportunity: Optional[str] = None
    detected_language: Optional[str] = None
    memory_id: Optional[str] = None

# ============================================================
# MAIN ENDPOINT - LLM CLASSIFICATION
# ============================================================

@app.post("/gating/ethical-route", response_model=GatingResponse)
async def ethical_routing(request: GatingRequest):
    """Process text through ethical growth framework with LLM classification"""
    
    logger.info(f"📝 Processing text for user {request.user_id}: {request.text[:50]}...")
    
    if not request.database_url:
        raise HTTPException(status_code=400, detail="database_url is required")
    
    db_conn = psycopg2.connect(request.database_url)
    
    try:
        # 1. Detect language
        lang = detect_language(request.text)
        logger.info(f"🌍 Detected language: {lang}")
        
        # 2. Generate embedding
        logger.info(f"🧠 Generating embedding...")
        embedding = await generate_embedding(request.text)
        
        if not embedding:
            logger.warning("⚠️  Embedding generation failed")
        
        # 3. ✅ LLM CLASSIFICATION (NEW!)
        logger.info(f"🤖 Using LLM for classification...")
        llm_result = await classify_with_llm(request.text, lang)
        
        classification = llm_result['classification']
        ethical_scores = {
            'self_awareness': llm_result['self_awareness'],
            'emotional_regulation': llm_result['emotional_regulation'],
            'compassion': llm_result['compassion'],
            'integrity': llm_result['integrity'],
            'growth_mindset': llm_result['growth_mindset'],
            'wisdom': llm_result['wisdom'],
            'transcendence': llm_result['transcendence'],
        }
        
        logger.info(f"✅ LLM Classification: {classification}")
        logger.info(f"📊 Reasoning: {llm_result.get('reasoning', 'N/A')}")
        
        # 4. Determine growth stage
        growth_stage = determine_growth_stage(ethical_scores)
        
        # 5. Detect moments
        moments = detect_moments(ethical_scores, classification)
        
        # 6. Generate guidance
        reflection_prompt = get_reflection_prompt(growth_stage, lang)
        gentle_guidance = get_guidance(classification, ethical_scores, lang)
        
        # 7. Save to memory_embeddings
        memory_id = None
        if embedding:
            logger.info(f"💾 Saving to memory_embeddings...")
            memory_id = await save_memory_with_embedding(
                request.user_id,
                request.text,
                embedding,
                classification,
                lang,
                growth_stage,
                db_conn
            )
        else:
            logger.error("❌ Cannot save without embedding")
            raise HTTPException(status_code=500, detail="Embedding generation failed")
        
        # 8. Save ethical profile
        save_ethical_profile(request.user_id, ethical_scores, growth_stage, db_conn)
        
        # 9. Save interaction memory
        save_interaction_memory(
            request.user_id,
            request.text,
            classification,
            ethical_scores,
            moments,
            reflection_prompt,
            gentle_guidance,
            memory_id,
            db_conn
        )
        
        logger.info(f"✅ Processing completed: {classification}")
        
        return GatingResponse(
            status='success',
            routing=classification,
            ethical_scores=ethical_scores,
            growth_stage=growth_stage,
            moments=moments,
            insights={
                'strongest_dimension': max(ethical_scores, key=ethical_scores.get),
                'growth_area': min(ethical_scores, key=ethical_scores.get),
                'llm_reasoning': llm_result.get('reasoning', 'N/A')
            },
            reflection_prompt=reflection_prompt,
            gentle_guidance=gentle_guidance,
            growth_opportunity=f"Stage {growth_stage}/5",
            detected_language=lang,
            memory_id=memory_id
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_conn.close()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "ethical_growth_gating",
        "version": "3.0-llm",
        "multilingual": True,
        "embedding_model": EMBEDDING_MODEL,
        "classification_model": LLM_MODEL,
        "ollama_url": OLLAMA_URL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
