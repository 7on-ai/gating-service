#!/usr/bin/env python3
"""
Ethical Growth Gating Service - COMPLETE
✅ Personal Ollama Adapter Integration
✅ Multilingual Support
✅ Full LoRA Integration
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

OLLAMA_URL = os.getenv("OLLAMA_EXTERNAL_URL", "http://ollama:11434")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"

# Personal Ollama Adapter Classification
async def classify_with_ollama_adapter(text: str, user_id: str, lang: str) -> Dict:
    try:
        model_name = f"ethical-{user_id[:8]}-v1"
        logger.info(f"🤖 Trying personal adapter: {model_name}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"""Classify this text into ONE category:

Text: "{text}"

Categories: growth_memory, challenge_memory, wisdom_moment, needs_support, neutral_interaction

Respond with ONLY the category name.""",
                    "stream": False,
                    "options": {"temperature": 0.2}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                classification = data.get("response", "").strip().lower().replace(' ', '_')
                
                valid = ['growth_memory', 'challenge_memory', 'wisdom_moment', 
                        'needs_support', 'neutral_interaction']
                
                if classification in valid:
                    logger.info(f"✅ Personal adapter: {classification}")
                    scores = get_default_ethical_scores(classification)
                    return {
                        'classification': classification,
                        **scores,
                        'reasoning': f'Personal adapter: {model_name}'
                    }
    
    except Exception as e:
        logger.warning(f"⚠️ Adapter failed: {e}")
    
    logger.info(f"🔄 Fallback to LLM")
    return await classify_with_llm(text, lang)

def get_default_ethical_scores(classification: str) -> Dict:
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

async def classify_with_llm(text: str, lang: str) -> Dict:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": f"""Classify: "{text}"

Categories: growth_memory, challenge_memory, wisdom_moment, needs_support, neutral_interaction

JSON only:
{{"classification": "...", "self_awareness": 0.7, "emotional_regulation": 0.6, "compassion": 0.7, "integrity": 0.6, "growth_mindset": 0.7, "wisdom": 0.6, "transcendence": 0.5, "reasoning": "..."}}""",
                    "stream": False,
                    "options": {"temperature": 0.2}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                llm_response = data.get("response", "")
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response)
                
                if json_match:
                    result = json.loads(json_match.group())
                    valid = ['growth_memory', 'challenge_memory', 'wisdom_moment',
                            'needs_support', 'neutral_interaction']
                    
                    if result.get('classification') in valid:
                        return result
    
    except Exception as e:
        logger.error(f"LLM error: {e}")
    
    return get_fallback_classification(text, detect_language(text))

def get_fallback_classification(text: str, lang: str) -> Dict:
    text_lower = text.lower()
    
    growth_keywords = ['รัก', 'ขอบคุณ', 'เรียนรู้', 'สำเร็จ', 'love', 'thank', 'learn', 'success']
    challenge_keywords = ['โกรธ', 'เครียด', 'เสียใจ', 'angry', 'stress', 'sad']
    wisdom_keywords = ['ปัญญา', 'สติ', 'wisdom', 'insight']
    
    if any(kw in text_lower for kw in challenge_keywords):
        return {'classification': 'challenge_memory', **get_default_ethical_scores('challenge_memory'), 'reasoning': 'Keyword match'}
    if any(kw in text_lower for kw in wisdom_keywords):
        return {'classification': 'wisdom_moment', **get_default_ethical_scores('wisdom_moment'), 'reasoning': 'Keyword match'}
    if any(kw in text_lower for kw in growth_keywords):
        return {'classification': 'growth_memory', **get_default_ethical_scores('growth_memory'), 'reasoning': 'Keyword match'}
    
    return {'classification': 'neutral_interaction', **get_default_ethical_scores('neutral_interaction'), 'reasoning': 'Default'}

def detect_language(text: str) -> str:
    if re.search(r'[\u0E00-\u0E7F]', text):
        return 'th'
    return 'en'

async def generate_embedding(text: str) -> Optional[List[float]]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text}
            )
            if response.status_code == 200:
                return response.json().get("embedding")
    except:
        pass
    return None

def detect_moments(ethical_scores: Dict, classification: str) -> List[Dict]:
    moments = []
    if classification == 'needs_support':
        moments.append({'type': 'crisis', 'severity': 'critical', 'description': 'User needs support'})
    elif classification in ['growth_memory', 'wisdom_moment']:
        moments.append({'type': 'growth', 'severity': 'positive', 'description': 'Positive development'})
    return moments

def determine_growth_stage(ethical_scores: Dict[str, float]) -> int:
    avg = sum(ethical_scores.values()) / len(ethical_scores)
    if avg < 0.3: return 1
    elif avg < 0.5: return 2
    elif avg < 0.7: return 3
    elif avg < 0.85: return 4
    else: return 5

def get_guidance(classification: str, ethical_scores: Dict, lang: str) -> str:
    guidance = {
        'growth_memory': {'en': "Wonderful progress!", 'th': "ยอดเยี่ยม!"},
        'challenge_memory': {'en': "Take it one step at a time.", 'th': "ทีละขั้นนะคะ"},
        'wisdom_moment': {'en': "Beautiful insight.", 'th': "ข้อคิดที่สวยงาม"},
        'needs_support': {'en': "I'm here for you.", 'th': "ฉันอยู่ตรงนี้"},
        'neutral_interaction': {'en': "Thank you for sharing.", 'th': "ขอบคุณ"}
    }
    return guidance.get(classification, guidance['neutral_interaction']).get(lang, guidance[classification]['en'])

def get_reflection_prompt(classification: str, stage: int, lang: str) -> str:
    prompts = {
        'growth_memory': {'en': "What did you learn?", 'th': "คุณเรียนรู้อะไร?"},
        'challenge_memory': {'en': "How can you grow?", 'th': "คุณจะเติบโตได้อย่างไร?"},
        'wisdom_moment': {'en': "How will this insight help?", 'th': "ข้อคิดนี้จะช่วยอย่างไร?"},
        'needs_support': {'en': "What support do you need?", 'th': "ต้องการความช่วยเหลืออะไร?"},
        'neutral_interaction': {'en': "What are you thinking?", 'th': "คุณคิดอะไรอยู่?"}
    }
    return prompts.get(classification, prompts['neutral_interaction']).get(lang, prompts[classification]['en'])

def save_interaction_memory(user_id, text, classification, ethical_scores, moments, 
                            reflection_prompt, gentle_guidance, memory_embedding_id, db_conn):
    cursor = db_conn.cursor()
    weight_map = {'growth_memory': 1.5, 'challenge_memory': 2.0, 'wisdom_moment': 2.5, 
                  'neutral_interaction': 0.8, 'needs_support': 1.0}
    training_weight = weight_map.get(classification, 1.0)
    
    cursor.execute("""
        INSERT INTO user_data_schema.interaction_memories
        (user_id, text, classification, ethical_scores, moments, reflection_prompt, 
         gentle_guidance, training_weight, approved_for_training, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id
    """, (user_id, text, classification, json.dumps(ethical_scores), json.dumps(moments),
          reflection_prompt, gentle_guidance, training_weight, classification != 'needs_support',
          json.dumps({'source': 'gating', 'memory_embedding_id': memory_embedding_id})))
    
    memory_id = cursor.fetchone()[0]
    db_conn.commit()
    cursor.close()
    return memory_id

# API Models
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
    db_conn = psycopg2.connect(request.database_url)
    
    try:
        lang = detect_language(request.text)
        embedding = await generate_embedding(request.text)
        if not embedding:
            raise HTTPException(500, "Embedding failed")
        
        result = await classify_with_ollama_adapter(request.text, request.user_id, lang)
        
        classification = result['classification']
        ethical_scores = {k: result[k] for k in ['self_awareness', 'emotional_regulation', 
                         'compassion', 'integrity', 'growth_mindset', 'wisdom', 'transcendence']}
        
        growth_stage = determine_growth_stage(ethical_scores)
        moments = detect_moments(ethical_scores, classification)
        reflection_prompt = get_reflection_prompt(classification, growth_stage, lang)
        gentle_guidance = get_guidance(classification, ethical_scores, lang)
        
        cursor = db_conn.cursor()
        vector_str = f"[{','.join(map(str, embedding))}]"
        cursor.execute("""
            INSERT INTO user_data_schema.memory_embeddings
            (user_id, content, embedding, metadata, created_at)
            VALUES (%s, %s, %s::vector, %s, NOW()) RETURNING id
        """, (request.user_id, request.text, vector_str, 
              json.dumps({'classification': classification, 'language': lang})))
        memory_id = str(cursor.fetchone()[0])
        db_conn.commit()
        cursor.close()
        
        save_interaction_memory(request.user_id, request.text, classification, ethical_scores,
                               moments, reflection_prompt, gentle_guidance, memory_id, db_conn)
        
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
        """, (request.user_id, ethical_scores['self_awareness'], ethical_scores['emotional_regulation'],
              ethical_scores['compassion'], ethical_scores['integrity'], ethical_scores['growth_mindset'],
              ethical_scores['wisdom'], ethical_scores['transcendence'], growth_stage))
        db_conn.commit()
        cursor.close()
        
        return GatingResponse(
            status='success', routing=classification, ethical_scores=ethical_scores,
            growth_stage=growth_stage, moments=moments, reflection_prompt=reflection_prompt,
            gentle_guidance=gentle_guidance, detected_language=lang, memory_id=memory_id,
            reasoning=result.get('reasoning', 'No reasoning')
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))
    finally:
        db_conn.close()

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "6.0-complete"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
