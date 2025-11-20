#!/usr/bin/env python3
"""
🌍 Multilingual Ethical Growth Gating Service - IMPROVED THAI SUPPORT v2
✅ Better Thai keyword detection
✅ Enhanced LLM prompts for Thai language
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
# IMPROVED MULTILINGUAL CLASSIFICATION
# ============================================================

async def classify_with_llm(text: str, lang: str) -> Dict:
    """Use Ollama LLM to classify memory with BETTER multilingual support"""
    
    # ✅ IMPROVED: Better Thai examples in prompt
    if lang == 'th':
        prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านการเติบโตทางจริยธรรม วิเคราะห์ข้อความภาษาไทยนี้และตอบเป็น JSON เท่านั้น

ข้อความ: "{text}"

จัดหมวดหมู่เป็น 1 ประเภท:

หมวดหมู่:
- growth_memory: ความรู้สึกเชิงบวก ความกตัญญู การเรียนรู้ ความรัก ความเชื่อทางศาสนา (พระพุทธเจ้า พระเจ้า อัลลอฮ์) การสวดมนต์ ทำบุญ ความซาบซึ้งในธรรมชาติ
- challenge_memory: อารมณ์เชิงลบ ความก้าวร้าว ความรุนแรง ความโกรธ ความเกลียดชัง
- wisdom_moment: การไตร่ตรองลึกซึ้ง ข้อคิดทางปรัชญา การตรัสรู้ สติปัญญา
- needs_support: วิกฤต ความสิ้นหวัง ความทุกข์ระดับรุนแรง ต้องการความช่วยเหลือ
- neutral_interaction: การสนทนาทั่วไป ข้อเท็จจริง ข้อมูลเฉยๆ

ตัวอย่าง:
- "ฉันรักพระพุทธเจ้า" = growth_memory (ศาสนา/ความรัก)
- "ขอบคุณพระเจ้า" = growth_memory (ความกตัญญู/ศาสนา)
- "ต้นไม้สวยงามมาก" = growth_memory (ธรรมชาติ)
- "ฉันเกลียดเขา" = challenge_memory (ความโกรธ)
- "ชีวิตไม่มีความหมาย" = needs_support (วิกฤต)

ตอบเป็น JSON เท่านั้น (ไม่ต้องมี markdown):
{{
  "classification": "ชื่อประเภท",
  "self_awareness": 0.0-1.0,
  "emotional_regulation": 0.0-1.0,
  "compassion": 0.0-1.0,
  "integrity": 0.0-1.0,
  "growth_mindset": 0.0-1.0,
  "wisdom": 0.0-1.0,
  "transcendence": 0.0-1.0,
  "reasoning": "คำอธิบายสั้นๆ ภาษาไทย"
}}"""
    else:
        # English and other languages
        prompt = f"""You are an ethical growth analyst. Analyze this text and respond ONLY with valid JSON.

Text: "{text}"
Language: {lang.upper()}

Classify into ONE category. Consider cultural context:

Categories:
- growth_memory: Positive emotions, gratitude, spiritual/religious growth (God, Buddha, Allah, Jesus), faith, love, learning, appreciation, nature appreciation, kindness
- challenge_memory: Negative emotions, aggression, violence, anger, conflict, harm, hatred
- wisdom_moment: Deep philosophical reflection, insights, enlightenment, meditation, contemplation
- needs_support: Crisis, despair, self-harm thoughts, severe distress, hopelessness
- neutral_interaction: Everyday conversation, neutral statements, factual information

Examples:
- "I love God" = growth_memory (religious/love)
- "Thank you Buddha" = growth_memory (gratitude/religious)
- "Beautiful tree" = growth_memory (nature)
- "I hate them" = challenge_memory (anger)
- "Life is meaningless" = needs_support (crisis)

Respond with ONLY this JSON (no markdown, no explanation):
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
}}"""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower for more consistent classification
                        "top_p": 0.9,
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"LLM classification error: {response.status_code}")
                return get_fallback_classification(text, lang)
            
            data = response.json()
            llm_response = data.get("response", "")
            
            logger.info(f"🤖 LLM raw response: {llm_response[:200]}")
            
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
                    logger.warning(f"⚠️ Invalid classification: {result.get('classification')}")
                    result['classification'] = 'neutral_interaction'
                
                # Ensure all scores are present and valid
                for key in ['self_awareness', 'emotional_regulation', 'compassion', 
                           'integrity', 'growth_mindset', 'wisdom', 'transcendence']:
                    if key not in result or not isinstance(result[key], (int, float)):
                        result[key] = 0.5
                    result[key] = max(0.0, min(1.0, float(result[key])))
                
                logger.info(f"✅ LLM classified as: {result['classification']} (lang: {lang})")
                return result
            else:
                logger.warning("⚠️ Could not parse LLM JSON response")
                return get_fallback_classification(text, lang)
                
    except Exception as e:
        logger.error(f"❌ LLM classification error: {e}")
        return get_fallback_classification(text, lang)

def get_fallback_classification(text: str, lang: str) -> Dict:
    """✅ IMPROVED: Better Thai keyword detection"""
    text_lower = text.lower()
    
    logger.info(f"🔍 Fallback classification for: {text[:50]} (lang: {lang})")
    
    # ✅ ENHANCED Thai keywords
    if lang == 'th':
        growth_keywords = [
            'รัก', 'ขอบคุณ', 'กตัญญู', 'เรียนรู้', 'พัฒนา', 'เติบโต',
            'พระพุทธเจ้า', 'พระ', 'พระเจ้า', 'อัลลอฮ์', 'ธรรม', 'บูชา', 
            'สวดมนต์', 'ทำบุญ', 'ไหว้พระ', 'ศรัทธา', 'เชื่อ',
            'ธรรมชาติ', 'ต้นไม้', 'ภูเขา', 'ทะเล', 'สวยงาม', 'งดงาม',
            'ซาบซึ้ง', 'ดีงาม', 'ใจดี', 'เมตตา', 'กรุณา', 'เห็นอกเห็นใจ',
            'ช่วยเหลือ', 'แบ่งปัน', 'ให้', 'ดี', 'สุข', 'สันติ', 'สงบ'
        ]
        
        challenge_keywords = [
            'ฆ่า', 'ทำร้าย', 'โกรธ', 'เกลียด', 'ชัง', 'ทำลาย', 'ร้าย',
            'แก้แค้น', 'รุนแรง', 'ต่อสู้', 'ทะเลาะ', 'โกง', 'หลอกลวง',
            'เจ็บปวด', 'ทุกข์', 'เศร้า', 'โดดเดี่ยว'
        ]
        
        wisdom_keywords = [
            'ปัญญา', 'สติ', 'สมาธิ', 'ตรัสรู้', 'รู้แจ้ง', 'เห็นแจ้ง',
            'ไตร่ตรอง', 'คิด', 'ปรัชญา', 'ธรรมะ', 'วิปัสสนา',
            'เข้าใจ', 'รู้', 'ตระหนัก', 'หยั่งรู้'
        ]
        
        support_keywords = [
            'ฆ่าตัวตาย', 'ตาย', 'สิ้นหวัง', 'ไม่มีความหมาย', 'ไร้ค่า',
            'ทำไม่ได้', 'ยอมแพ้', 'จบชีวิต', 'ช่วยด้วย', 'วิกฤต'
        ]
        
        # Check keywords with logging
        for keyword in growth_keywords:
            if keyword in text:
                logger.info(f"✅ Thai growth keyword found: {keyword}")
                return {
                    'classification': 'growth_memory',
                    'self_awareness': 0.7,
                    'emotional_regulation': 0.6,
                    'compassion': 0.7,
                    'integrity': 0.6,
                    'growth_mindset': 0.7,
                    'wisdom': 0.6,
                    'transcendence': 0.6,
                    'reasoning': f'Thai growth keyword detected: {keyword}'
                }
        
        for keyword in support_keywords:
            if keyword in text:
                logger.info(f"⚠️ Thai support keyword found: {keyword}")
                return {
                    'classification': 'needs_support',
                    'self_awareness': 0.3,
                    'emotional_regulation': 0.2,
                    'compassion': 0.4,
                    'integrity': 0.4,
                    'growth_mindset': 0.3,
                    'wisdom': 0.3,
                    'transcendence': 0.2,
                    'reasoning': f'Thai support keyword detected: {keyword}'
                }
        
        for keyword in challenge_keywords:
            if keyword in text:
                logger.info(f"⚠️ Thai challenge keyword found: {keyword}")
                return {
                    'classification': 'challenge_memory',
                    'self_awareness': 0.3,
                    'emotional_regulation': 0.2,
                    'compassion': 0.4,
                    'integrity': 0.4,
                    'growth_mindset': 0.3,
                    'wisdom': 0.3,
                    'transcendence': 0.2,
                    'reasoning': f'Thai challenge keyword detected: {keyword}'
                }
        
        for keyword in wisdom_keywords:
            if keyword in text:
                logger.info(f"✅ Thai wisdom keyword found: {keyword}")
                return {
                    'classification': 'wisdom_moment',
                    'self_awareness': 0.7,
                    'emotional_regulation': 0.7,
                    'compassion': 0.7,
                    'integrity': 0.7,
                    'growth_mindset': 0.7,
                    'wisdom': 0.8,
                    'transcendence': 0.7,
                    'reasoning': f'Thai wisdom keyword detected: {keyword}'
                }
    
    # English and other languages
    else:
        growth_keywords = ['love', 'thank', 'grateful', 'learn', 'improve', 'grow', 'appreciate', 
                          'god', 'buddha', 'jesus', 'allah', 'prayer', 'worship', 'faith',
                          'nature', 'beautiful', 'tree', 'mountain', 'sea', 'kind', 'help', 'compassion']
        
        challenge_keywords = ['kill', 'murder', 'hurt', 'harm', 'attack', 'hate', 'destroy', 
                             'revenge', 'violent', 'angry', 'rage', 'fight']
        
        wisdom_keywords = ['wisdom', 'insight', 'enlightenment', 'meditation', 'contemplation', 
                          'reflection', 'philosophy', 'truth', 'understanding', 'awareness']
        
        support_keywords = ['suicide', 'die', 'hopeless', 'worthless', 'end it', 'kill myself',
                           'no meaning', 'give up', 'help me', 'crisis']
        
        if any(keyword in text_lower for keyword in growth_keywords):
            logger.info(f"✅ English growth keyword found")
            return {
                'classification': 'growth_memory',
                'self_awareness': 0.7,
                'emotional_regulation': 0.6,
                'compassion': 0.7,
                'integrity': 0.6,
                'growth_mindset': 0.7,
                'wisdom': 0.6,
                'transcendence': 0.6,
                'reasoning': f'Growth keyword detected in {lang}'
            }
        
        if any(keyword in text_lower for keyword in support_keywords):
            return {
                'classification': 'needs_support',
                'self_awareness': 0.3,
                'emotional_regulation': 0.2,
                'compassion': 0.4,
                'integrity': 0.4,
                'growth_mindset': 0.3,
                'wisdom': 0.3,
                'transcendence': 0.2,
                'reasoning': f'Support keyword detected in {lang}'
            }
        
        if any(keyword in text_lower for keyword in challenge_keywords):
            return {
                'classification': 'challenge_memory',
                'self_awareness': 0.3,
                'emotional_regulation': 0.2,
                'compassion': 0.4,
                'integrity': 0.4,
                'growth_mindset': 0.3,
                'wisdom': 0.3,
                'transcendence': 0.2,
                'reasoning': f'Challenge keyword detected in {lang}'
            }
        
        if any(keyword in text_lower for keyword in wisdom_keywords):
            return {
                'classification': 'wisdom_moment',
                'self_awareness': 0.7,
                'emotional_regulation': 0.7,
                'compassion': 0.7,
                'integrity': 0.7,
                'growth_mindset': 0.7,
                'wisdom': 0.8,
                'transcendence': 0.7,
                'reasoning': f'Wisdom keyword detected in {lang}'
            }
    
    # Default neutral
    logger.info(f"ℹ️ No keywords matched, defaulting to neutral")
    return {
        'classification': 'neutral_interaction',
        'self_awareness': 0.5,
        'emotional_regulation': 0.5,
        'compassion': 0.5,
        'integrity': 0.5,
        'growth_mindset': 0.5,
        'wisdom': 0.5,
        'transcendence': 0.3,
        'reasoning': f'Fallback: Neutral classification for {lang}'
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
    """Enhanced language detection"""
    # Thai
    if re.search(r'[\u0E00-\u0E7F]', text):
        return 'th'
    # Chinese
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return 'zh'
    # Japanese
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return 'ja'
    # Korean
    elif re.search(r'[\uAC00-\uD7AF]', text):
        return 'ko'
    # Arabic
    elif re.search(r'[\u0600-\u06FF]', text):
        return 'ar'
    else:
        return 'en'

def detect_moments(ethical_scores: Dict, classification: str) -> List[Dict]:
    """Detect significant moments"""
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
    """Determine growth stage"""
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
    """Save to memory_embeddings"""
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
    """Save to interaction_memories"""
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
# MAIN ENDPOINT
# ============================================================

@app.post("/gating/ethical-route", response_model=GatingResponse)
async def ethical_routing(request: GatingRequest):
    """Process text with improved Thai support"""
    
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
            logger.warning("⚠️ Embedding generation failed")
        
        # 3. LLM CLASSIFICATION with fallback
        logger.info(f"🤖 Using LLM for classification (language: {lang})...")
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
        
        logger.info(f"✅ Classification: {classification}")
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
        "version": "4.0-improved-thai",
        "supported_languages": ["en", "th", "zh", "ja", "ko", "ar", "and more"],
        "multilingual": True,
        "embedding_model": EMBEDDING_MODEL,
        "classification_model": LLM_MODEL,
        "ollama_url": OLLAMA_URL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
