#!/usr/bin/env python3
"""
🌍 Multilingual Ethical Growth Gating Service - COMPLETE FIX
✅ Creates embeddings and stores in memory_embeddings table
✅ Links to interaction_memories properly
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ethical Growth Gating Service")

# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://ollama.ollama.svc.cluster.local:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # 768 dimensions

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
# MULTILINGUAL PATTERNS
# ============================================================

class MultilingualPatterns:
    """Language-agnostic patterns using sentiment and semantic markers"""
    
    SELF_REFLECTION = {
        'en': ['why', 'because', 'should i', 'is it right', 'wonder if', 'thinking about'],
        'th': ['ทำไม', 'เพราะอะไร', 'ควรไหม', 'ถูกหรือเปล่า', 'สงสัย', 'คิดว่า'],
        'zh': ['为什么', '因为', '应该', '是否', '想知道', '思考'],
        'ja': ['なぜ', 'どうして', 'すべき', '正しい', '考える', '思う'],
        'ko': ['왜', '때문에', '해야', '옳은', '생각', '궁금'],
        'universal': ['?', '...', '🤔']
    }
    
    EMPATHY = {
        'en': ['they feel', 'if i were', 'understand feeling', 'their perspective'],
        'th': ['เขารู้สึกยังไง', 'ถ้าเป็นเขา', 'เข้าใจความรู้สึก', 'มุมมองเขา'],
        'zh': ['他们感觉', '如果我是', '理解感受', '他们的角度'],
        'universal': ['❤️', '🫂', '💝']
    }
    
    ACCOUNTABILITY = {
        'en': ['my fault', 'responsible', 'apologize', 'fix', 'my mistake'],
        'th': ['ผิดของฉัน', 'รับผิดชอบ', 'ขอโทษ', 'แก้ไข', 'ความผิดพลาด'],
        'zh': ['我的错', '负责', '道歉', '改正', '我的错误'],
        'universal': ['🙏', '🙇']
    }
    
    GRATITUDE = {
        'en': ['thank', 'grateful', 'appreciate', 'lucky', 'blessed'],
        'th': ['ขอบคุณ', 'ขอบใจ', 'ดีใจที่มี', 'โชคดีที่', 'มีความสุข'],
        'zh': ['谢谢', '感激', '感谢', '幸运', '感恩'],
        'universal': ['🙏', '❤️', '😊', '💖']
    }
    
    GROWTH_SEEKING = {
        'en': ['learn', 'improve', 'develop', 'change', 'grow', 'better'],
        'th': ['เรียนรู้', 'พัฒนา', 'ทำให้ดีขึ้น', 'เปลี่ยนแปลง', 'เติบโต'],
        'zh': ['学习', '改进', '发展', '改变', '成长', '更好'],
        'universal': ['📚', '🌱', '💪', '⬆️']
    }
    
    AGGRESSION = {
        'en': ['kill', 'hurt', 'harm', 'attack', 'destroy', 'hate'],
        'th': ['ฆ่า', 'ทำร้าย', 'เจ็บ', 'โจมตี', 'ทำลาย', 'เกลียด'],
        'zh': ['杀', '伤害', '攻击', '破坏', '恨'],
        'universal': ['🔪', '💀', '😡', '🤬']
    }
    
    DESPAIR = {
        'en': ['want to die', 'meaningless', 'worthless', 'hopeless', 'end it'],
        'th': ['อยากตาย', 'ไม่มีความหมาย', 'ไร้ค่า', 'สิ้นหวัง', 'จบชีวิต'],
        'zh': ['想死', '无意义', '无价值', '绝望', '结束'],
        'universal': ['💔', '😭', '🖤']
    }

PATTERNS = MultilingualPatterns()

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

def score_pattern_multilingual(text: str, pattern_dict: Dict, lang: str = None) -> float:
    if not lang:
        lang = detect_language(text)
    
    text_lower = text.lower()
    patterns = pattern_dict.get(lang, []) + pattern_dict.get('universal', [])
    
    if not patterns:
        patterns = pattern_dict.get('en', []) + pattern_dict.get('universal', [])
    
    matches = sum(1 for pattern in patterns if pattern in text_lower)
    return min(matches / max(len(patterns) * 0.3, 1), 1.0)

def analyze_ethical_dimensions_multilingual(text: str, user_history: Dict) -> Dict[str, float]:
    lang = detect_language(text)
    scores = {}
    
    self_aware_score = score_pattern_multilingual(text, PATTERNS.SELF_REFLECTION, lang)
    scores['self_awareness'] = min(1.0, 
        self_aware_score * 0.7 + user_history.get('baseline_self_awareness', 0.3)
    )
    
    aggression_score = score_pattern_multilingual(text, PATTERNS.AGGRESSION, lang)
    scores['emotional_regulation'] = max(0.0, min(1.0,
        (1.0 - aggression_score) * 0.7 + user_history.get('baseline_regulation', 0.4)
    ))
    
    empathy_score = score_pattern_multilingual(text, PATTERNS.EMPATHY, lang)
    scores['compassion'] = min(1.0,
        empathy_score * 0.7 + user_history.get('baseline_compassion', 0.4)
    )
    
    accountability_score = score_pattern_multilingual(text, PATTERNS.ACCOUNTABILITY, lang)
    scores['integrity'] = min(1.0,
        accountability_score * 0.7 + user_history.get('baseline_integrity', 0.5)
    )
    
    growth_score = score_pattern_multilingual(text, PATTERNS.GROWTH_SEEKING, lang)
    scores['growth_mindset'] = min(1.0,
        growth_score * 0.7 + user_history.get('baseline_growth', 0.4)
    )
    
    wisdom_score = (self_aware_score + empathy_score) / 2
    scores['wisdom'] = min(1.0,
        wisdom_score * 0.6 + user_history.get('baseline_wisdom', 0.3)
    )
    
    gratitude_score = score_pattern_multilingual(text, PATTERNS.GRATITUDE, lang)
    transcendent_score = (gratitude_score + growth_score) / 2
    scores['transcendence'] = min(1.0,
        transcendent_score * 0.5 + user_history.get('baseline_transcendence', 0.2)
    )
    
    return scores

def detect_moments_multilingual(text: str, ethical_scores: Dict) -> List[Dict]:
    lang = detect_language(text)
    moments = []
    
    reflection_score = score_pattern_multilingual(text, PATTERNS.SELF_REFLECTION, lang)
    if reflection_score > 0.6:
        moments.append({
            'type': 'breakthrough',
            'severity': 'positive',
            'description': 'User shows self-reflection',
            'timestamp': datetime.now().isoformat()
        })
    
    if ethical_scores.get('emotional_regulation', 0.5) < 0.3:
        moments.append({
            'type': 'struggle',
            'severity': 'neutral',
            'description': 'User experiencing difficulty',
            'timestamp': datetime.now().isoformat()
        })
    
    despair_score = score_pattern_multilingual(text, PATTERNS.DESPAIR, lang)
    if despair_score > 0.5:
        moments.append({
            'type': 'crisis',
            'severity': 'critical',
            'description': 'User in emotional crisis',
            'timestamp': datetime.now().isoformat(),
            'requires_intervention': True
        })
    
    growth_score = score_pattern_multilingual(text, PATTERNS.GROWTH_SEEKING, lang)
    if growth_score > 0.6:
        moments.append({
            'type': 'growth',
            'severity': 'positive',
            'description': 'User showing growth mindset',
            'timestamp': datetime.now().isoformat()
        })
    
    return moments

def determine_growth_stage(ethical_scores: Dict[str, float]) -> int:
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

def classify_for_learning(text: str, ethical_scores: Dict, moments: List[Dict], stage: int) -> str:
    if any(m.get('severity') == 'critical' for m in moments):
        return 'needs_support'
    
    growth_moments = [m for m in moments if m.get('type') == 'growth']
    if growth_moments or sum(ethical_scores.values()) / len(ethical_scores) > 0.7:
        return 'growth_memory'
    
    if any(m.get('type') == 'breakthrough' for m in moments):
        return 'wisdom_moment'
    
    if any(m.get('type') == 'struggle' for m in moments):
        return 'challenge_memory'
    
    return 'neutral_interaction'

# ============================================================
# DATABASE OPERATIONS - FIXED VERSION
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
    """✅ Save to memory_embeddings with vector and metadata"""
    cursor = db_conn.cursor()
    
    # Convert embedding to PostgreSQL vector format
    vector_str = f"[{','.join(map(str, embedding))}]"
    
    # ✅ Store classification in metadata
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
        'zh': "我很担心你。请联系心理健康专业人士。",
    },
    'emotional_dysregulation': {
        'en': "Take a deep breath. These feelings will pass.",
        'th': "ลองหายใจเข้าลึกๆ ความรู้สึกนี้จะผ่านไป",
        'zh': "深呼吸。这些感觉会过去的。",
    },
}

REFLECTION_PROMPTS = {
    1: {
        'en': "What are you feeling right now?",
        'th': "สิ่งที่คุณกำลังรู้สึกตอนนี้คืออะไร?",
        'zh': "你现在感觉如何？",
    },
    2: {
        'en': "If someone else were in this situation, how would they feel?",
        'th': "ถ้าคนอื่นอยู่ในสถานการณ์นี้ เขาจะรู้สึกยังไง?",
        'zh': "如果其他人处于这种情况，他们会有什么感受？",
    },
    3: {
        'en': "What values does this decision reflect?",
        'th': "การตัดสินใจนี้สะท้อนคุณค่าอะไร?",
        'zh': "这个决定反映了什么价值观？",
    },
}

def get_guidance_multilingual(template_key: str, lang: str) -> str:
    templates = GUIDANCE_TEMPLATES.get(template_key, {})
    return templates.get(lang, templates.get('en', ''))

def get_reflection_prompt_multilingual(stage: int, lang: str) -> str:
    prompts = REFLECTION_PROMPTS.get(stage, REFLECTION_PROMPTS[2])
    return prompts.get(lang, prompts.get('en', ''))

def create_gentle_guidance_multilingual(moments: List[Dict], ethical_scores: Dict, lang: str) -> Optional[str]:
    crisis_moments = [m for m in moments if m.get('severity') == 'critical']
    if crisis_moments:
        return get_guidance_multilingual('crisis', lang)
    
    if ethical_scores.get('emotional_regulation', 0.5) < 0.3:
        return get_guidance_multilingual('emotional_dysregulation', lang)
    
    return None

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
# MAIN ENDPOINT - COMPLETE FIX
# ============================================================

@app.post("/gating/ethical-route", response_model=GatingResponse)
async def ethical_routing(request: GatingRequest):
    """Process text through ethical growth framework"""
    
    logger.info(f"📝 Processing text for user {request.user_id}: {request.text[:50]}...")
    
    # Validate database_url
    if not request.database_url:
        raise HTTPException(status_code=400, detail="database_url is required")
    
    db_conn = psycopg2.connect(request.database_url)
    
    try:
        # 1. Detect language
        lang = detect_language(request.text)
        logger.info(f"🌍 Detected language: {lang}")
        
        # 2. ✅ Generate embedding FIRST
        logger.info(f"🧠 Generating embedding...")
        embedding = await generate_embedding(request.text)
        
        if not embedding:
            logger.warning("⚠️  Embedding generation failed, continuing without it")
        
        # 3. Get user history
        user_history = get_user_ethical_history(request.user_id, db_conn)
        
        # 4. Analyze ethical dimensions
        ethical_scores = analyze_ethical_dimensions_multilingual(request.text, user_history)
        
        # 5. Determine growth stage
        growth_stage = determine_growth_stage(ethical_scores)
        
        # 6. Detect moments
        moments = detect_moments_multilingual(request.text, ethical_scores)
        
        # 7. Generate guidance
        reflection_prompt = get_reflection_prompt_multilingual(growth_stage, lang)
        gentle_guidance = create_gentle_guidance_multilingual(moments, ethical_scores, lang)
        
        # 8. Classify
        classification = classify_for_learning(request.text, ethical_scores, moments, growth_stage)
        
        # 9. ✅ Save to memory_embeddings FIRST (with embedding)
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
        
        # 10. Save ethical profile
        save_ethical_profile(request.user_id, ethical_scores, growth_stage, db_conn)
        
        # 11. Save interaction memory (linked to memory_embeddings)
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
                'growth_area': min(ethical_scores, key=ethical_scores.get)
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
        "version": "2.0",
        "multilingual": True,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_url": OLLAMA_URL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
