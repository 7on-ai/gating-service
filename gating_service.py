#!/usr/bin/env python3
"""
🌍 Multilingual Ethical Growth Gating Service
Supports all languages with language-agnostic scoring
แทนที่: gating_service.py เดิม
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json

app = FastAPI(title="Ethical Growth Gating Service")

# ============================================================
# MULTILINGUAL CONFIGURATION
# ============================================================

class MultilingualPatterns:
    """Language-agnostic patterns using sentiment and semantic markers"""
    
    # Universal patterns (work across languages)
    SELF_REFLECTION = {
        'en': ['why', 'because', 'should i', 'is it right', 'wonder if', 'thinking about'],
        'th': ['ทำไม', 'เพราะอะไร', 'ควรไหม', 'ถูกหรือเปล่า', 'สงสัย', 'คิดว่า'],
        'zh': ['为什么', '因为', '应该', '是否', '想知道', '思考'],
        'ja': ['なぜ', 'どうして', 'すべき', '正しい', '考える', '思う'],
        'ko': ['왜', '때문에', '해야', '옳은', '생각', '궁금'],
        'es': ['por qué', 'porque', 'debería', 'correcto', 'me pregunto', 'pienso'],
        'fr': ['pourquoi', 'parce que', 'devrais', 'correct', 'je me demande', 'je pense'],
        'de': ['warum', 'weil', 'sollte', 'richtig', 'frage mich', 'denke'],
        'universal': ['?', '...', '🤔']  # Universal markers
    }
    
    EMPATHY = {
        'en': ['they feel', 'if i were', 'understand feeling', 'their perspective'],
        'th': ['เขารู้สึกยังไง', 'ถ้าเป็นเขา', 'เข้าใจความรู้สึก', 'มุมมองเขา'],
        'zh': ['他们感觉', '如果我是', '理解感受', '他们的角度'],
        'ja': ['彼らの気持ち', 'もし私が', '気持ちを理解', '彼らの視点'],
        'ko': ['그들이 느끼는', '내가 그들', '감정 이해', '그들의 관점'],
        'universal': ['❤️', '🫂', '💝']
    }
    
    ACCOUNTABILITY = {
        'en': ['my fault', 'responsible', 'apologize', 'fix', 'my mistake'],
        'th': ['ผิดของฉัน', 'รับผิดชอบ', 'ขอโทษ', 'แก้ไข', 'ความผิดพลาด'],
        'zh': ['我的错', '负责', '道歉', '改正', '我的错误'],
        'ja': ['私の過ち', '責任', '謝る', '直す', '私の間違い'],
        'ko': ['내 잘못', '책임', '사과', '고치다', '내 실수'],
        'universal': ['🙏', '🙇']
    }
    
    GRATITUDE = {
        'en': ['thank', 'grateful', 'appreciate', 'lucky', 'blessed'],
        'th': ['ขอบคุณ', 'ขอบใจ', 'ดีใจที่มี', 'โชคดีที่', 'มีความสุข'],
        'zh': ['谢谢', '感激', '感谢', '幸运', '感恩'],
        'ja': ['ありがとう', '感謝', '嬉しい', '幸運', '恵まれ'],
        'ko': ['감사', '고마워', '기쁘다', '행운', '축복'],
        'universal': ['🙏', '❤️', '😊', '💖']
    }
    
    GROWTH_SEEKING = {
        'en': ['learn', 'improve', 'develop', 'change', 'grow', 'better'],
        'th': ['เรียนรู้', 'พัฒนา', 'ทำให้ดีขึ้น', 'เปลี่ยนแปลง', 'เติบโต'],
        'zh': ['学习', '改进', '发展', '改变', '成长', '更好'],
        'ja': ['学ぶ', '改善', '発展', '変化', '成長', 'より良く'],
        'ko': ['배우다', '개선', '발전', '변화', '성장', '더 나은'],
        'universal': ['📚', '🌱', '💪', '⬆️']
    }
    
    # Concerning patterns (language-agnostic harm detection)
    AGGRESSION = {
        'en': ['kill', 'hurt', 'harm', 'attack', 'destroy', 'hate'],
        'th': ['ฆ่า', 'ทำร้าย', 'เจ็บ', 'โจมตี', 'ทำลาย', 'เกลียด'],
        'zh': ['杀', '伤害', '攻击', '破坏', '恨'],
        'ja': ['殺す', '傷つける', '攻撃', '破壊', '憎む'],
        'ko': ['죽이다', '해치다', '공격', '파괴', '미워하다'],
        'universal': ['🔪', '💀', '😡', '🤬']
    }
    
    DESPAIR = {
        'en': ['want to die', 'meaningless', 'worthless', 'hopeless', 'end it'],
        'th': ['อยากตาย', 'ไม่มีความหมาย', 'ไร้ค่า', 'สิ้นหวัง', 'จบชีวิต'],
        'zh': ['想死', '无意义', '无价值', '绝望', '结束'],
        'ja': ['死にたい', '無意味', '価値がない', '絶望', '終わり'],
        'ko': ['죽고 싶다', '무의미', '가치없다', '절망', '끝내다'],
        'universal': ['💔', '😭', '🖤']
    }

PATTERNS = MultilingualPatterns()

# ============================================================
# LANGUAGE DETECTION (Simple heuristic)
# ============================================================

def detect_language(text: str) -> str:
    """Simple language detection based on character sets"""
    
    # Check for specific character ranges
    if re.search(r'[\u0E00-\u0E7F]', text):  # Thai
        return 'th'
    elif re.search(r'[\u4E00-\u9FFF]', text):  # Chinese
        return 'zh'
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):  # Japanese
        return 'ja'
    elif re.search(r'[\uAC00-\uD7AF]', text):  # Korean
        return 'ko'
    elif re.search(r'[áéíóúñ¿¡]', text, re.IGNORECASE):  # Spanish
        return 'es'
    elif re.search(r'[àâçéèêëîïôùûü]', text, re.IGNORECASE):  # French
        return 'fr'
    elif re.search(r'[äöüß]', text, re.IGNORECASE):  # German
        return 'de'
    else:
        return 'en'  # Default to English

# ============================================================
# LANGUAGE-AGNOSTIC SCORING
# ============================================================

def score_pattern_multilingual(text: str, pattern_dict: Dict, lang: str = None) -> float:
    """
    Score text against multilingual patterns
    Returns 0-1 score
    """
    if not lang:
        lang = detect_language(text)
    
    text_lower = text.lower()
    
    # Get patterns for detected language + universal
    patterns = pattern_dict.get(lang, []) + pattern_dict.get('universal', [])
    
    if not patterns:
        # Fallback to English if language not supported
        patterns = pattern_dict.get('en', []) + pattern_dict.get('universal', [])
    
    matches = sum(1 for pattern in patterns if pattern in text_lower)
    
    if len(patterns) == 0:
        return 0.0
    
    return min(matches / max(len(patterns) * 0.3, 1), 1.0)

def analyze_ethical_dimensions_multilingual(text: str, user_history: Dict) -> Dict[str, float]:
    """
    Language-agnostic ethical dimension analysis
    """
    lang = detect_language(text)
    
    scores = {}
    
    # 1. Self-awareness
    self_aware_score = score_pattern_multilingual(text, PATTERNS.SELF_REFLECTION, lang)
    scores['self_awareness'] = min(1.0, 
        self_aware_score * 0.7 + user_history.get('baseline_self_awareness', 0.3)
    )
    
    # 2. Emotional regulation (inverse of aggression)
    aggression_score = score_pattern_multilingual(text, PATTERNS.AGGRESSION, lang)
    scores['emotional_regulation'] = max(0.0, min(1.0,
        (1.0 - aggression_score) * 0.7 + user_history.get('baseline_regulation', 0.4)
    ))
    
    # 3. Compassion
    empathy_score = score_pattern_multilingual(text, PATTERNS.EMPATHY, lang)
    scores['compassion'] = min(1.0,
        empathy_score * 0.7 + user_history.get('baseline_compassion', 0.4)
    )
    
    # 4. Integrity
    accountability_score = score_pattern_multilingual(text, PATTERNS.ACCOUNTABILITY, lang)
    scores['integrity'] = min(1.0,
        accountability_score * 0.7 + user_history.get('baseline_integrity', 0.5)
    )
    
    # 5. Growth mindset
    growth_score = score_pattern_multilingual(text, PATTERNS.GROWTH_SEEKING, lang)
    scores['growth_mindset'] = min(1.0,
        growth_score * 0.7 + user_history.get('baseline_growth', 0.4)
    )
    
    # 6. Wisdom (combination of reflection + empathy)
    wisdom_score = (self_aware_score + empathy_score) / 2
    scores['wisdom'] = min(1.0,
        wisdom_score * 0.6 + user_history.get('baseline_wisdom', 0.3)
    )
    
    # 7. Transcendence (gratitude + growth)
    gratitude_score = score_pattern_multilingual(text, PATTERNS.GRATITUDE, lang)
    transcendent_score = (gratitude_score + growth_score) / 2
    scores['transcendence'] = min(1.0,
        transcendent_score * 0.5 + user_history.get('baseline_transcendence', 0.2)
    )
    
    return scores

def detect_moments_multilingual(text: str, ethical_scores: Dict) -> List[Dict]:
    """
    Language-agnostic moment detection
    """
    lang = detect_language(text)
    moments = []
    
    # Breakthrough moment
    reflection_score = score_pattern_multilingual(text, PATTERNS.SELF_REFLECTION, lang)
    if reflection_score > 0.6:
        moments.append({
            'type': 'breakthrough',
            'severity': 'positive',
            'description': 'User shows self-reflection',
            'timestamp': datetime.now().isoformat()
        })
    
    # Struggle moment
    if ethical_scores.get('emotional_regulation', 0.5) < 0.3:
        moments.append({
            'type': 'struggle',
            'severity': 'neutral',
            'description': 'User experiencing difficulty',
            'timestamp': datetime.now().isoformat()
        })
    
    # Crisis moment
    despair_score = score_pattern_multilingual(text, PATTERNS.DESPAIR, lang)
    if despair_score > 0.5:
        moments.append({
            'type': 'crisis',
            'severity': 'critical',
            'description': 'User in emotional crisis',
            'timestamp': datetime.now().isoformat(),
            'requires_intervention': True
        })
    
    # Growth moment
    growth_score = score_pattern_multilingual(text, PATTERNS.GROWTH_SEEKING, lang)
    if growth_score > 0.6:
        moments.append({
            'type': 'growth',
            'severity': 'positive',
            'description': 'User showing growth mindset',
            'timestamp': datetime.now().isoformat()
        })
    
    return moments

# ============================================================
# MULTILINGUAL GUIDANCE TEMPLATES
# ============================================================

GUIDANCE_TEMPLATES = {
    'crisis': {
        'en': "I'm concerned about you. Please reach out to a mental health professional. Crisis hotline: [LOCAL NUMBER]",
        'th': "ฉันเป็นห่วงคุณมาก โปรดติดต่อสายด่วนสุขภาพจิต 1323 หรือพูดคุยกับคนที่คุณไว้วางใจ",
        'zh': "我很担心你。请联系心理健康专业人士。危机热线：[当地号码]",
        'ja': "あなたのことが心配です。メンタルヘルスの専門家に連絡してください。",
        'ko': "걱정됩니다. 정신건강 전문가에게 연락하세요. 위기 상담전화: [지역 번호]",
        'es': "Me preocupo por ti. Por favor contacta a un profesional de salud mental.",
        'fr': "Je m'inquiète pour vous. Veuillez contacter un professionnel de la santé mentale.",
        'de': "Ich mache mir Sorgen um dich. Bitte kontaktiere einen Psychologen.",
    },
    'emotional_dysregulation': {
        'en': "Take a deep breath. These feelings will pass, and you'll see things more clearly.",
        'th': "ลองหายใจเข้าลึกๆ ค่อยๆ ปล่อยออก ความรู้สึกนี้จะผ่านไป แล้วคุณจะเห็นภาพชัดขึ้น",
        'zh': "深呼吸。这些感觉会过去的，你会看得更清楚。",
        'ja': "深呼吸してください。この感情は過ぎ去り、もっと明確に見えるようになります。",
        'ko': "심호흡을 하세요. 이 감정은 지나갈 것이고 더 명확하게 보일 것입니다.",
        'es': "Respira profundo. Estos sentimientos pasarán y verás las cosas con más claridad.",
        'fr': "Respirez profondément. Ces sentiments passeront et vous verrez plus clair.",
        'de': "Atme tief durch. Diese Gefühle werden vergehen und du wirst klarer sehen.",
    },
    'empathy_encouragement': {
        'en': "Perhaps seeing from their perspective might help. They may be facing something we don't know about.",
        'th': "บางทีการมองจากมุมของอีกฝ่ายอาจช่วยได้ เขาอาจกำลังเผชิญอะไรที่เราไม่รู้ก็ได้",
        'zh': "也许从他们的角度看会有帮助。他们可能面临着我们不知道的事情。",
        'ja': "彼らの視点から見ることが役立つかもしれません。彼らは私たちが知らない何かに直面しているかもしれません。",
        'ko': "그들의 관점에서 보는 것이 도움이 될 수 있습니다. 그들은 우리가 모르는 무언가를 겪고 있을 수 있습니다.",
        'es': "Quizás ver desde su perspectiva ayude. Pueden estar enfrentando algo que no sabemos.",
        'fr': "Peut-être que voir de leur perspective aiderait. Ils font peut-être face à quelque chose que nous ne savons pas.",
        'de': "Vielleicht hilft es, aus ihrer Perspektive zu sehen. Sie könnten mit etwas konfrontiert sein, von dem wir nichts wissen.",
    }
}

REFLECTION_PROMPTS = {
    1: {  # Pre-conventional
        'en': "What are you feeling right now? Can you tell me?",
        'th': "สิ่งที่คุณกำลังรู้สึกตอนนี้คืออะไร? ลองบอกฉันได้ไหม",
        'zh': "你现在感觉如何？能告诉我吗？",
        'ja': "今どう感じていますか？教えてもらえますか？",
        'ko': "지금 어떤 기분이세요? 말해줄 수 있나요?",
    },
    2: {  # Conventional
        'en': "If someone else were in this situation, how would they feel?",
        'th': "ถ้าคนอื่นอยู่ในสถานการณ์นี้ คุณคิดว่าเขาจะรู้สึกยังไง?",
        'zh': "如果其他人处于这种情况，他们会有什么感受？",
        'ja': "もし他の誰かがこの状況にいたら、どう感じると思いますか？",
        'ko': "다른 사람이 이 상황에 있다면 어떻게 느낄까요?",
    },
    3: {  # Post-conventional
        'en': "What values does this decision reflect?",
        'th': "การตัดสินใจนี้สะท้อนคุณค่าอะไรของคุณ?",
        'zh': "这个决定反映了什么价值观？",
        'ja': "この決定はどんな価値観を反映していますか？",
        'ko': "이 결정은 어떤 가치를 반영하나요?",
    },
    4: {  # Integrated
        'en': "Does this action align with the person you want to become?",
        'th': "สิ่งนี้ช่วยให้คุณเป็นคนที่คุณอยากเป็นไหม?",
        'zh': "这个行动是否符合你想成为的人？",
        'ja': "この行動はあなたがなりたい人と一致していますか？",
        'ko': "이 행동이 당신이 되고 싶은 사람과 일치하나요?",
    },
    5: {  # Transcendent
        'en': "How can this action create goodness for the world?",
        'th': "การกระทำนี้สร้างความดีงามให้โลกได้อย่างไร?",
        'zh': "这个行动如何为世界创造善意？",
        'ja': "この行動はどのように世界に善をもたらすことができますか？",
        'ko': "이 행동이 세상에 어떻게 선을 만들 수 있나요?",
    }
}

def get_guidance_multilingual(template_key: str, lang: str) -> str:
    """Get guidance in user's language"""
    templates = GUIDANCE_TEMPLATES.get(template_key, {})
    return templates.get(lang, templates.get('en', ''))

def get_reflection_prompt_multilingual(stage: int, lang: str) -> str:
    """Get reflection prompt in user's language"""
    prompts = REFLECTION_PROMPTS.get(stage, REFLECTION_PROMPTS[2])
    return prompts.get(lang, prompts.get('en', ''))

# ============================================================
# REST OF THE CODE (Same as before but with multilingual support)
# ============================================================

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

def create_gentle_guidance_multilingual(moments: List[Dict], ethical_scores: Dict, lang: str) -> Optional[str]:
    crisis_moments = [m for m in moments if m.get('severity') == 'critical']
    
    if crisis_moments:
        return get_guidance_multilingual('crisis', lang)
    
    if ethical_scores.get('emotional_regulation', 0.5) < 0.3:
        return get_guidance_multilingual('emotional_dysregulation', lang)
    
    if ethical_scores.get('compassion', 0.5) < 0.4:
        return get_guidance_multilingual('empathy_encouragement', lang)
    
    return None

def classify_for_learning(
    text: str, 
    ethical_scores: Dict, 
    moments: List[Dict],
    stage: int
) -> str:
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

def save_interaction_memory(
    user_id: str, 
    text: str, 
    classification: str,
    ethical_scores: Dict,
    moments: List[Dict],
    reflection_prompt: str,
    gentle_guidance: Optional[str],
    db_conn
):
    cursor = db_conn.cursor()
    
    cursor.execute("""
        INSERT INTO user_data_schema.interaction_memories
        (user_id, text, classification, ethical_scores, moments, 
         reflection_prompt, gentle_guidance, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
    """, (
        user_id,
        text,
        classification,
        json.dumps(ethical_scores),
        json.dumps(moments),
        reflection_prompt,
        gentle_guidance,
        json.dumps({'source': 'gating_service'})
    ))
    
    db_conn.commit()
    cursor.close()

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

# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.post("/gating/ethical-route", response_model=GatingResponse)
async def ethical_routing(request: GatingRequest):
    db_conn = psycopg2.connect(request.database_url)
    
    try:
        # Detect language
        lang = detect_language(request.text)
        
        # Get user history
        user_history = get_user_ethical_history(request.user_id, db_conn)
        
        # Analyze ethical dimensions (multilingual)
        ethical_scores = analyze_ethical_dimensions_multilingual(request.text, user_history)
        
        # Determine growth stage
        growth_stage = determine_growth_stage(ethical_scores)
        
        # Detect moments (multilingual)
        moments = detect_moments_multilingual(request.text, ethical_scores)
        
        # Generate reflection prompt (in user's language)
        reflection_prompt = get_reflection_prompt_multilingual(growth_stage, lang)
        
        # Generate gentle guidance (in user's language)
        gentle_guidance = create_gentle_guidance_multilingual(moments, ethical_scores, lang)
        
        # Classify for learning
        classification = classify_for_learning(
            request.text, ethical_scores, moments, growth_stage
        )
        
        # Save to database
        save_ethical_profile(request.user_id, ethical_scores, growth_stage, db_conn)
        
        save_interaction_memory(
            request.user_id,
            request.text,
            classification,
            ethical_scores,
            moments,
            reflection_prompt,
            gentle_guidance,
            db_conn
        )
        
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
            growth_opportunity=f"Stage {growth_stage}/5 - Focus on {min(ethical_scores, key=ethical_scores.get)}",
            detected_language=lang
        )
        
    finally:
        db_conn.close()

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "ethical_growth_gating",
        "multilingual": True,
        "supported_languages": ["en", "th", "zh", "ja", "ko", "es", "fr", "de"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
