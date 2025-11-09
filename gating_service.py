#!/usr/bin/env python3
"""
Complete Gating Service with Two-Channel + MCL
✅ Fixed: Shared service but per-user database
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json

app = FastAPI(title="Gating Service")

CONFIG = {
    "alignment_threshold": 0.5,
    "severity_threshold": 0.3,
    "toxicity_threshold": 0.2,
    "mcl_chain_window": 300,
}

PII_PATTERNS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
}

RED_LINES = {
    'violence': ['kill', 'murder', 'hurt', 'harm', 'attack', 'shoot', 'stab', 'ฆ่า', 'ทำร้าย'],
    'illegal': ['hack', 'steal', 'fraud', 'scam', 'counterfeit', 'rob', 'แฮก', 'ขโมย'],
    'hate': ['racist', 'sexist', 'homophobic', 'discriminate', 'slur', 'เหยียด'],
    'manipulation': ['manipulate', 'gaslight', 'deceive', 'trick', 'exploit', 'หลอกลวง'],
    'self_harm': ['suicide', 'kill myself', 'end my life', 'self-harm', 'ฆ่าตัวตาย'],
}

MORAL_TAXONOMY = {
    'benevolent_transgression': 'ทำผิดเพื่อผู้อื่น',
    'malevolent_altruism': 'ทำดีเพื่อจุดประสงค์ชั่ว',
    'necessary_harm': 'ทำร้ายเพื่อป้องกัน',
    'hidden_bad_intent': 'ดูเหมือนดีแต่เจตนาร้าย',
    'mixed_moral_chain': 'หลายaction บิดเบี้ยว',
    'neutral_contextual': 'ไม่มี moral weight ชัด',
}

class GatingRequest(BaseModel):
    user_id: str
    text: str
    database_url: str  # ✅ CRITICAL: ต้องส่งมาทุก request
    session_id: Optional[str] = None
    metadata: Optional[Dict] = {}

class GatingResponse(BaseModel):
    status: str
    routing: str
    valence: str
    scores: Dict
    mcl_entry: Optional[Dict] = None
    safe_counterfactual: Optional[str] = None

def sanitize(text: str) -> str:
    for name, pattern in PII_PATTERNS.items():
        text = pattern.sub(f'[REDACTED_{name.upper()}]', text)
    return text

def check_red_lines(text: str) -> List[str]:
    text_lower = text.lower()
    triggered = []
    for category, keywords in RED_LINES.items():
        for keyword in keywords:
            if keyword in text_lower:
                triggered.append(category)
                break
    return list(set(triggered))

def simple_valence_classifier(text: str) -> Dict[str, float]:
    text_lower = text.lower()
    
    positive_words = ['ขอบคุณ', 'ดี', 'ช่วย', 'รัก', 'สุข', 'good', 'great', 'thank', 'love', 'help', 'wonderful', 'excellent']
    negative_words = ['เกลียด', 'แย่', 'เจ็บ', 'ทำร้าย', 'hate', 'bad', 'hurt', 'harm', 'kill', 'terrible', 'awful']
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
    
    return {
        'positive': pos_count / total if total > 0 else 0.33,
        'negative': neg_count / total if total > 0 else 0.33,
        'neutral': 1.0 - (pos_count + neg_count) / max(total, 1)
    }

def simple_toxicity_score(text: str, red_line_triggers: List[str]) -> float:
    if not red_line_triggers:
        return 0.0
    
    severity_weights = {
        'violence': 0.9,
        'self_harm': 1.0,
        'illegal': 0.7,
        'hate': 0.8,
        'manipulation': 0.5,
    }
    
    max_severity = max([severity_weights.get(t, 0.5) for t in red_line_triggers], default=0.0)
    return max_severity

def calculate_alignment(valence: Dict, toxicity: float, quality: float = 0.7) -> float:
    return (
        0.5 * valence['positive'] +
        0.3 * quality -
        0.7 * toxicity
    )

def detect_chain(user_id: str, session_id: str, db_conn) -> Optional[List[Dict]]:
    if not session_id:
        return None
    
    cursor = db_conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        query = """
        SELECT text, created_at
        FROM user_data_schema.gating_logs
        WHERE user_id = %s 
          AND created_at > NOW() - INTERVAL '%s seconds'
        ORDER BY created_at DESC
        LIMIT 5
        """
        
        cursor.execute(query, (user_id, CONFIG['mcl_chain_window']))
        messages = cursor.fetchall()
        
        if len(messages) < 2:
            return None
        
        return [{'step': i+1, 'text': m['text'], 'timestamp': m['created_at'].isoformat()}
                for i, m in enumerate(reversed(messages))]
    finally:
        cursor.close()

def infer_intention(chain: List[Dict]) -> float:
    last_text = chain[-1]['text'].lower()
    positive_indicators = ['ช่วย', 'แม่', 'พ่อ', 'ครอบครัว', 'help', 'family', 'save', 'protect']
    
    score = 0.5
    for indicator in positive_indicators:
        if indicator in last_text:
            score += 0.1
    
    return min(score, 1.0)

def estimate_necessity(chain: List[Dict]) -> float:
    necessity_words = ['จำเป็น', 'ฉุกเฉิน', 'ป่วย', 'หิว', 'emergency', 'necessary', 'sick', 'hungry']
    
    text = ' '.join([c['text'].lower() for c in chain])
    score = 0.3
    
    for word in necessity_words:
        if word in text:
            score += 0.15
    
    return min(score, 1.0)

def classify_moral_chain(chain: List[Dict], intention: float, necessity: float, 
                        harm: float, benefit: float) -> str:
    if intention > 0.7 and harm > 0.3 and benefit > 0.5:
        return 'benevolent_transgression'
    elif intention < 0.3 and benefit > 0.5:
        return 'malevolent_altruism'
    elif necessity > 0.7 and harm > 0.5:
        return 'necessary_harm'
    elif intention < 0.5 and harm > 0.3:
        return 'hidden_bad_intent'
    elif abs(harm - benefit) < 0.2:
        return 'mixed_moral_chain'
    else:
        return 'neutral_contextual'

def generate_safe_counterfactual(text: str, shadow_tag: str) -> str:
    templates = {
        'violence': 'ฉันไม่สามารถให้คำแนะนำที่เกี่ยวกับความรุนแรงได้ หากคุณต้องการความช่วยเหลือ โปรดติดต่อหน่วยงานที่เกี่ยวข้อง',
        'illegal': 'ฉันไม่สามารถช่วยเหลือในกิจกรรมที่ผิดกฎหมายได้ แนะนำให้ปรึกษาผู้เชี่ยวชาญทางกฎหมาย',
        'self_harm': 'ฉันเป็นห่วงคุณมาก โปรดติดต่อสายด่วนสุขภาพจิต 1323 หรือพูดคุยกับคนที่คุณไว้ใจ',
        'hate': 'ฉันไม่สร้างเนื้อหาที่เหยียดหยามหรือเลือกปฏิบัติใดๆ ทุกคนสมควรได้รับความเคารพ',
        'manipulation': 'ฉันส่งเสริมการสื่อสารที่ตรงไปตรงมาและให้เกียรติ การบิดเบือนทำร้ายความไว้วางใจ',
    }
    
    return templates.get(shadow_tag, 'ขออภัย ฉันไม่สามารถช่วยเหลือในเรื่องนี้ได้')

@app.post("/gating/route", response_model=GatingResponse)
async def route_message(request: GatingRequest):
    """
    ✅ Fixed: ใช้ database_url ที่ส่งมาแต่ละ request
    Shared service แต่แยก database per-user
    """
    
    # ✅ เชื่อมต่อ DB ของ user นั้น ๆ
    db_conn = psycopg2.connect(request.database_url)
    
    try:
        clean_text = sanitize(request.text)
        red_line_triggers = check_red_lines(clean_text)
        valence_scores = simple_valence_classifier(clean_text)
        valence = max(valence_scores, key=valence_scores.get)
        toxicity = simple_toxicity_score(clean_text, red_line_triggers)
        alignment = calculate_alignment(valence_scores, toxicity)
        severity = valence_scores['negative'] * toxicity
        
        chain = detect_chain(request.user_id, request.session_id, db_conn)
        mcl_entry = None
        
        if chain:
            chain.append({'step': len(chain) + 1, 'text': clean_text, 'timestamp': datetime.now().isoformat()})
            
            intention = infer_intention(chain)
            necessity = estimate_necessity(chain)
            harm = toxicity
            benefit = valence_scores['positive']
            
            moral_class = classify_moral_chain(chain, intention, necessity, harm, benefit)
            
            mcl_entry = {
                'user_id': request.user_id,
                'event_chain': chain,
                'intention_score': intention,
                'necessity_score': necessity,
                'harm_score': harm,
                'benefit_score': benefit,
                'moral_classification': moral_class,
                'summary': f"Chain of {len(chain)} events classified as {moral_class}"
            }
            
            cursor = db_conn.cursor()
            cursor.execute("""
                INSERT INTO user_data_schema.mcl_chains 
                (user_id, event_chain, intention_score, necessity_score, harm_score, benefit_score, moral_classification, summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                mcl_entry['user_id'],
                json.dumps(mcl_entry['event_chain']),
                mcl_entry['intention_score'],
                mcl_entry['necessity_score'],
                mcl_entry['harm_score'],
                mcl_entry['benefit_score'],
                mcl_entry['moral_classification'],
                mcl_entry['summary']
            ))
            db_conn.commit()
            cursor.close()
        
        routing = None
        safe_counterfactual = None
        
        if valence == 'positive' and alignment >= CONFIG['alignment_threshold']:
            routing = 'good'
            cursor = db_conn.cursor()
            cursor.execute("""
                INSERT INTO user_data_schema.stm_good 
                (user_id, text, valence, alignment_score, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (request.user_id, clean_text, valence, alignment, json.dumps(request.metadata or {})))
            db_conn.commit()
            cursor.close()
            
        elif valence == 'negative' and severity >= CONFIG['severity_threshold']:
            routing = 'bad'
            shadow_tag = red_line_triggers[0] if red_line_triggers else 'general_negative'
            safe_counterfactual = generate_safe_counterfactual(clean_text, shadow_tag)
            
            cursor = db_conn.cursor()
            cursor.execute("""
                INSERT INTO user_data_schema.stm_bad 
                (user_id, text, valence, severity_score, toxicity_score, shadow_tag, safe_counterfactual, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (request.user_id, clean_text, valence, severity, toxicity, shadow_tag, safe_counterfactual, json.dumps(request.metadata or {})))
            db_conn.commit()
            cursor.close()
            
        else:
            routing = 'review'
            cursor = db_conn.cursor()
            cursor.execute("""
                INSERT INTO user_data_schema.stm_review 
                (user_id, text, gating_reason, metadata)
                VALUES (%s, %s, %s, %s)
            """, (request.user_id, clean_text, f"Ambiguous: valence={valence}, alignment={alignment:.2f}", json.dumps(request.metadata or {})))
            db_conn.commit()
            cursor.close()
        
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO user_data_schema.gating_logs 
            (user_id, input_text, routing_decision, valence_scores, toxicity_score, rules_triggered, mcl_detected)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            request.user_id,
            clean_text,
            routing,
            json.dumps(valence_scores),
            toxicity,
            red_line_triggers,
            chain is not None
        ))
        db_conn.commit()
        cursor.close()
        
        return GatingResponse(
            status='success',
            routing=routing,
            valence=valence,
            scores={
                'valence': valence_scores,
                'alignment': alignment,
                'toxicity': toxicity,
                'severity': severity
            },
            mcl_entry=mcl_entry,
            safe_counterfactual=safe_counterfactual
        )
        
    finally:
        db_conn.close()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "gating", "shared": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
