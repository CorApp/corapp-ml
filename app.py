"""
CorApp ML — v5.0.0 COMPLETO
Sistema de inteligencia artificial para ventas por WhatsApp en español colombiano.

Arquitectura:
- Clasificador de intenciones con TF-IDF + LinearSVC
- Extractor de datos de entrega ultra-robusto
- Tres algoritmos de similitud combinados (Levenshtein + Jaro + Jaro-Winkler)
- Aliases y abreviaciones del español colombiano coloquial
- Respuestas naturales variadas por intención
- Manejo de frustración y empatía
- Análisis de fallos para mejora continua
- Siempre responde — nunca silencio

Basado en:
- 400+ conversaciones reales de CorApp
- Patrones del español colombiano coloquial
- Errores tipográficos y abreviaciones comunes
- Modismos y expresiones locales
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import json
import re
import unicodedata
import random
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from telegram_service import create_tenant_group
from telethon.errors import FloodWaitError

app = Flask(__name__)
model = None


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize(text: str) -> str:
    """Normaliza texto — minúsculas, sin tildes, sin espacios extra, sin
    puntuación de frase (?!¿¡.,;:). Sin esto, "mango?" nunca era igual a
    "mango" en comparaciones exactas — bug real detectado en producción:
    "Tienes mango?" no reconocía el producto porque la palabra extraída
    era literalmente "mango?", no "mango". No afecta a extract_address_
    indications() ni clean_noise(), que capturan el valor final por fuera
    de esta función y sí preservan # y - para direcciones."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r'[?!¿¡.,;:]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================
# ALGORITMOS DE SIMILITUD
# ============================================================

def levenshtein(a: str, b: str) -> float:
    """Distancia de edición normalizada — buena para palabras largas."""
    a, b = normalize(a), normalize(b)
    if a == b: return 1.0
    if not a or not b: return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return 1.0 - dp[n] / max(m, n)


def jaro(a: str, b: str) -> float:
    """Similitud Jaro — excelente para palabras cortas."""
    a, b = normalize(a), normalize(b)
    if a == b: return 1.0
    if not a or not b: return 0.0
    md = max(len(a), len(b)) // 2 - 1
    if md < 0: md = 0
    am = [False] * len(a)
    bm = [False] * len(b)
    matches = trans = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - md), min(i + md + 1, len(b))):
            if bm[j] or ca != b[j]: continue
            am[i] = bm[j] = True
            matches += 1
            break
    if not matches: return 0.0
    k = 0
    for i in range(len(a)):
        if not am[i]: continue
        while not bm[k]: k += 1
        if a[i] != b[k]: trans += 1
        k += 1
    return (matches/len(a) + matches/len(b) + (matches - trans/2)/matches) / 3


def jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler — mejor para palabras con prefijo común."""
    j = jaro(a, b)
    an, bn = normalize(a), normalize(b)
    p = sum(1 for i in range(min(4, len(an), len(bn))) if an[i] == bn[i])
    return j + p * 0.1 * (1 - j)


def similarity(a: str, b: str) -> float:
    """
    Combina los tres algoritmos.
    Para palabras cortas Jaro-Winkler es mejor.
    Para palabras largas Levenshtein es más preciso.
    """
    la = len(normalize(a))
    lev = levenshtein(a, b)
    jw = jaro_winkler(a, b)
    if la <= 5:
        return max(lev, jw)
    return lev * 0.55 + jw * 0.45


# ============================================================
# ML — CLASIFICADOR DE INTENCIONES
# ============================================================

def train_model():
    with open('data/training.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences, labels = [], []
    for intent in data['intents']:
        for example in intent['examples']:
            sentences.append(normalize(example))
            labels.append(intent['tag'])
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            max_features=50000,
        )),
        ('clf', LinearSVC(max_iter=3000, C=1.2, class_weight='balanced'))
    ])
    pipeline.fit(sentences, labels)
    os.makedirs('model', exist_ok=True)
    joblib.dump(pipeline, 'model/intent_classifier.pkl')
    return pipeline


def load_model():
    global model
    if model is None:
        model = train_model() if not os.path.exists('model/intent_classifier.pkl') \
            else joblib.load('model/intent_classifier.pkl')
    return model


def classify(text: str) -> tuple:
    clf = load_model()
    norm_text = normalize(text)
    intent = clf.predict([norm_text])[0]
    scores = clf.decision_function([norm_text])[0]
    exp = np.exp(scores - np.max(scores))
    confidence = float(np.max(exp) / exp.sum())
    return intent, confidence


# ============================================================
# DATOS DE REFERENCIA — DÍAS
# ============================================================

VALID_DAYS = {
    "lunes": "Lunes", "martes": "Martes",
    "miercoles": "Miércoles", "miércoles": "Miércoles",
    "jueves": "Jueves", "viernes": "Viernes",
    "sabado": "Sábado", "sábado": "Sábado",
}

DAY_ALIASES = {
    "lun": "Lunes", "lns": "Lunes", "lnes": "Lunes", "lun.": "Lunes",
    "el lunes": "Lunes", "este lunes": "Lunes",
    "mar": "Martes", "mrt": "Martes", "mrts": "Martes",
    "el martes": "Martes", "este martes": "Martes",
    "mie": "Miércoles", "mier": "Miércoles", "mirc": "Miércoles",
    "mierc": "Miércoles", "miercole": "Miércoles", "miercols": "Miércoles",
    "merco": "Miércoles", "el miercoles": "Miércoles",
    "jue": "Jueves", "jues": "Jueves", "jvs": "Jueves", "jves": "Jueves",
    "juev": "Jueves", "juevs": "Jueves", "jve": "Jueves", "jv": "Jueves",
    "el jueves": "Jueves",
    "vie": "Viernes", "vies": "Viernes", "vrs": "Viernes",
    "viern": "Viernes", "vierens": "Viernes", "el viernes": "Viernes",
    "sab": "Sábado", "sabs": "Sábado", "sbd": "Sábado",
    "sabdo": "Sábado", "sbdo": "Sábado", "sab.": "Sábado",
    "el sabado": "Sábado", "el sábado": "Sábado",
}

INVALID_DAYS = {
    "domingo": "Los domingos no hacemos entregas 😊 Puedes elegir entre Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "manana": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "mañana": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "hoy": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "pasado manana": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "pasado mañana": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "entre semana": "¿Cuál día entre semana prefieres? Lunes, Martes, Miércoles, Jueves o Viernes",
    "fin de semana": "El único día de fin de semana disponible es el Sábado 😊",
    "lo antes posible": "Con gusto 😊 ¿Cuál día prefieres? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "lo mas pronto": "Con gusto 😊 ¿Cuál día prefieres? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "cualquier dia": "¿Cuál día te queda mejor? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "cuando puedan": "¿Cuál día te queda mejor? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "pronto": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
}


# ============================================================
# DATOS DE REFERENCIA — LOCALIDADES
# ============================================================

VALID_LOCATIONS = {
    "bosa": "Bosa", "kennedy": "Kennedy", "puente aranda": "Puente aranda",
    "tunjuelito": "Tunjuelito", "antonio narino": "Antonio narino",
    "antonio nariño": "Antonio narino", "teusaquillo": "Teusaquillo",
    "barrios unidos": "Barrios unidos", "martires": "Martirez",
    "mártires": "Martirez", "martirez": "Martirez",
    "fontibon": "Fontibon", "fontibón": "Fontibon",
    "engativa": "Engativa", "engativá": "Engativa",
    "chapinero": "Chapinero", "usaquen": "Usaquen", "usaquén": "Usaquen",
    "soacha": "Soacha", "candelaria": "Candelaria", "suba": "Suba",
    "rafael uribe": "Rafael Uribe Uribe", "rafael uribe uribe": "Rafael Uribe Uribe",
    "ciudad bolivar": "Ciudad Bolivar", "ciudad bolívar": "Ciudad Bolivar",
    "san cristobal": "San Cristobal", "san cristóbal": "San Cristobal",
    "usme": "Usme", "sumapaz": "Sumapaz",
    "modelia": "Fontibon", "capellania": "Fontibon", "capellanía": "Fontibon",
    "fontibón sur": "Fontibon", "patio bonito": "Kennedy",
    "ciudad montes": "Puente aranda", "tibabuyes": "Suba",
    "tibabuyes universal": "Suba", "pinar": "Suba", "pinar de suba": "Suba",
    "porvenir": "Bosa", "bosa piamonte": "Bosa", "bosa libertad": "Bosa",
    "bosa nueva": "Bosa", "bosa el porvenir": "Bosa",
    "senderos del porvenir": "Bosa", "la libertad bosa": "Bosa",
    "alqueria": "Kennedy", "alquería": "Kennedy",
    "alqueria de la fragua": "Kennedy", "alquería de la fragua": "Kennedy",
    "prado veraniego": "Suba", "prado pinzon": "Suba", "prado pinzón": "Suba",
    "ciudad kennedy": "Kennedy", "cedro": "Engativa", "el cedro": "Engativa",
    "alamos": "Engativa", "álamos": "Engativa", "portales": "Engativa",
    "portales norte": "Engativa", "san agustin": "Kennedy", "san agustín": "Kennedy",
    "corabastos": "Kennedy", "zona industrial kennedy": "Kennedy",
    "castellon de los condes": "Kennedy", "castellón de los condes": "Kennedy",
    "cra 87b": "Kennedy", "antiguo country": "Chapinero", "country": "Chapinero",
    "rosales": "Chapinero", "portal de rosales": "Chapinero",
    "chapinero alto": "Chapinero", "chapinero norte": "Chapinero",
    "gran estacion": "Teusaquillo", "gran estación": "Teusaquillo",
    "palermo": "Teusaquillo", "la soledad": "Teusaquillo", "armenia": "Teusaquillo",
    "bahia solano": "Fontibon", "bahía solano": "Fontibon",
    "ciudad salitre": "Fontibon", "salitre": "Fontibon", "la giralda": "Fontibon",
    "villa del prado": "Suba", "alhambra": "Suba", "cedritos": "Usaquen",
    "santa barbara": "Usaquen", "santa bárbara": "Usaquen",
    "mazuren": "Suba", "mazurén": "Suba", "niza": "Suba",
    "bello horizonte": "Suba", "verbenal": "Usaquen",
    "toberin": "Usaquen", "toberín": "Usaquen", "country norte": "Usaquen",
    "santa cecilia": "Engativa", "villa luz": "Engativa",
    "gaitan": "Barrios unidos", "gaitán": "Barrios unidos",
    "alcazares": "Barrios unidos", "siete de agosto": "Barrios unidos",
    "la floresta": "Engativa", "floresta": "Engativa",
    "quirigua": "Engativa", "quiriguá": "Engativa",
    "minuto de dios": "Engativa", "bachue": "Engativa",
    "tintal": "Kennedy", "americas": "Kennedy", "américas": "Kennedy",
    "timiza": "Kennedy", "muzú": "Puente aranda", "muzu": "Puente aranda",
    "la esperanza": "Kennedy", "candelaria la nueva": "Kennedy",
    "cundinamarca": None, "zipaquira": None, "zipaquirá": None,
    "chia": None, "chía": None, "sopo": None, "sopó": None,
    "cajica": None, "cajicá": None, "mosquera": None,
    "madrid cundinamarca": None, "facatativa": None, "facatativá": None,
    "funza": None, "tocancipa": None, "tocancipá": None,
    "la calera": None, "cota": None, "sibate": None, "sibaté": None,
}


# ============================================================
# LIMPIEZA DE RUIDO
# ============================================================

NOISE_PATTERNS = [
    r'[\w\.-]+@[\w\.-]+\.\w+',
    r'\b3\d{9}\b',
    r'\b\d{7,10}\b',
    r'\bpago\s+(?:contra\s+entrega|en\s+efectivo|nequi|bre-?b|transferencia|electronica)\b',
    r'\btelefono\s*[:*]?\s*[\d\s\-]+',
    r'\bcel(?:ular)?\s*[:*]?\s*[\d\s\-]+',
    r'\bcorreo\s*[:*]?\s*\S+',
    r'\bhoras?\s+de\s+la\s+ma[nñ]ana\b',
    r'\bhorario\s*[:*]?\s*[\d\sapm:]+',
    r'\bcontacto\s*[:*]?\s*[\d\s\-]+',
]


def clean_noise(text: str) -> str:
    for p in NOISE_PATTERNS:
        text = re.sub(p, ' ', text, flags=re.IGNORECASE)
    # IMPORTANTE: colapsar solo espacios/tabs horizontales, NUNCA saltos de
    # línea. extract_labeled() y extract_address_indications() dependen de
    # los \n reales para saber dónde termina cada campo (dirección, barrio,
    # localidad, día). Si se colapsan aquí, un mensaje multilínea se vuelve
    # una sola línea larga y la dirección "se come" todo lo que sigue
    # (localidad, día) porque el regex de respaldo es greedy.
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)  # colapsar líneas vacías múltiples
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


# ============================================================
# EXTRACCIÓN DE CAMPOS CON ETIQUETAS
# ============================================================

def extract_labeled(text: str) -> dict:
    fields = {}
    patterns = {
        'name':        r'(?:\*?\s*nombre(?:\s+completo)?\s*\*?)\s*[:*\-]?\s*(.+)',
        'address':     r'(?:\*?\s*direcci[oó]n(?:\s+completa)?\s*\*?|dir)\s*[:*\-]?\s*(.+)',
        'indications': r'(?:\*?\s*(?:indicaciones?|detalles?|barrio|referencias?|datos?\s+adicionales?|informaci[oó]n\s+adicional)\s*\*?)\s*[:*\-]?\s*(.+)',
        'locality':    r'(?:\*?\s*localidad(?:\s+de\s+entrega)?\s*\*?)\s*[:*\-]?\s*(.+)',
        'day':         r'(?:\*?\s*(?:d[ií]a(?:\s+de\s+entrega)?|fecha(?:\s+de\s+entrega)?|entrega\s+el|dia\s+entrega)\s*\*?)\s*[:*\-]?\s*(.+)',
    }
    tl = text.lower()
    for field, pattern in patterns.items():
        m = re.search(pattern, tl, re.IGNORECASE | re.MULTILINE)
        if m:
            val = re.sub(r'[*]', '', m.group(1).strip().split('\n')[0]).strip()
            if val and normalize(val) not in ('ninguna','ninguno','n/a','na','-','no tengo','no hay'):
                fields[field] = val
    return fields

# ============================================================
# EXTRACCIÓN DE DÍA
# ============================================================

def extract_day(text: str):
    norm = normalize(text)
    words = norm.split()

    for k, v in VALID_DAYS.items():
        if re.search(r'\b' + normalize(k) + r'\b', norm):
            return v, None

    for w in words:
        wc = w.strip('.,;:!?')
        if wc in DAY_ALIASES:
            return DAY_ALIASES[wc], None
    for alias, val in DAY_ALIASES.items():
        if ' ' in alias and alias in norm:
            return val, None

    for k, v in VALID_DAYS.items():
        kn = normalize(k)
        patterns = [
            r'(?:proximo|próximo|el|este|para\s+el|para\s+el\s+dia|el\s+dia)\s+' + kn,
            kn + r'\s+(?:que\s+viene|próximo|proximo|siguiente)',
        ]
        for p in patterns:
            if re.search(p, norm):
                return v, None

    for bad, msg in INVALID_DAYS.items():
        if re.search(r'\b' + normalize(bad) + r'\b', norm):
            return None, msg

    date_p = r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b'
    if re.search(date_p, norm):
        return None, "Por favor envíame solo el nombre del día 😊 Por ejemplo: Viernes"

    EXCLUDE_FROM_FUZZY_DAY = set()
    for k in VALID_LOCATIONS:
        for w in normalize(k).split():
            if len(w) >= 4:
                EXCLUDE_FROM_FUZZY_DAY.add(w)
    EXCLUDE_FROM_FUZZY_DAY.update([
        'maria', 'marta', 'mario', 'marco', 'lucia', 'luisa', 'laura',
        'diana', 'dina', 'bosa', 'suba', 'cali', 'lopez', 'loaiza',
        'calle', 'carrera', 'avenida', 'diagonal', 'transversal',
        'casa', 'apto', 'piso', 'torre', 'bloque', 'norte', 'sur',
        'este', 'oeste', 'bis', 'interior', 'local', 'oficina',
    ])

    best_v, best_s = None, 0.0
    threshold = 0.75
    for w in words:
        wc = w.strip('.,;:!?')
        if len(wc) < 3:
            continue
        if wc in EXCLUDE_FROM_FUZZY_DAY:
            continue
        for k, v in VALID_DAYS.items():
            s = similarity(wc, k)
            if s > best_s and s >= threshold:
                best_s, best_v = s, v
    if best_v:
        return best_v, None

    vague = ['lo antes', 'lo mas pronto', 'urgente', 'ya', 'ahora',
             'cuando puedan', 'pronto', 'rapido', 'rápido']
    for v in vague:
        if v in norm:
            return None, "¿Cuál día te queda mejor? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado 😊"

    return None, None

# ============================================================
# EXTRACCIÓN DE LOCALIDAD
# ============================================================

def extract_locality(text: str):
    norm = normalize(text)
    words = norm.split()

    for k, v in sorted(VALID_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        kn = normalize(k)
        if re.search(r'\b' + re.escape(kn) + r'\b', norm):
            if v is None:
                return None, "Lo sentimos, por ahora no llegamos a esa zona 😔 Cubrimos: Bosa, Kennedy, Suba, Chapinero, Engativá, Fontibón, Teusaquillo, Usaquén, Barrios Unidos, Puente Aranda, Tunjuelito, Antonio Nariño, Mártires, Soacha y Candelaria"
            return v, None

    best_v, best_s = None, 0.0
    best_invalid = False
    thresh = 0.78

    for w in words:
        if len(w) < 4:
            continue
        for k, v in VALID_LOCATIONS.items():
            if ' ' in k:
                continue
            s = similarity(w, normalize(k))
            if s > best_s and s >= thresh:
                best_s, best_v, best_invalid = s, v, (v is None)

    for i in range(len(words) - 1):
        bg = words[i] + ' ' + words[i+1]
        for k, v in VALID_LOCATIONS.items():
            if ' ' not in k:
                continue
            s = similarity(bg, normalize(k))
            if s > best_s and s >= 0.80:
                best_s, best_v, best_invalid = s, v, (v is None)

    if best_v is not None:
        return best_v, None
    if best_invalid:
        return None, "Lo sentimos, por ahora no llegamos a esa zona 😔"

    return None, None

# ============================================================
# EXTRACCIÓN DE DIRECCIÓN E INDICACIONES
# ============================================================

ADDR_START = r'(?:calle|cll|cl|carrera|cra|cr|kra|avenida|av|transversal|transv|tranv|tv|diagonal|dg|autopista|ak)'
ADDR_START_STRICT = r'\b(?:calle|cll|carrera|avenida|av|transversal|tv|diagonal|dg|autopista|ak)\b'

COMMON_NAMES = {
    'pedro', 'carlos', 'maria', 'juan', 'jose', 'luis', 'ana', 'sofia',
    'miguel', 'jorge', 'andres', 'alejandro', 'david', 'daniel', 'paula',
    'laura', 'diana', 'monica', 'patricia', 'andrea', 'camila', 'valentina',
    'julian', 'sebastian', 'nicolas', 'felipe', 'sergio', 'oscar', 'hugo',
    'ivan', 'omar', 'edgar', 'henry', 'mario', 'hector', 'rafael', 'cesar',
    'gustavo', 'rodrigo', 'nelson', 'wilson', 'jhon', 'john', 'jhonatan',
    'cristian', 'christian', 'fabian', 'hernan', 'fredy', 'freddy', 'giovanny',
    'giovanni', 'leidy', 'lady', 'angie', 'yuli', 'juli', 'paola', 'carol',
    'jenny', 'yenny', 'tatiana', 'viviana', 'lorena', 'natalia', 'catalina',
    'alejandra', 'marcela', 'claudia', 'sandra', 'esperanza', 'luz', 'gloria',
    'olga', 'martha', 'marta', 'blanca', 'rosa', 'carmen', 'amparo',
}

INDIC_KW = [
    'apto','apartamento','apt','torre','bloque','interior','int',
    'piso','local','oficina','conjunto','edificio','etapa','unidad',
    'porteria','portería','dejar en','entregar en','llamar','timbrar',
    'rejas','reja','esquina','frente','cerca','al lado',
    'despues','después','si no estoy','dejar con','portero',
    'casa azul','casa blanca','casa roja','casa verde','casa amarilla',
    'primer piso','segundo piso','tercer piso','cuarto piso',
    'peluqueria','peluquería','tienda','drogueria','droguería',
    'supermercado','parque','iglesia','colegio','hospital','clinica',
    'clínica','farmacia','restaurante','panaderia','panadería',
    'diagonal a','frente al','al lado de','cerca al','detras','detrás',
]


def extract_address_indications(text: str) -> tuple:
    lines_orig = text.split('\n')
    if len([l for l in lines_orig if l.strip()]) == 1:
        m = re.search(ADDR_START_STRICT, text, re.IGNORECASE)
        if m:
            text = text[m.start():]

    lines = [l.strip() for l in text.replace(',', '\n').split('\n') if l.strip()]
    address = None
    indics = []

    for line in lines:
        nl = normalize(line)
        cl = clean_noise(line).strip()
        if not cl:
            continue
        if re.match(ADDR_START, nl, re.IGNORECASE):
            m = re.match(
                r'(' + ADDR_START + r'\s*[\w\s.\-#bis]+?\d+[\w\s.\-#]*\d*)',
                cl, re.IGNORECASE
            )
            if m:
                address = m.group(1).strip()
                rest = cl[len(address):].strip().strip(',').strip()
                if rest and len(rest) > 2:
                    indics.append(rest)
            else:
                address = cl
        elif any(kw in nl for kw in INDIC_KW):
            c = clean_noise(line).strip()
            if c and len(c) > 2:
                indics.append(c)

    if not address:
        # OJO: usar [ \t] en vez de \s — \s también matchea \n, y este
        # regex de respaldo (re.search sin anclar a una línea) podía
        # "tragarse" el resto del mensaje completo cuando el nombre y la
        # dirección venían juntos en la misma línea (ej: "Maria Calle 5 #10-20
        # \nBarrio X\nLunes" — sin este fix, address terminaba incluyendo
        # también el barrio y el día de las líneas siguientes).
        m = re.search(
            r'(' + ADDR_START + r'[ \t]*[\w \t.\-#]+?\d+[\w \t.\-#]*\d*)',
            text, re.IGNORECASE
        )
        if m:
            address = m.group(1).strip()

    seen, clean_indics = set(), []
    for p in indics:
        np_ = normalize(p)
        if np_ not in seen and len(np_) > 2:
            seen.add(np_)
            clean_indics.append(p)

    return address, ', '.join(clean_indics[:3])

# ============================================================
# EXTRACCIÓN DE NOMBRE
# ============================================================

def extract_name(text: str):
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) > 1:
        for line in lines[:2]:
            n = normalize(line)
            if (re.match(r'^[a-záéíóúñ\s]+$', n)
                    and 1 <= len(line.split()) <= 5
                    and not re.search(ADDR_START_STRICT, n)):
                return line.strip().title()

    norm_text = normalize(text)
    m = re.search(ADDR_START_STRICT, norm_text)
    if m:
        before = norm_text[:m.start()].strip()
        if before:
            words = before.split()
            if re.match(r'^[a-záéíóúñ\s]+$', before):
                if 2 <= len(words) <= 4:
                    return ' '.join(w.capitalize() for w in words)
                elif len(words) == 1 and words[0] in COMMON_NAMES:
                    return words[0].capitalize()

    return None


# ============================================================
# EXTRACTOR PRINCIPAL
# ============================================================

def _strip_trailing_known_value(address: str, *values: str) -> str:
    """
    Recorta del final de `address` una repetición de un valor ya resuelto
    por separado (localidad o día) que quedó pegada a la dirección.

    Pasa esto SOLO en mensajes de una sola línea física sin comas ni saltos
    de línea (ej: "Cra 74 # 160 - 83 Suba Lunes") — ahí el regex de
    dirección no tiene ninguna marca estructural donde detenerse y captura
    todo lo que sigue. Con saltos de línea o comas, extract_address_indications
    ya delimita bien el campo y esta función no encuentra nada que recortar.
    """
    if not address:
        return address
    result = address
    changed = True
    while changed:
        changed = False
        stripped = result.rstrip(' .,-')
        for value in values:
            if not value:
                continue
            for variant in {value, value.lower(), normalize(value)}:
                if not variant:
                    continue
                if (stripped.lower().endswith(variant.lower())
                        and len(stripped) > len(variant)):
                    cut_point = len(stripped) - len(variant)
                    if stripped[cut_point - 1] in ' .,-':
                        candidate = stripped[:cut_point].rstrip(' .,-')
                        if len(candidate) >= 3:
                            result = candidate
                            changed = True
                            break
            if changed:
                break
    return result.strip()


def extract_delivery(text: str, valid_locations=None, valid_days=None, valid_times=None) -> dict:
    if not text or not text.strip():
        return {"error": True, "errorMessage": "No recibí ningún mensaje 😊", "info": None}

    words = text.strip().split()
    if len(words) <= 2:
        day, _ = extract_day(text)
        if day:
            return {"error": True,
                    "errorMessage": "Solo recibí el día de entrega 😊 Por favor envíame también tu nombre, dirección y localidad",
                    "info": None}
        return {"error": True, "errorMessage": "No se pudo extraer la información de entrega", "info": None}

    clean = clean_noise(text)
    labeled = extract_labeled(clean)

    raw_name = labeled.get('name', '')
    if raw_name:
        nn = normalize(raw_name.split('\n')[0].strip())
        name = raw_name.split('\n')[0].strip() if re.match(r'^[a-záéíóúñ\s]+$', nn) and len(raw_name.split()) <= 5 else extract_name(clean)
    else:
        name = extract_name(clean)

    address = indications = None
    indications = labeled.get('indications', '')
    if labeled.get('address'):
        address = labeled['address']
        if not indications:
            _, ind = extract_address_indications(clean)
            indications = ind
    else:
        address, ind = extract_address_indications(clean)
        if ind and not indications:
            indications = ind

    locality = locality_err = None
    if labeled.get('locality'):
        locality, locality_err = extract_locality(labeled['locality'])
    if not locality and not locality_err:
        locality, locality_err = extract_locality(clean)

    if locality_err and "no llegamos" in locality_err:
        return {"error": True, "errorMessage": locality_err, "info": None}

    day = day_err = None
    if labeled.get('day'):
        day, day_err = extract_day(labeled['day'])
    if not day and not day_err:
        day, day_err = extract_day(clean)

    if not day and day_err:
        return {"error": True, "errorMessage": day_err, "info": None}

    # Recortar localidad/día que hayan quedado pegados al final de la
    # dirección en mensajes de una sola línea sin comas ni saltos de línea.
    if address:
        address = _strip_trailing_known_value(address, day or "", locality or "")

    missing = []
    if not address:
        missing.append("dirección completa (ej: Calle 13 #45-67)")
    if not locality:
        missing.append("localidad (ej: Kennedy, Suba, Chapinero)")
    if not day:
        missing.append("día de entrega (Lunes a Sábado)")

    if missing:
        if len(missing) == 1:
            f = missing[0]
            art = "el" if f.startswith("día") else "la"
            msg = f"Faltó {art} {f} 😊 ¿Me lo puedes enviar?"
        elif len(missing) == 2:
            msg = f"Faltan: {missing[0]} y {missing[1]} 😊"
        else:
            msg = "No se pudo extraer la información de entrega"
        return {"error": True, "errorMessage": msg, "info": None}

    return {
        "error": False,
        "errorMessage": None,
        "info": {
            "address": address,
            "indications": indications or "",
            "locationDelivery": locality,
            "dayDelivery": day,
            "timeDelivery": "morning",
            "latitude": 4.7110,
            "longitude": -74.0721,
            "userName": name,
        }
    }


# ============================================================
# DICCIONARIO DE CONTEXTO POR TIPO DE NEGOCIO
# ============================================================

BUSINESS_CONTEXT: dict[str, list[str]] = {
    "muebles": [
        "cama", "camas", "sofa", "sofas", "silla", "sillas", "mesa", "mesas",
        "closet", "closets", "armario", "armarios", "comoda", "comodas",
        "escritorio", "escritorios", "estante", "estantes", "biblioteca",
        "buro", "buros", "camarote", "camarotes", "litera", "literas",
        "colchon", "colchones", "sala", "salas", "alcoba", "alcobas", "habitacion",
        "comedor", "comedores", "mueble", "muebles", "madera", "tapizado", "espejo",
        "espejos", "tocador", "tocadores", "alacena", "alacenas", "vitrina",
        "vitrinas", "rinconera", "rinconeras", "baul", "baules", "puff",
        "taburete", "taburetes", "mecedora", "mecedoras", "poltrona",
        "poltronas", "sillon", "sillones", "zapatera", "zapateras",
        "sofas camas", "sofa cama", "sofacama", "sofacamas",
        "sala comedor", "sala y comedor", "mesa comedor", "silla comedor", "sillas comedor",
    ],
    "frutas_verduras": [
        "manzana", "manzanas", "pera", "peras", "naranja", "naranjas",
        "limon", "limones", "banano", "bananos", "platano", "platanos",
        "mango", "mangos", "papaya", "papayas", "fresa", "fresas",
        "mora", "moras", "uva", "uvas", "sandia", "melon", "melones",
        "kiwi", "piña", "pinas", "guayaba", "guayabas", "mandarina",
        "mandarinas", "tomate", "tomates", "papa", "papas", "cebolla",
        "cebollas", "zanahoria", "zanahorias", "lechuga", "lechugas",
        "espinaca", "espinacas", "brocoli", "coliflor", "pepino",
        "pepinos", "aguacate", "aguacates", "cilantro", "apio",
        "verdura", "verduras", "fruta", "frutas", "mercado", "kilo",
    ],
    "ropa": [
        "camisa", "camisas", "pantalon", "pantalones", "vestido", "vestidos",
        "falda", "faldas", "blusa", "blusas", "chaqueta", "chaquetas",
        "abrigo", "abrigos", "zapato", "zapatos", "tenis", "bota", "botas",
        "sandalias", "ropa", "prenda", "prendas", "tela", "tallas",
        "camiseta", "camisetas", "jean", "jeans", "sudadera", "sudaderas",
        "pijama", "pijamas", "ropa interior", "calcetines", "medias",
        "gorra", "gorras", "bolso", "bolsos", "cartera", "carteras",
        "cinturon", "cinturones", "corbata", "corbatas", "bufanda",
    ],
    "restaurante": [
        "menu", "plato", "platos", "almuerzo", "almuerzos", "desayuno",
        "desayunos", "cena", "cenas", "sopa", "sopas", "bandeja", "corriente",
        "jugo", "jugos", "bebida", "bebidas", "postre", "postres",
        "arroz", "pollo", "carne", "carnes", "pescado", "ensalada",
        "ensaladas", "hamburguesa", "hamburguesas", "pizza", "pizzas",
        "pasta", "pastas", "sandwich", "sandwiches", "comida", "pedido",
    ],
    "drogueria": [
        "medicamento", "medicamentos", "pastilla", "pastillas", "capsula",
        "capsulas", "jarabe", "jarabes", "crema", "cremas", "shampoo",
        "acondicionador", "vitamina", "vitaminas", "suplemento", "suplementos",
        "gel", "antibiotico", "antibioticos", "analgesico", "analgesicos",
        "antigripal", "antigripales", "desinfectante", "alcohol", "tapabocas",
        "jeringa", "jeringas", "vendaje", "vendajes", "pañal", "pañales",
        "protector", "solar", "perfume", "perfumes", "locion", "lociones",
    ],
    "panaderia": [
        "pan", "panes", "croissant", "croissants", "galleta", "galletas",
        "torta", "tortas", "pastel", "pasteles", "ponque", "ponques",
        "buñuelo", "buñuelos", "empanada", "empanadas", "arepa", "arepas",
        "almojabana", "almojabanas", "mogolla", "mogollas", "pandebono",
        "pandebonos", "roscón", "roscones", "muffin", "muffins",
        "brownie", "brownies", "cheesecake", "donut", "donuts",
        "cafe", "chocolate", "bebida", "bebidas",
    ],
}

BUSINESS_TYPE_DESCRIPTIONS = {
    "frutas_verduras": {
        "products_label": "frutas y verduras frescas",
        "emoji": "🥦🍎",
        "delivery_label": "domicilio",
        "catalog_verb": "ver los productos frescos de hoy",
    },
    "muebles": {
        "products_label": "muebles y productos para el hogar",
        "emoji": "🛋️🪑",
        "delivery_label": "entrega a domicilio",
        "catalog_verb": "ver nuestros muebles",
    },
    "ropa": {
        "products_label": "ropa y accesorios",
        "emoji": "👕👗",
        "delivery_label": "envío a domicilio",
        "catalog_verb": "ver nuestras prendas",
    },
    "restaurante": {
        "products_label": "platos y bebidas",
        "emoji": "🍽️🥘",
        "delivery_label": "domicilio",
        "catalog_verb": "ver el menú",
    },
    "drogueria": {
        "products_label": "medicamentos y productos de salud",
        "emoji": "💊🏥",
        "delivery_label": "domicilio",
        "catalog_verb": "ver los productos disponibles",
    },
    "panaderia": {
        "products_label": "pan, pasteles y productos de panadería",
        "emoji": "🥐🍰",
        "delivery_label": "domicilio",
        "catalog_verb": "ver los productos del día",
    },
}

DEFAULT_BUSINESS = {
    "products_label": "productos",
    "emoji": "📦",
    "delivery_label": "domicilio",
    "catalog_verb": "ver nuestros productos",
}


# ============================================================
# BÚSQUEDA DE PRODUCTOS
# ============================================================

def get_business_context_words(business_type: str) -> list[str]:
    return BUSINESS_CONTEXT.get(business_type, [])


def extract_product_from_url(message: str, products: list) -> dict | None:
    match = re.search(r'wa\.me/p/([^/\s]+)', message)
    if not match:
        return None
    product_id = match.group(1)
    for product in products:
        wa_url = product.get("whatsapp_url", "")
        if product_id in wa_url:
            return product
    return None


# ============================================================
# MATCH ESTRICTO DE CATEGORÍA (con prioridad absoluta)
# ============================================================

# Palabras de relleno comunes en mensajes de WhatsApp que no aportan
# significado de categoría/producto — se descartan antes de comparar.
# Alcance decidido explícitamente: esto permite detectar la categoría
# dentro de frases más largas ("quiero ver los combos" → "combos"),
# pero NO absorbe calificadores de producto ("princesa" en "combo
# princesa" no está aquí, así que esa frase no se reduce a "combo").
CATEGORY_FILLER_WORDS = {
    "quiero", "ver", "veo", "los", "las", "el", "la", "un", "una",
    "unos", "unas", "por", "favor", "porfa", "porfavor", "dame",
    "muestrame", "muéstrame", "enseñame", "enséñame", "mostrar",
    "tienen", "tienes", "hay", "algo", "de", "en", "son", "es",
    "que", "para", "con", "y", "o", "me", "puedo", "podria",
    "podría", "quisiera", "busco", "buscando", "necesito", "tambien",
    "también", "pls", "please", "oye", "hola",
}


def _word_form_variants(word: str) -> set[str]:
    """
    Genera variantes simples singular/plural de una palabra normalizada,
    sin depender de scoring fuzzy — es una regla determinística, no
    aproximada, pensada para que "sofa" == "sofas" o "comedor" ==
    "comedores" cuenten como coincidencia EXACTA de categoría.
    """
    variants = {word}
    if word.endswith('es') and len(word) > 4:
        variants.add(word[:-2])          # comedores -> comedor
    if word.endswith('s') and len(word) > 3:
        variants.add(word[:-1])          # combos -> combo
    if not word.endswith('s'):
        variants.add(word + 's')         # sofa -> sofas
        variants.add(word + 'es')        # comedor -> comedores
    return variants


def _category_phrase_variants(category_norm: str) -> set[str]:
    """Variantes de una categoría (posiblemente multi-palabra) combinando
    las variantes singular/plural de cada palabra que la compone."""
    words = category_norm.split()
    if not words:
        return set()
    per_word_variants = [_word_form_variants(w) for w in words]
    return {' '.join(combo) for combo in itertools.product(*per_word_variants)}


def _category_word_variants(category_norm: str) -> set[str]:
    """Unión de las variantes singular/plural de CADA palabra de la
    categoría por separado (no de la frase completa) — permite reconocer
    categorías multi-palabra ("Combos de muebles") cuando el cliente solo
    menciona una parte ("combos", "muebles")."""
    variants = set()
    for w in category_norm.split():
        variants |= _word_form_variants(w)
    return variants


def _match_category_exact_or_phrase(query_norm: str, products: list) -> list:
    """
    PASO 0 de find_products_by_query — prioridad absoluta de categoría.

    Retorna la lista de productos cuya categoría coincide con el query,
    por cualquiera de estas vías:
    a) literalmente (forma actual de la categoría en la base), o
    b) en singular/plural ("sofas" cliente vs "Sofá" categoría), o
    c) como núcleo de una frase más larga tras quitar palabras de
       relleno ("quiero ver los combos" -> núcleo "combos"), o
    d) palabra por palabra contra una categoría de varias palabras
       ("combos" o "muebles" sueltos vs categoría "Combos de muebles"
       — el cliente no tiene por qué repetir la categoría completa).

    Si no hay match, retorna [] y el flujo sigue con el scoring normal
    (por nombre de producto y luego categoría fuzzy).
    """
    core_words = [w for w in query_norm.split() if w not in CATEGORY_FILLER_WORDS]
    core_phrase = ' '.join(core_words)

    if not core_phrase:
        return []

    matches = []
    for product in products:
        category_norm = normalize(str(product.get("category", "")))
        if not category_norm:
            continue

        # (a)(b)(c) — frase completa (o su núcleo) coincide con la categoría
        phrase_variants = _category_phrase_variants(category_norm)
        if query_norm in phrase_variants or core_phrase in phrase_variants:
            matches.append(product)
            continue

        # (d) — cada palabra del núcleo del query aparece como palabra
        # (o variante singular/plural) DENTRO de la categoría. Esto es lo
        # que resuelve "combos" solo contra "Combos de muebles": no exige
        # que el query repita la categoría completa, solo que todo lo que
        # el cliente escribió (sin relleno) pertenezca a la categoría.
        # "combo princesa" NO dispara esto porque "princesa" no es palabra
        # de ninguna categoría — sigue cayendo en scoring de nombre de
        # producto, igual que antes.
        category_word_variants = _category_word_variants(category_norm)
        if core_words and all(
            _word_form_variants(w) & category_word_variants for w in core_words
        ):
            matches.append(product)

    return matches


def find_products_by_query(query: str, products: list) -> tuple[list, str]:
    """
    Busca productos por query del cliente.

    Retorna (productos, método) donde método es:
    - 'category_exact'  → query coincide con una categoría (literal, plural/
                           singular, o dentro de una frase con relleno)
    - 'category_fuzzy'  → query coincide por fuzzy (typo) con una categoría
    - 'product_name'    → query coincide con nombre de producto específico
    - 'none'            → sin resultados

    PRIORIDAD ABSOLUTA DE CATEGORÍA (fix — ver PASO 0):
    Cuando el query es (o contiene, quitando palabras de relleno) el nombre
    de una categoría — en su forma literal o en singular/plural — eso
    CORTOCIRCUITA el flujo: se retorna la lista de esa categoría de inmediato,
    sin pasar por el scoring de nombre de producto. Antes, ese match competía
    por score contra nombres de producto y a veces perdía (ej: "combos" vs
    categoría "Combos" con score 1.0, pero un producto cuyo nombre contenía
    parcialmente esas letras podía acercarse). Esto ya no puede pasar: si hay
    match de categoría en PASO 0, no se evalúa ningún producto por nombre.

    ALCANCE DECIDIDO EXPLÍCITAMENTE (no implícito): el cortocircuito de
    categoría SÍ cubre frases más largas que solo contienen el nombre de la
    categoría más palabras de relleno ("quiero ver los combos" → detecta
    "combos" tras quitar "quiero ver los"). NO cubre frases que además traen
    un calificador de producto específico ("combo princesa" → "princesa" no
    es relleno, así que no dispara el cortocircuito y cae al scoring normal,
    donde debe ganar 'product_name' — comportamiento ya cubierto por el fix
    anterior de esta misma función).

    CONTRATO CON EL BACKEND: cuando el método retornado es 'category_exact'
    o 'category_fuzzy', el caller (/respond-tenant) NO debe setear
    matched_product — eso solo aplica a 'product_name'. Este contrato ya
    está implementado así en el endpoint (solo llena matched_product cuando
    search_method == 'product_name'), así que no requiere cambios ahí.
    """
    if not products or not query:
        return [], 'none'

    query_norm = normalize(query)
    query_words = [w for w in query_norm.split() if len(w) >= 3]

    # ── PASO 0: Cortocircuito de categoría — prioridad absoluta ────────────
    category_match_products = _match_category_exact_or_phrase(query_norm, products)
    if category_match_products:
        return category_match_products[:8], 'category_exact'

    # ── PASO 1: Buscar siempre por nombre de producto específico ──────────
    # Esto corre SIEMPRE que no hubo match de categoría en el PASO 0
    best_name_product = None
    best_name_score = 0.0

    for product in products:
        name_norm = normalize(str(product.get("name", "")))
        name_words = [w for w in name_norm.split() if len(w) >= 3]

        # Score query completo vs nombre completo
        score = max(
            levenshtein(query_norm, name_norm),
            jaro_winkler(query_norm, name_norm),
        )

        # Score palabra por palabra del query vs nombre
        for w in query_words:
            ws = max(levenshtein(w, name_norm), jaro_winkler(w, name_norm))
            if ws > score:
                score = ws
            # Coincidencia exacta de palabra dentro del nombre
            if w in name_words or w in name_norm:
                score = max(score, 0.92)

        # Score palabras del nombre vs query (bidireccional)
        for nw in name_words:
            nws = max(levenshtein(nw, query_norm), jaro_winkler(nw, query_norm))
            if nws > score:
                score = nws
            if nw in query_words or nw in query_norm:
                score = max(score, 0.92)

        # Si TODAS las palabras del query aparecen en el nombre del producto,
        # es un match completo por nombre — debe ganarle a un match parcial
        # de categoría por una sola palabra (ej: "combo" vs "combos" = 0.96,
        # pero "combo princesa" completo en el nombre debe pesar más que eso)
        if query_words and all(w in name_norm for w in query_words):
            score = max(score, 0.97)

        if score > best_name_score:
            best_name_score = score
            best_name_product = product

    # ── PASO 2: Categoría por fuzzy (typos) — ya no incluye el caso exacto,
    # ese se resolvió en el PASO 0 con prioridad absoluta ──────────────────
    fuzzy_category_matches = []
    for product in products:
        category_norm = normalize(str(product.get("category", "")))
        if not category_norm:
            continue
        cat_score = max(levenshtein(query_norm, category_norm), jaro_winkler(query_norm, category_norm))
        if cat_score >= 0.85:
            fuzzy_category_matches.append((product, cat_score))
            continue
        if ' ' not in category_norm:
            word_score = max(
                (max(levenshtein(w, category_norm), jaro_winkler(w, category_norm)) for w in query_words),
                default=0
            )
            if word_score >= 0.88:
                fuzzy_category_matches.append((product, word_score))

    best_cat_score = max((s for _, s in fuzzy_category_matches), default=0.0)

    # ── PASO 3: Decidir — producto específico gana si su score es mayor ───
    PRODUCT_NAME_THRESHOLD = 0.75
    if (best_name_product and
        best_name_score >= PRODUCT_NAME_THRESHOLD and
        best_name_score > best_cat_score):
        return [best_name_product], 'product_name'

    # Si hay match fuzzy de categoría → categoría
    if fuzzy_category_matches:
        fuzzy_category_matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in fuzzy_category_matches[:8]], 'category_fuzzy'

    # Si hay match por nombre aunque no ganó → retornarlo igual
    if best_name_product and best_name_score >= PRODUCT_NAME_THRESHOLD:
        return [best_name_product], 'product_name'

    # Paso 3B: búsqueda por nombre más amplia (umbral más bajo)
    name_matches = []
    for product in products:
        name_norm = normalize(str(product.get("name", "")))
        for word in query_words:
            if word in name_norm:
                name_matches.append((product, 0.9))
                break
        else:
            name_score = max(levenshtein(query_norm, name_norm), jaro_winkler(query_norm, name_norm))
            word_score = max(
                (max(levenshtein(w, name_norm), jaro_winkler(w, name_norm)) for w in query_words),
                default=0
            )
            best = max(name_score, word_score)
            if best >= 0.72:
                name_matches.append((product, best))

    name_matches.sort(key=lambda x: x[1], reverse=True)
    if name_matches:
        return [m[0] for m in name_matches[:5]], 'product_name'

    return [], 'none'


def is_product_query(message: str, products: list, business_type: str = "") -> bool:
    if extract_product_from_url(message, products):
        return True

    msg_norm = normalize(message)
    msg_words = [w for w in msg_norm.split() if len(w) >= 4]

    if not msg_words:
        return False

    if business_type:
        context_words = set(get_business_context_words(business_type))
        for word in msg_words:
            if word in context_words:
                return True
            for cw in context_words:
                if len(cw) >= 7 and len(word) >= 7 and (word in cw or cw in word):
                    return True

    for product in products:
        cat_norm = normalize(str(product.get("category", "")))
        if cat_norm:
            for word in msg_words:
                if word in cat_norm or cat_norm in word:
                    return True
            if max((jaro_winkler(w, cat_norm) for w in msg_words), default=0) >= 0.85:
                return True

        name_norm = normalize(str(product.get("name", "")))
        name_words = [w for w in name_norm.split() if len(w) >= 4]
        for word in msg_words:
            for nw in name_words:
                if word == nw or (len(word) >= 5 and word in name_norm):
                    return True
        if max((jaro_winkler(w, name_norm) for w in msg_words), default=0) >= 0.85:
            return True

    return False


# ============================================================
# GENERADOR DE RESPUESTAS PARA TENANTS
# ============================================================

def format_price(price) -> str:
    """
    Formatea un precio a '$1.234.567' de forma SEGURA — nunca lanza
    excepción. Un precio con formato raro en un solo producto no debe
    tumbar toda la respuesta cuando se está listando una categoría entera
    (bug real: price="2.700.000" con puntos como separador de miles hacía
    fallar int(float(price)) y /respond-tenant no respondía nada).

    Tolera: números, strings con puntos de miles ("2.700.000"), strings
    con coma decimal ("2700000,00"), símbolo $, espacios, vacío o None.
    Si de verdad no se puede interpretar, retorna "" en vez de romper la
    petición completa.
    """
    if price is None or price == "":
        return ""
    try:
        value = float(price)
    except (TypeError, ValueError):
        s = str(price).strip().replace("$", "").replace(" ", "")
        if re.match(r'^\d{1,3}(\.\d{3})+$', s):
            # "2.700.000" — puntos como separador de miles, sin decimales
            s = s.replace(".", "")
        elif re.search(r',\d{1,2}$', s):
            # "2.700.000,00" o "2700000,00" — coma decimal estilo latino
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
        try:
            value = float(s)
        except (TypeError, ValueError):
            return ""
    try:
        return f"${int(value):,}".replace(",", ".")
    except (TypeError, ValueError, OverflowError):
        return ""


def build_tenant_response(intent: str, confidence: float, message: str, tenant: dict, products: list) -> str:
    bot_name = tenant.get("bot_name", "tu asistente")
    business_name = tenant.get("business_name", "nuestro negocio")
    business_type = tenant.get("business_type", "otro")
    catalog_url = tenant.get("catalog_url", "")
    delivery_type = tenant.get("delivery_type", "domicilio")
    locations = tenant.get("locations", [])

    biz = BUSINESS_TYPE_DESCRIPTIONS.get(business_type, DEFAULT_BUSINESS)
    bus_emoji = biz["emoji"]

    location_names = [l.get("name", "") for l in locations if l.get("name")]
    locations_text = ", ".join(location_names) if location_names else "Bogota"

    all_days = set()
    for loc in locations:
        for day in loc.get("days", []):
            all_days.add(day)
    days_text = ", ".join(sorted(all_days)) if all_days else "Lunes a Sabado"

    if delivery_type == "domicilio":
        delivery_text = f"entregamos a domicilio gratis en {locations_text}"
    elif delivery_type == "punto_fisico":
        delivery_text = "puedes recoger en nuestro punto fisico"
    else:
        delivery_text = f"hacemos domicilio en {locations_text} y tambien puedes recoger en el punto"

    product_from_url = extract_product_from_url(message, products)
    if product_from_url:
        matching_products = [product_from_url]
        search_method = 'product_name'
    else:
        matching_products, search_method = find_products_by_query(message, products)

    if intent == "saludo":
        catalog_line = f" Puedes ver el catalogo completo aqui: {catalog_url}" if catalog_url else ""
        return random.choice([
            f"Hola! Soy {bot_name} de {business_name}. Tenemos {biz['products_label']} {bus_emoji} y {delivery_text}.{catalog_line} En que te puedo ayudar hoy?",
            f"Hola! Bienvenido a {business_name}. Soy {bot_name}, tu asistente virtual.{catalog_line} Que estas buscando hoy? {bus_emoji}",
            f"Hola! Soy {bot_name} de {business_name}. Estoy aqui para ayudarte con {biz['products_label']}.{catalog_line} Que necesitas?",
        ])

    elif intent == "consulta_producto":
        if matching_products:
            # Producto específico — mostrar ficha detallada
            if search_method == 'product_name':
                p = matching_products[0]
                price_fmt = format_price(p.get("price", 0))
                response = f"{bus_emoji} *{p.get('name', 'Producto')}*"
                if price_fmt: response += f"\nPrecio: {price_fmt}"
                if p.get("description"): response += f"\n\n{p['description']}"
                if p.get("whatsapp_url"): response += f"\n\nVer en catalogo: {p['whatsapp_url']}"
                return response + "\n\n¿Te gustaria pedirlo? 😊"

            # Categoría — mostrar lista (solo nombre, precio y link — sin
            # descripción. Además de ser más limpio para una lista, evita
            # superar el límite de 4096 caracteres de un mensaje de texto
            # de WhatsApp: con la descripción completa de 2-3 productos ya
            # se pasa ese límite, la API de Meta rechaza el envío y el bot
            # queda "mudo" sin ningún error visible, porque el backend no
            # revisa la respuesta del fetch a la Graph API.
            response = f"Claro! {bus_emoji} Encontre estos productos que te pueden interesar:\n\n"
            for p in matching_products:
                price_fmt = format_price(p.get("price", 0))
                line = f"*{p.get('name', 'Producto')}*"
                if price_fmt: line += f" - {price_fmt}"
                if p.get("whatsapp_url"): line += f"\n  Ver producto: {p['whatsapp_url']}"
                response += line + "\n\n"
            response += "Te interesa alguno? 😊"

            # Red de seguridad adicional: si aun así el mensaje quedó muy
            # largo (categoría con muchos productos), cortar y mandar al
            # catálogo en vez de arriesgar que WhatsApp rechace el envío.
            WHATSAPP_SAFE_LIMIT = 3500
            if len(response) > WHATSAPP_SAFE_LIMIT:
                response = response[:WHATSAPP_SAFE_LIMIT].rsplit("\n\n", 1)[0]
                response += f"\n\n...y más 😊 Ve el catálogo completo aquí: {catalog_url}"
            return response
        elif catalog_url:
            return random.choice([
                f"Claro! {bus_emoji} Puedes {biz['catalog_verb']} aqui: {catalog_url} Hay algo especifico que estes buscando?",
                f"Tenemos varios {biz['products_label']} disponibles {bus_emoji} Entra al catalogo: {catalog_url}",
            ])
        return f"Claro! {bus_emoji} Tenemos varios {biz['products_label']} disponibles. Que estas buscando exactamente?"

    elif intent == "consulta_catalogo":
        if catalog_url:
            return random.choice([
                f"Con gusto! Aqui tienes el catalogo de {business_name}: {catalog_url} {bus_emoji} Agrega lo que necesites.",
                f"Puedes ver todos nuestros {biz['products_label']} aqui: {catalog_url} {bus_emoji} Te ayudo con algo especifico?",
            ])
        return f"Escribeme que {biz['products_label']} buscas y te ayudo de inmediato {bus_emoji}"

    elif intent == "consulta_domicilio":
        if delivery_type == "punto_fisico":
            return "Puedes recoger tu pedido en nuestro punto fisico. Quieres mas informacion?"
        return random.choice([
            f"Si! {delivery_text.capitalize()} Los dias de entrega son {days_text}. Quieres hacer un pedido?",
            f"Claro! {delivery_text.capitalize()} Dias disponibles: {days_text}. En que zona estas?",
        ])

    elif intent == "hora_pedido":
        return random.choice([
            f"Las entregas son en horas de la manana entre 7 AM y 12 PM del dia que elijas. Tenemos disponibilidad los {days_text}.",
            f"Entregamos de 7 AM a 12 PM los dias {days_text}. Para cuando quieres tu pedido?",
        ])

    elif intent == "metodo_pago":
        return random.choice([
            "Aceptamos efectivo, Nequi y Bre-b El pago es al recibir tu pedido - sin anticipos!",
            "Puedes pagar en efectivo, Nequi o Bre-b cuando llegue tu pedido Sin anticipos!",
        ])

    elif intent == "pedido_no_ha_llegado":
        return random.choice([
            f"Lamento mucho la espera Ya informo al equipo de {business_name} ahora mismo para que revisen tu entrega.",
            "Que pena! Voy a notificar al equipo de inmediato. Pronto te contactaran para resolver esto.",
        ])

    elif intent == "queja_servicio":
        return random.choice([
            f"Lamento mucho lo ocurrido Tu comentario es muy importante para {business_name}. Ya informo al equipo para darte una solucion.",
            "Entiendo tu molestia y me disculpo Ya escalo tu caso al equipo para que te contacten pronto.",
        ])

    elif intent == "consulta_estado_pedido":
        return "Dejame verificar Me confirmas tu nombre y el dia que programaste la entrega?"

    elif intent == "agradecimiento":
        return random.choice([
            f"Con mucho gusto! Fue un placer ayudarte. Cuando quieras hacer otro pedido, aqui estamos en {business_name}.",
            f"A ti! Si necesitas algo mas, no dudes en escribir. {bus_emoji}",
        ])

    elif intent == "despedida":
        return random.choice([
            f"Hasta luego! Fue un placer atenderte en {business_name}. Cuando quieras volver, aqui estaremos.",
            f"Chao! Espero verte pronto. Recuerda que el catalogo siempre esta disponible: {catalog_url}",
        ])

    elif intent == "consulta_precio_minimo":
        return f"El pedido minimo es de $100.000 COP Y el domicilio es completamente gratis! Empezamos? {catalog_url}"

    elif intent == "pedir_hablar_humano":
        return f"Claro! Ya notifico a un asesor de {business_name} para que te atienda personalmente. En un momento te contactaran."

    elif intent == "pensar_luego":
        return random.choice([
            f"Sin problema! Cuando estes listo el catalogo te espera: {catalog_url} Aqui estare!",
            f"Perfecto! No hay afan. Cuando decidas, entra aqui: {catalog_url} Te espero!",
        ])

    elif intent == "festivos_horario_especial":
        return f"Trabajamos {days_text} Los domingos y festivos no hay entregas. Quieres programar para un dia disponible?"

    elif intent == "fuera_de_tema":
        return random.choice([
            f"Por ahora solo manejo lo de {business_name} Te puedo ayudar con algun {biz['products_label']}? {catalog_url}",
            f"Eso esta fuera de lo que puedo ayudarte, pero ya informo al equipo Para compras: {catalog_url}",
        ])

    elif intent == "consulta_descuentos":
        return random.choice([
            f"Nuestros precios ya son los mejores porque compramos directo! Sin intermediarios. Catalogo: {catalog_url}",
            f"Los precios del catalogo ya son los mejores del mercado {bus_emoji} {catalog_url}",
        ])

    elif intent == "cambiar_pedido":
        return f"Claro! Dime que quieres cambiar o agregar y lo notifico al equipo de {business_name} de inmediato 😊"

    elif intent == "dejar_vecino_porteria":
        return "Claro! Cuando envies tus datos de entrega agrega en indicaciones: 'Dejar con el portero' o 'Dejar con vecino del 201' y el domiciliario lo tendra en cuenta 😊"

    # Fallback con productos
    if matching_products and confidence > 0.1:
        response = f"Mira estos productos que tenemos {bus_emoji}\n\n"
        for p in matching_products:
            line = f"- {p.get('name', 'Producto')}"
            if p.get("whatsapp_url"): line += f"\n  {p['whatsapp_url']}"
            response += line + "\n\n"
        return response + "Te interesa alguno? 😊"

    return random.choice([
        f"Hola! Soy {bot_name} de {business_name}. En que te puedo ayudar? {catalog_url}",
        f"Entiendo En que te puedo ayudar? {bus_emoji}",
        f"Aqui estoy para ayudarte! Que necesitas de {business_name}?",
    ])


# ============================================================
# ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Verificar que el servicio está activo."""
    return jsonify({
        'status': 'ok',
        'service': 'corapp-ml',
        'version': '5.0.0',
        'endpoints': ['/predict', '/respond-tenant', '/extract-delivery', '/analyze-failures', '/retrain', '/create-group'],
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Clasifica la intención de un mensaje. Usado por corapp-backend para el flujo de CorApp."""
    data = request.get_json() or {}
    message = data.get('message', '')

    if not message:
        return jsonify({'intent': 'unknown', 'confidence': 0.0, 'use_fallback': True})

    try:
        intent, confidence = classify(message)
        use_fallback = confidence < 0.35
        return jsonify({
            'intent': intent,
            'confidence': round(confidence, 4),
            'use_fallback': use_fallback,
        })
    except Exception as e:
        print(f"❌ Error en /predict: {e}")
        return jsonify({'intent': 'unknown', 'confidence': 0.0, 'use_fallback': True})


@app.route('/respond-tenant', methods=['POST'])
def respond_tenant():
    """Genera respuesta contextualizada para un tenant específico."""
    data = request.get_json() or {}
    message = data.get('message', '')
    tenant = data.get('tenant', {})
    products = data.get('products', [])

    if not message:
        bot_name = tenant.get('bot_name', 'tu asistente')
        business_name = tenant.get('business_name', 'nuestro negocio')
        catalog_url = tenant.get('catalog_url', '')
        return jsonify({
            'response': f'Hola! Soy {bot_name} de {business_name}. En que te puedo ayudar? {catalog_url}',
            'intent': 'saludo',
            'confidence': 1.0,
            'matched_product': None,
            'matched_product_url': None,
        })

    # ✅ Regla determinística de respaldo: si el mensaje pide el catálogo
    # literalmente, forzar consulta_catalogo directo. No depende del
    # clasificador ML ni de is_product_query — evita que quede a merced
    # de coincidencias fuzzy accidentales (ej: "regalas" comparte letras
    # con nombres de producto como "Gala" y puede colar un match falso).
    if re.search(r'\bcatalogo\b', normalize(message)):
        intent, confidence = 'consulta_catalogo', 0.99
    elif is_product_query(message, products, tenant.get("business_type", "")):
        intent, confidence = 'consulta_producto', 0.99
    else:
        try:
            intent, confidence = classify(message)
        except Exception:
            intent, confidence = 'unknown', 0.0

    print(f"🔍 DEBUG mensaje={message!r} intent={intent} confidence={confidence}")
    response = build_tenant_response(intent, confidence, message, tenant, products)

    # Detectar producto específico para que el backend guarde el contexto
    # (pending_product_url / pending_product_name) sin volver a buscarlo.
    matched_product = None
    matched_product_url = None
    if intent == 'consulta_producto' and products:
        product_from_url = extract_product_from_url(message, products)
        if product_from_url:
            matched_product = product_from_url.get('name')
            matched_product_url = product_from_url.get('whatsapp_url')
        else:
            matching, search_method = find_products_by_query(message, products)
            if matching and search_method == 'product_name':
                matched_product = matching[0].get('name')
                matched_product_url = matching[0].get('whatsapp_url')

    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': round(confidence, 4),
        'matched_product': matched_product,
        'matched_product_url': matched_product_url,
    })


@app.route('/extract-delivery', methods=['POST'])
def extract_delivery_endpoint():
    """Extrae datos de entrega de un mensaje de texto."""
    data = request.get_json() or {}
    message = data.get('message', '')

    if not message:
        return jsonify({"success": False, "error": "Mensaje vacio"})

    result = extract_delivery(message)
    info = result.get('info', {})

    address = info.get('address') if info else None
    if not address and not result.get('error'):
        address = message.strip()

    if not address:
        return jsonify({
            "success": False,
            "error": result.get('errorMessage', 'No se pudo extraer la dirección'),
        })

    return jsonify({
        "success": True,
        "name": info.get('userName') if info else None,
        "address": address,
        "indications": info.get('indications', '') if info else '',
        "location": info.get('locationDelivery') if info else None,
        "day": info.get('dayDelivery') if info else None,
        "latitude": info.get('latitude', 4.7110) if info else 4.7110,
        "longitude": info.get('longitude', -74.0721) if info else -74.0721,
    })


@app.route('/analyze-failures', methods=['POST'])
def analyze_failures():
    """Analiza mensajes que fallaron la clasificación."""
    data = request.get_json() or {}
    messages = data.get('messages', [])

    if not messages:
        return jsonify({'results': []})

    results = []
    for msg in messages:
        text = msg.get('text', '')
        if not text:
            continue
        try:
            intent, confidence = classify(text)
            results.append({
                'text': text,
                'intent': intent,
                'confidence': round(confidence, 4),
                'use_fallback': confidence < 0.35,
                'suggestion': 'Agregar al training' if confidence < 0.35 else 'OK',
            })
        except Exception as e:
            results.append({'text': text, 'intent': 'error', 'confidence': 0.0, 'use_fallback': True, 'suggestion': str(e)})

    return jsonify({'results': results})


@app.route('/retrain', methods=['POST'])
def retrain():
    """Fuerza reentrenamiento del modelo desde data/training.json."""
    global model
    try:
        model = None
        if os.path.exists('model/intent_classifier.pkl'):
            os.remove('model/intent_classifier.pkl')
        model = train_model()
        return jsonify({'status': 'ok', 'message': '✅ Modelo reentrenado exitosamente'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/create-group', methods=['POST'])
def create_group():
    """Crea un grupo de Telegram para un tenant nuevo. Requiere X-Service-Secret."""
    secret = request.headers.get('X-Service-Secret')
    if secret != os.environ.get('TELEGRAM_SERVICE_SECRET'):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    business_name = data.get('business_name')

    if not business_name:
        return jsonify({"error": "business_name es requerido"}), 400

    try:
        result = create_tenant_group(business_name)
        return jsonify(result), 200
    except FloodWaitError as e:
        print(f"⚠️ FloodWaitError creando grupo Telegram: esperar {e.seconds}s")
        return jsonify({
            "error": f"Telegram limitó la cuenta temporalmente. Reintenta en {e.seconds} segundos."
        }), 429
    except Exception as e:
        print(f"❌ Error creando grupo Telegram: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# v5.0.0 — modelo completo con fuzzy search, aliases colombianos,
# respuestas empáticas, análisis de fallos y soporte multi-tenant
# v5.1.0 — agregado módulo Telegram para creación de grupos de tenants
# v5.2.0 — /respond-tenant ahora retorna matched_product
# v5.3.0 — normalize() quita puntuación de frase (fix "mango?");
#          regla determinística para "catalogo" antes del clasificador
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)