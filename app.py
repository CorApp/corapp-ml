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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

app = Flask(__name__)
model = None


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize(text: str) -> str:
    """Normaliza texto — minúsculas, sin tildes, sin espacios extra."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r'\s+', ' ', text)
    return text


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

# Alias colombianos — abreviaciones y errores comunes
DAY_ALIASES = {
    # Lunes
    "lun": "Lunes", "lns": "Lunes", "lnes": "Lunes", "lun.": "Lunes",
    "el lunes": "Lunes", "este lunes": "Lunes",
    # Martes
    "mar": "Martes", "mrt": "Martes", "mrts": "Martes",
    "el martes": "Martes", "este martes": "Martes",
    # Miércoles
    "mie": "Miércoles", "mier": "Miércoles", "mirc": "Miércoles",
    "mierc": "Miércoles", "miercole": "Miércoles", "miercols": "Miércoles",
    "merco": "Miércoles", "el miercoles": "Miércoles",
    # Jueves
    "jue": "Jueves", "jues": "Jueves", "jvs": "Jueves", "jves": "Jueves",
    "juev": "Jueves", "juevs": "Jueves", "jve": "Jueves", "jv": "Jueves",
    "el jueves": "Jueves",
    # Viernes
    "vie": "Viernes", "vies": "Viernes", "vrs": "Viernes",
    "viern": "Viernes", "vierens": "Viernes", "el viernes": "Viernes",
    # Sábado
    "sab": "Sábado", "sabs": "Sábado", "sbd": "Sábado",
    "sabdo": "Sábado", "sbdo": "Sábado", "sab.": "Sábado",
    "el sabado": "Sábado", "el sábado": "Sábado",
}

# Días inválidos con mensajes naturales
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
    # Localidades oficiales de Bogotá
    "bosa": "Bosa",
    "kennedy": "Kennedy",
    "puente aranda": "Puente aranda",
    "tunjuelito": "Tunjuelito",
    "antonio narino": "Antonio narino",
    "antonio nariño": "Antonio narino",
    "teusaquillo": "Teusaquillo",
    "barrios unidos": "Barrios unidos",
    "martires": "Martirez",
    "mártires": "Martirez",
    "martirez": "Martirez",
    "fontibon": "Fontibon",
    "fontibón": "Fontibon",
    "engativa": "Engativa",
    "engativá": "Engativa",
    "chapinero": "Chapinero",
    "usaquen": "Usaquen",
    "usaquén": "Usaquen",
    "soacha": "Soacha",
    "candelaria": "Candelaria",
    "suba": "Suba",
    "rafael uribe": "Rafael Uribe Uribe",
    "rafael uribe uribe": "Rafael Uribe Uribe",
    "ciudad bolivar": "Ciudad Bolivar",
    "ciudad bolívar": "Ciudad Bolivar",
    "san cristobal": "San Cristobal",
    "san cristóbal": "San Cristobal",
    "usme": "Usme",
    "sumapaz": "Sumapaz",
    # Barrios → Localidad
    "modelia": "Fontibon",
    "capellania": "Fontibon",
    "capellanía": "Fontibon",
    "fontibón sur": "Fontibon",
    "patio bonito": "Kennedy",
    "ciudad montes": "Puente aranda",
    "tibabuyes": "Suba",
    "tibabuyes universal": "Suba",
    "pinar": "Suba",
    "pinar de suba": "Suba",
    "porvenir": "Bosa",
    "bosa piamonte": "Bosa",
    "bosa libertad": "Bosa",
    "bosa nueva": "Bosa",
    "bosa el porvenir": "Bosa",
    "senderos del porvenir": "Bosa",
    "la libertad bosa": "Bosa",
    "alqueria": "Kennedy",
    "alquería": "Kennedy",
    "alqueria de la fragua": "Kennedy",
    "alquería de la fragua": "Kennedy",
    "prado veraniego": "Suba",
    "prado pinzon": "Suba",
    "prado pinzón": "Suba",
    "ciudad kennedy": "Kennedy",
    "cedro": "Engativa",
    "el cedro": "Engativa",
    "alamos": "Engativa",
    "álamos": "Engativa",
    "portales": "Engativa",
    "portales norte": "Engativa",
    "san agustin": "Kennedy",
    "san agustín": "Kennedy",
    "corabastos": "Kennedy",
    "zona industrial kennedy": "Kennedy",
    "castellon de los condes": "Kennedy",
    "castellón de los condes": "Kennedy",
    "cra 87b": "Kennedy",
    "antiguo country": "Chapinero",
    "country": "Chapinero",
    "rosales": "Chapinero",
    "portal de rosales": "Chapinero",
    "chapinero alto": "Chapinero",
    "chapinero norte": "Chapinero",
    "gran estacion": "Teusaquillo",
    "gran estación": "Teusaquillo",
    "palermo": "Teusaquillo",
    "la soledad": "Teusaquillo",
    "armenia": "Teusaquillo",
    "bahia solano": "Fontibon",
    "bahía solano": "Fontibon",
    "ciudad salitre": "Fontibon",
    "salitre": "Fontibon",
    "la giralda": "Fontibon",
    "villa del prado": "Suba",
    "alhambra": "Suba",
    "cedritos": "Usaquen",
    "santa barbara": "Usaquen",
    "santa bárbara": "Usaquen",
    "mazuren": "Suba",
    "mazurén": "Suba",
    "niza": "Suba",
    "bello horizonte": "Suba",
    "verbenal": "Usaquen",
    "toberin": "Usaquen",
    "toberín": "Usaquen",
    "country norte": "Usaquen",
    "santa cecilia": "Engativa",
    "villa luz": "Engativa",
    "gaitan": "Barrios unidos",
    "gaitán": "Barrios unidos",
    "alcazares": "Barrios unidos",
    "siete de agosto": "Barrios unidos",
    "la floresta": "Engativa",
    "floresta": "Engativa",
    "quirigua": "Engativa",
    "quiriguá": "Engativa",
    "minuto de dios": "Engativa",
    "bachue": "Engativa",
    "tintal": "Kennedy",
    "americas": "Kennedy",
    "américas": "Kennedy",
    "timiza": "Kennedy",
    "muzú": "Puente aranda",
    "muzu": "Puente aranda",
    "la esperanza": "Kennedy",
    "candelaria la nueva": "Kennedy",
    # Municipios fuera de cobertura → None
    "cundinamarca": None,
    "zipaquira": None,
    "zipaquirá": None,
    "chia": None,
    "chía": None,
    "sopo": None,
    "sopó": None,
    "cajica": None,
    "cajicá": None,
    "mosquera": None,
    "madrid cundinamarca": None,
    "facatativa": None,
    "facatativá": None,
    "funza": None,
    "tocancipa": None,
    "tocancipá": None,
    "la calera": None,
    "cota": None,
    "sibate": None,
    "sibaté": None,
}

# ============================================================
# LIMPIEZA DE RUIDO
# ============================================================

NOISE_PATTERNS = [
    r'[\w\.-]+@[\w\.-]+\.\w+',                              # emails
    r'\b3\d{9}\b',                                           # cel colombiano
    r'\b\d{7,10}\b',                                         # otros números
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
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# EXTRACCIÓN DE CAMPOS CON ETIQUETAS
# ============================================================

def extract_labeled(text: str) -> dict:
    """
    Detecta cuando el usuario usa etiquetas explícitas.
    Soporta: Nombre:, Dirección:, Barrio:, Localidad:, Día:
    Con o sin asteriscos, con o sin dos puntos.
    """
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
# EXTRACCIÓN DE DÍA — ULTRA ROBUSTA
# ============================================================

def extract_day(text: str):
    """
    7 estrategias en cascada para extraer el día:
    1. Exacto
    2. Alias y abreviaciones colombianas
    3. Contexto (próximo lunes, para el martes)
    4. Días inválidos con mensaje empático
    5. Fecha completa (28 de mayo)
    6. Búsqueda difusa multi-algoritmo
    7. Detección de intención temporal vaga
    """
    norm = normalize(text)
    words = norm.split()

    # 1. Exacto
    for k, v in VALID_DAYS.items():
        if re.search(r'\b' + normalize(k) + r'\b', norm):
            return v, None

    # 2. Alias y abreviaciones
    for w in words:
        wc = w.strip('.,;:!?')
        if wc in DAY_ALIASES:
            return DAY_ALIASES[wc], None
    # También buscar frases de alias
    for alias, val in DAY_ALIASES.items():
        if ' ' in alias and alias in norm:
            return val, None

    # 3. Contexto — "próximo X", "para el X", "el X que viene"
    for k, v in VALID_DAYS.items():
        kn = normalize(k)
        patterns = [
            r'(?:proximo|próximo|el|este|para\s+el|para\s+el\s+dia|el\s+dia)\s+' + kn,
            kn + r'\s+(?:que\s+viene|próximo|proximo|siguiente)',
        ]
        for p in patterns:
            if re.search(p, norm):
                return v, None

    # 4. Días inválidos con mensaje empático
    for bad, msg in INVALID_DAYS.items():
        if re.search(r'\b' + normalize(bad) + r'\b', norm):
            return None, msg

    # 5. Fecha completa
    date_p = r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b'
    if re.search(date_p, norm):
        return None, "Por favor envíame solo el nombre del día 😊 Por ejemplo: Viernes"

    # 6. Difusa multi-algoritmo
    # Palabras a excluir: localidades, nombres comunes, palabras de dirección
    EXCLUDE_FROM_FUZZY_DAY = set()
    for k in VALID_LOCATIONS:
        for w in normalize(k).split():
            if len(w) >= 4:
                EXCLUDE_FROM_FUZZY_DAY.add(w)
    # Nombres propios colombianos comunes que confunden el fuzzy
    EXCLUDE_FROM_FUZZY_DAY.update([
        'maria', 'marta', 'mario', 'marco', 'lucia', 'luisa', 'laura',
        'diana', 'dina', 'bosa', 'suba', 'cali', 'lopez', 'loaiza',
        'calle', 'carrera', 'avenida', 'diagonal', 'transversal',
        'casa', 'apto', 'piso', 'torre', 'bloque', 'norte', 'sur',
        'este', 'oeste', 'bis', 'interior', 'local', 'oficina',
    ])

    best_v, best_s = None, 0.0
    threshold = 0.75  # Más estricto para evitar falsos positivos
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

    # 7. Intención temporal vaga
    vague = ['lo antes', 'lo mas pronto', 'urgente', 'ya', 'ahora',
             'cuando puedan', 'pronto', 'rapido', 'rápido']
    for v in vague:
        if v in norm:
            return None, "¿Cuál día te queda mejor? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado 😊"

    return None, None


# ============================================================
# EXTRACCIÓN DE LOCALIDAD — ULTRA ROBUSTA
# ============================================================

def extract_locality(text: str):
    """
    4 estrategias para extraer localidad:
    1. Exacto incluyendo barrios y sectores
    2. Difusa para localidades principales
    3. Difusa para bigrams (dos palabras)
    4. Detección de municipios fuera de cobertura
    """
    norm = normalize(text)
    words = norm.split()

    # 1. Exacto — más largo primero para evitar matches parciales
    for k, v in sorted(VALID_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        kn = normalize(k)
        if re.search(r'\b' + re.escape(kn) + r'\b', norm):
            if v is None:
                return None, "Lo sentimos, por ahora no llegamos a esa zona 😔 Cubrimos: Bosa, Kennedy, Suba, Chapinero, Engativá, Fontibón, Teusaquillo, Usaquén, Barrios Unidos, Puente Aranda, Tunjuelito, Antonio Nariño, Mártires, Soacha y Candelaria"
            return v, None

    # 2. Difusa palabras individuales
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

    # 3. Difusa bigrams
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

# Versión estricta con word boundaries para extracción de nombres
ADDR_START_STRICT = r'\b(?:calle|cll|carrera|avenida|av|transversal|tv|diagonal|dg|autopista|ak)\b'

# Nombres propios colombianos comunes — para detectar nombre de 1 palabra antes de la dirección
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
    # Si todo está en una línea, empezar desde la primera palabra de dirección
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

    # Buscar en texto completo si no encontró
    if not address:
        m = re.search(
            r'(' + ADDR_START + r'\s*[\w\s.\-#]+?\d+[\w\s.\-#]*\d*)',
            text, re.IGNORECASE
        )
        if m:
            address = m.group(1).strip()

    # Deduplicar indicaciones
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
    """
    Extrae nombre del usuario.
    Maneja el caso donde nombre y dirección están en la misma línea.
    Estrategia: buscar texto antes de la primera palabra de dirección.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Múltiples líneas — primera línea que sea solo letras es el nombre
    if len(lines) > 1:
        for line in lines[:2]:
            n = normalize(line)
            if (re.match(r'^[a-záéíóúñ\s]+$', n)
                    and 1 <= len(line.split()) <= 5
                    and not re.search(ADDR_START_STRICT, n)):
                return line.strip().title()

    # Una sola línea o no se encontró — extraer antes de la dirección
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

def extract_delivery(text: str, valid_locations=None, valid_days=None, valid_times=None) -> dict:
    """
    Extrae datos de entrega con máxima robustez.
    Errores descriptivos y empáticos por campo faltante.
    """
    if not text or not text.strip():
        return {"error": True, "errorMessage": "No recibí ningún mensaje 😊", "info": None}

    # Mensaje muy corto
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

    # Nombre
    raw_name = labeled.get('name', '')
    if raw_name:
        nn = normalize(raw_name.split('\n')[0].strip())
        name = raw_name.split('\n')[0].strip() if re.match(r'^[a-záéíóúñ\s]+$', nn) and len(raw_name.split()) <= 5 else extract_name(clean)
    else:
        name = extract_name(clean)

    # Dirección
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

    # Localidad
    locality = locality_err = None
    if labeled.get('locality'):
        locality, locality_err = extract_locality(labeled['locality'])
    if not locality and not locality_err:
        locality, locality_err = extract_locality(clean)

    if locality_err and "no llegamos" in locality_err:
        return {"error": True, "errorMessage": locality_err, "info": None}

    # Día
    day = day_err = None
    if labeled.get('day'):
        day, day_err = extract_day(labeled['day'])
    if not day and not day_err:
        day, day_err = extract_day(clean)

    if not day and day_err:
        return {"error": True, "errorMessage": day_err, "info": None}

    # Errores descriptivos
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
# RESPUESTAS NATURALES — siempre responde algo
# ============================================================

RESPONSES = {
    "saludo": [
        "¡Hola! 😊 Soy Vecinito de CorApp. ¿En qué te puedo ayudar hoy?",
        "¡Hola vecino! 👋 Tenemos frutas y verduras frescas con domicilio gratis. ¿Qué necesitas?",
        "¡Bienvenido a CorApp! 😊 Compra tu mercado fresco directo de la central de abastos. ¿Te ayudo con algo?",
        "¡Quiubo! 😊 Soy Vecinito de CorApp, listo para ayudarte con tu mercado fresco. ¿Qué necesitas hoy?",
        "¡Hola! 👋 Aquí Vecinito de CorApp. Frutas y verduras frescas a tu puerta, ¡gratis el domicilio! ¿Te ayudo?",
    ],
    "consulta_catalogo": [
        "¡Claro! Aquí el catálogo: https://wa.me/c/573124929496 😊 Agrega lo que necesites y sigue los pasos.",
        "El proceso es fácil 😊\n1️⃣ Entra al catálogo: https://wa.me/c/573124929496\n2️⃣ Agrega productos al carrito\n3️⃣ Toca \'Ver carrito\' → \'Realizar pedido\'\n¡Y listo! 🛒",
        "Entra aquí, escoge lo que necesites y sigue los pasos: https://wa.me/c/573124929496 🛒 ¡Yo te ayudo si tienes dudas!",
        "¡Con gusto te ayudo! 😊 Primero entra al catálogo: https://wa.me/c/573124929496 — agrega lo que quieras al carrito y cuando termines tocas \'Ver carrito\' y luego \'Realizar pedido\'.",
    ],
    "consulta_domicilio": [
        "¡El domicilio es completamente gratis! 🚚 ¿En qué localidad estás?",
        "Sí, domicilio gratis a casi toda Bogotá 😊 Cubrimos Kennedy, Suba, Bosa, Chapinero, Engativá, Fontibón, Teusaquillo, Usaquén, Barrios Unidos, Puente Aranda, Tunjuelito, Soacha y más.",
        "¡Gratis y a tu puerta! 🏠 Entregamos entre 7 AM y 12 PM del día que elijas. ¿En qué zona estás?",
        "¡Sí! El envío es cero pesos 😊🚚 Llegamos a casi toda Bogotá. Cuéntame en qué barrio estás.",
    ],
    "hora_pedido": [
        "Entregamos de 7:00 AM a 12:00 PM ☀️ Si pides hoy antes de las 7 PM, el mercado llega mañana en la mañana.",
        "Las entregas son en la mañana 🚚 entre 7 AM y 12 PM del día que elijas. ¿Quieres hacer el pedido ahora?",
        "Si pides hoy, mañana tienes tu mercado fresco en la puerta 😊 El horario de entrega es 7 AM - 12 PM.",
        "¡Rápido! 😊 Si haces el pedido hoy antes de las 7 PM, mañana lo tienes. Las entregas son de 7 AM a 12 PM.",
    ],
    "metodo_pago": [
        "Aceptamos efectivo, Nequi y Bre-b 💳 Todo se paga cuando llega el pedido a tu puerta — sin anticipos.",
        "Puedes pagar en efectivo, Nequi o Bre-b cuando llegue tu pedido 😊 ¡Sin anticipos, sin complicaciones!",
        "El pago es contra entrega 🙌 Efectivo, Nequi o Bre-b. Solo pagas cuando recibas tu mercado.",
        "¡Sin anticipos! 😊 Pagas cuando llegue: efectivo, Nequi o Bre-b. Así de fácil.",
    ],
    "consulta_producto": [
        "¡Tenemos muchos productos frescos! 🥦🍎 Dime cuál buscas y te comparto el enlace directo al catálogo.",
        "Puedes ver todo aquí: https://wa.me/c/573124929496 😊 ¿Qué producto necesitas hoy?",
        "¡Claro! ¿Qué producto buscas? Te comparto el enlace directo para que lo agregues fácil 😊",
        "Tenemos frutas, verduras y mucho más fresquito 🥦🍊 Dime qué necesitas o entra al catálogo: https://wa.me/c/573124929496",
    ],
    "pedido_no_ha_llegado": [
        "Lamento mucho la espera 😔 Ya notifico al equipo ahora mismo para que revisen el estado de tu entrega.",
        "¡Qué pena! Voy a informar al supervisor de inmediato 🚨 Pronto te contactarán para darte una respuesta.",
        "Entiendo tu preocupación 🙏 Ya escalo tu caso al equipo de logística urgente. Gracias por tu paciencia.",
        "Lo siento mucho 😔 Ya informé al equipo. Te contactarán muy pronto para resolver esto.",
    ],
    "queja_servicio": [
        "Lamento mucho lo ocurrido 😔 Tu opinión es muy importante para nosotros. Ya informo al supervisor para darte una solución.",
        "Entiendo tu molestia y me disculpo de verdad 🙏 Tu caso ya está en manos del equipo para resolverlo cuanto antes.",
        "Tienes toda la razón y lo siento mucho 😔 Ya notifiqué al equipo — te contactarán pronto para darte una solución.",
        "Me disculpo por la experiencia 🙏 Eso no debería pasar. Ya informé al equipo para que te contacten urgente.",
    ],
    "consulta_estado_pedido": [
        "Déjame verificar 🔍 ¿Me confirmas el día que programaste la entrega?",
        "Voy a informar al equipo para que te confirmen el estado de tu pedido 📦 ¿Tienes el número de orden?",
        "Ya informo al equipo para que revisen tu pedido 😊 En un momento tendrás respuesta.",
    ],
    "fuera_de_tema": [
        "Por ahora solo manejo ventas de frutas y verduras 🥦 Pero ya informo al equipo tu mensaje. ¿Te ayudo con algo del catálogo? https://wa.me/c/573124929496",
        "Eso está fuera de lo que puedo ayudarte, pero ya informo al equipo 😊 Para compras: https://wa.me/c/573124929496",
        "No tengo esa información, pero ya le aviso a un supervisor 😊 Mientras tanto, ¿te puedo ayudar con algún producto?",
    ],
    "confirmar_pedido": [
        "Para confirmar escribe exactamente la palabra *Confirmar* 😊",
        "Solo escribe *Confirmar* para aprobar tu pedido ✅",
    ],
    "rechazar_pedido": [
        "Para cancelar escribe exactamente la palabra *Rechazar* 😊",
        "Solo escribe *Rechazar* para cancelar tu pedido 😊",
    ],
    "datos_entrega": [
        "¡Perfecto! Déjame procesar esos datos 😊",
        "Recibido, procesando tu información de entrega 📦",
    ],
    "consulta_precio_minimo": [
        "El pedido mínimo es de $100.000 COP 😊 Con eso ya te hacemos el domicilio completamente gratis. ¿Empezamos a armar tu pedido? https://wa.me/c/573124929496",
        "¡Desde $100.000 COP hacemos domicilio gratis! 🛒 Entra al catálogo y arma tu pedido: https://wa.me/c/573124929496",
        "El mínimo son $100.000 😊 ¡Y el domicilio es gratis! ¿Quieres ver qué tenemos disponible? https://wa.me/c/573124929496",
    ],
    "cambiar_pedido": [
        "Para cambiar o agregar productos a tu pedido ya confirmado, necesito que me indiques qué quieres cambiar y notifico al equipo 😊",
        "¡Claro! Dime qué quieres cambiar o agregar y lo notifico al equipo de inmediato 😊",
        "Para modificaciones en pedidos confirmados, ya notifico al equipo con tu solicitud. ¿Qué necesitas cambiar? 😊",
    ],
    "pedir_hablar_humano": [
        "¡Claro! Ya notifico a un asesor para que te atienda personalmente 😊 En un momento te contactarán.",
        "Entendido 😊 Ya informo al supervisor para que te atienda directamente. Pronto te escriben.",
        "¡Con gusto! Ya llamo a un asesor humano para que te ayude 😊 Dame un momento.",
    ],
    "consulta_descuentos": [
        "Por ahora nuestros precios ya son los mejores porque compramos directo en la central de abastos 😊 Sin intermediarios. ¿Quieres ver el catálogo? https://wa.me/c/573124929496",
        "Nuestro mayor descuento es que no tienes intermediarios 😊 Precios de central de abastos directo a tu casa. Catálogo: https://wa.me/c/573124929496",
        "Los precios que ves en el catálogo ya son los mejores del mercado 😊 ¡Directo de la central de abastos! https://wa.me/c/573124929496",
    ],
    "agradecimiento": [
        "¡Con mucho gusto! 😊 Fue un placer ayudarte. Cuando quieras hacer otro pedido, aquí estoy: https://wa.me/c/573124929496",
        "¡A ti! 😊 Si necesitas algo más no dudes en escribir. ¡Buen provecho con tu mercado! 🥦🍎",
        "¡De nada! 😊 Espero que disfrutes mucho tu mercado fresquito. Cuando quieras volver, aquí estaré.",
        "¡Con gusto vecino! 😊 Recuerda que puedes pedir cuando quieras. El catálogo siempre está disponible: https://wa.me/c/573124929496",
    ],
    "despedida": [
        "¡Hasta luego! 😊 Fue un placer atenderte. Cuando quieras tu mercado fresquito, aquí estaré.",
        "¡Chao! 😊 Espero verte pronto. Recuerda que el catálogo siempre está disponible: https://wa.me/c/573124929496",
        "¡Hasta pronto! 😊 Que tengas un excelente día. Cuando necesites mercado, aquí estamos.",
    ],
    "pensar_luego": [
        "¡Claro, tómate tu tiempo! 😊 Cuando estés listo el catálogo te espera: https://wa.me/c/573124929496 Aquí estaré para ayudarte.",
        "¡Sin problema! 😊 Cuando quieras hacer el pedido, solo entra aquí: https://wa.me/c/573124929496 Te espero.",
        "¡Perfecto! 😊 No hay afán. Cuando decidas, entra al catálogo: https://wa.me/c/573124929496 ¡Aquí estaré!",
    ],
    "festivos_horario_especial": [
        "Los domingos y festivos no hacemos entregas 😊 Nuestros días disponibles son: Lunes, Martes, Miércoles, Jueves, Viernes y Sábado.",
        "Trabajamos de Lunes a Sábado 😊 Los domingos y festivos no hay entregas. ¿Quieres programar para un día disponible?",
        "No hay entregas los domingos ni festivos 😊 Puedes elegir entre Lunes a Sábado. ¿Te ayudo a hacer el pedido? https://wa.me/c/573124929496",
    ],
    "dejar_vecino_porteria": [
        "¡Claro! Puedes dejar las instrucciones en el campo de indicaciones al enviar tus datos 😊 Por ejemplo: \'Dejar con el portero\' o \'Dejar con vecino del 201\'.",
        "Sí, el domiciliario sigue las instrucciones que dejes 😊 Solo indícalo cuando envíes tu dirección: \'Dejar en portería\' o \'Dejar con vecino\' y listo.",
        "¡Por supuesto! 😊 Cuando envíes tus datos de entrega, agrega en indicaciones: \'Si no hay nadie, dejar con portero\' y el domiciliario lo tendrá en cuenta.",
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


def find_products_by_query(query: str, products: list) -> list:
    """
    Busca productos:
    1. Primero por categoría — si encuentra productos en esa categoría los retorna todos
    2. Si no hay coincidencias por categoría, busca por nombre de producto
    """
    if not products or not query:
        return []

    query_norm = normalize(query)
    query_words = [w for w in query_norm.split() if len(w) >= 3]

    # ============================================================
    # PASO 1: Buscar por categoría
    # ============================================================
    category_matches = []
    for product in products:
        category_norm = normalize(str(product.get("category", "")))
        if not category_norm:
            continue

        # Coincidencia exacta de palabra
        for word in query_words:
            if word in category_norm or category_norm in word:
                category_matches.append(product)
                break
        else:
            # Similitud fuzzy con la categoría
            cat_score = max(
                levenshtein(query_norm, category_norm),
                jaro_winkler(query_norm, category_norm),
            )
            # También probar cada palabra del query contra la categoría
            word_score = max(
                (max(levenshtein(w, category_norm), jaro_winkler(w, category_norm))
                 for w in query_words),
                default=0
            )
            best = max(cat_score, word_score)
            if best >= 0.75:
                category_matches.append(product)

    if category_matches:
        return category_matches[:5]  # Retornar hasta 5 productos de la categoría

    # ============================================================
    # PASO 2: Si no hay resultados por categoría, buscar por nombre
    # ============================================================
    name_matches = []
    for product in products:
        name_norm = normalize(str(product.get("name", "")))

        for word in query_words:
            if word in name_norm:
                name_matches.append((product, 0.9))
                break
        else:
            name_score = max(
                levenshtein(query_norm, name_norm),
                jaro_winkler(query_norm, name_norm),
            )
            word_score = max(
                (max(levenshtein(w, name_norm), jaro_winkler(w, name_norm))
                 for w in query_words),
                default=0
            )
            best = max(name_score, word_score)
            if best >= 0.72:
                name_matches.append((product, best))

    name_matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in name_matches[:3]]


def build_tenant_response(
    intent: str,
    confidence: float,
    message: str,
    tenant: dict,
    products: list,
) -> str:
    """
    Genera una respuesta natural y contextualizada para el tenant.
    """
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

    # Detectar si el mensaje tiene URL de producto específico
    product_from_url = extract_product_from_url(message, products)
    if product_from_url:
        matching_products = [product_from_url]
    else:
        matching_products = find_products_by_query(message, products)

    if intent == "saludo":
        opts = [
            f"Hola! Soy {bot_name} de {business_name}. Tenemos {biz['products_label']} {bus_emoji} y {delivery_text}. En que te puedo ayudar hoy?",
            f"Hola! Bienvenido a {business_name}. Soy {bot_name}, tu asistente virtual. Que estas buscando hoy? {bus_emoji}",
            f"Hola! Soy {bot_name} de {business_name}. Estoy aqui para ayudarte con {biz['products_label']}. Que necesitas?",
        ]
        return random.choice(opts)

    elif intent == "consulta_producto":
        if matching_products:
            # Si viene de URL — respuesta detallada de un solo producto
            if product_from_url and len(matching_products) == 1:
                p = matching_products[0]
                price = p.get("price", 0)
                name = p.get("name", "Producto")
                description = p.get("description", "")
                wa_url = p.get("whatsapp_url", "")
                price_fmt = f"${int(float(price)):,}".replace(",", ".") if price else ""
                response = f"{bus_emoji} *{name}*"
                if price_fmt:
                    response += f"\nPrecio: {price_fmt}"
                if description:
                    response += f"\n\n{description}"
                if wa_url:
                    response += f"\n\nVer en catalogo: {wa_url}"
                response += "\n\n¿Te gustaria pedirlo? 😊"
                return response
            # Respuesta normal — lista de productos
            response = f"Claro! {bus_emoji} Encontre estos productos que te pueden interesar:\n\n"
            for p in matching_products:
                price = p.get("price", 0)
                wa_url = p.get("whatsapp_url", "")
                name = p.get("name", "Producto")
                description = p.get("description", "")
                price_fmt = f"${int(float(price)):,}".replace(",", ".") if price else ""
                line = f"*{name}*"
                if price_fmt:
                    line += f" - {price_fmt}"
                if description:
                    line += f"\n  {description}"
                if wa_url:
                    line += f"\n  Ver producto: {wa_url}"
                response += line + "\n\n"
            response += "Te interesa alguno? 😊"
            return response
        elif catalog_url:
            opts = [
                f"Claro! {bus_emoji} Puedes {biz['catalog_verb']} aqui: {catalog_url} Hay algo especifico que estes buscando?",
                f"Tenemos varios {biz['products_label']} disponibles {bus_emoji} Entra al catalogo: {catalog_url}",
            ]
            return random.choice(opts)
        return f"Claro! {bus_emoji} Tenemos varios {biz['products_label']} disponibles. Que estas buscando exactamente?"

    elif intent == "consulta_catalogo":
        if catalog_url:
            opts = [
                f"Con gusto! Aqui tienes el catalogo de {business_name}: {catalog_url} {bus_emoji} Agrega lo que necesites.",
                f"Puedes ver todos nuestros {biz['products_label']} aqui: {catalog_url} {bus_emoji} Te ayudo con algo especifico?",
            ]
            return random.choice(opts)
        return f"Escribeme que {biz['products_label']} buscas y te ayudo de inmediato {bus_emoji}"

    elif intent == "consulta_domicilio":
        if delivery_type == "punto_fisico":
            return f"Puedes recoger tu pedido en nuestro punto fisico. Quieres mas informacion?"
        opts = [
            f"Si! {delivery_text.capitalize()} Los dias de entrega son {days_text}. Quieres hacer un pedido?",
            f"Claro! {delivery_text.capitalize()} Dias disponibles: {days_text}. En que zona estas?",
        ]
        return random.choice(opts)

    elif intent == "hora_pedido":
        opts = [
            f"Las entregas son en horas de la manana entre 7 AM y 12 PM del dia que elijas. Tenemos disponibilidad los {days_text}.",
            f"Entregamos de 7 AM a 12 PM los dias {days_text}. Para cuando quieres tu pedido?",
        ]
        return random.choice(opts)

    elif intent == "metodo_pago":
        opts = [
            f"Aceptamos efectivo, Nequi y Bre-b El pago es al recibir tu pedido - sin anticipos!",
            f"Puedes pagar en efectivo, Nequi o Bre-b cuando llegue tu pedido Sin anticipos!",
        ]
        return random.choice(opts)

    elif intent == "pedido_no_ha_llegado":
        opts = [
            f"Lamento mucho la espera Ya informo al equipo de {business_name} ahora mismo para que revisen tu entrega.",
            f"Que pena! Voy a notificar al equipo de inmediato. Pronto te contactaran para resolver esto.",
        ]
        return random.choice(opts)

    elif intent == "queja_servicio":
        opts = [
            f"Lamento mucho lo ocurrido Tu comentario es muy importante para {business_name}. Ya informo al equipo para darte una solucion.",
            f"Entiendo tu molestia y me disculpo Ya escalo tu caso al equipo para que te contacten pronto.",
        ]
        return random.choice(opts)

    elif intent == "consulta_estado_pedido":
        return f"Dejame verificar Me confirmas tu nombre y el dia que programaste la entrega?"

    elif intent == "agradecimiento":
        opts = [
            f"Con mucho gusto! Fue un placer ayudarte. Cuando quieras hacer otro pedido, aqui estamos en {business_name}.",
            f"A ti! Si necesitas algo mas, no dudes en escribir. {bus_emoji}",
        ]
        return random.choice(opts)

    elif intent == "despedida":
        opts = [
            f"Hasta luego! Fue un placer atenderte en {business_name}. Cuando quieras volver, aqui estaremos.",
            f"Chao! Espero verte pronto. Recuerda que el catalogo siempre esta disponible: {catalog_url}",
        ]
        return random.choice(opts)

    elif intent == "consulta_precio_minimo":
        return f"El pedido minimo es de $100.000 COP Y el domicilio es completamente gratis! Empezamos? {catalog_url}"

    elif intent == "pedir_hablar_humano":
        return f"Claro! Ya notifico a un asesor de {business_name} para que te atienda personalmente. En un momento te contactaran."

    elif intent == "pensar_luego":
        opts = [
            f"Sin problema! Cuando estes listo el catalogo te espera: {catalog_url} Aqui estare!",
            f"Perfecto! No hay afan. Cuando decidas, entra aqui: {catalog_url} Te espero!",
        ]
        return random.choice(opts)

    elif intent == "festivos_horario_especial":
        return f"Trabajamos {days_text} Los domingos y festivos no hay entregas. Quieres programar para un dia disponible?"

    elif intent == "fuera_de_tema":
        opts = [
            f"Por ahora solo manejo lo de {business_name} Te puedo ayudar con algun {biz['products_label']}? {catalog_url}",
            f"Eso esta fuera de lo que puedo ayudarte, pero ya informo al equipo Para compras: {catalog_url}",
        ]
        return random.choice(opts)

    # Fallback con productos si hay
    if matching_products and confidence > 0.1:
        response = f"Mira estos productos que tenemos {bus_emoji}\n\n"
        for p in matching_products:
            wa_url = p.get("whatsapp_url", "")
            name = p.get("name", "Producto")
            line = f"- {name}"
            if wa_url:
                line += f"\n  {wa_url}"
            response += line + "\n\n"
        response += "Te interesa alguno? 😊"
        return response

    opts = [
        f"Hola! Soy {bot_name} de {business_name}. En que te puedo ayudar? {catalog_url}",
        f"Entiendo En que te puedo ayudar? {bus_emoji}",
        f"Aqui estoy para ayudarte! Que necesitas de {business_name}?",
    ]
    return random.choice(opts)



# ============================================================
# DICCIONARIO DE CONTEXTO POR TIPO DE NEGOCIO
# ============================================================

BUSINESS_CONTEXT: dict[str, list[str]] = {
    "muebles": [
        "cama", "camas", "sofa", "sofas", "silla", "sillas", "mesa", "mesas",
        "closet", "closets", "armario", "armarios", "comoda", "comodas",
        "escritorio", "escritorios", "estante", "estantes", "biblioteca",
        "biblioteca", "buro", "buros", "camarote", "camarotes", "litera",
        "literas", "colchon", "colchones", "sala", "alcoba", "habitacion",
        "comedor", "mueble", "muebles", "madera", "tapizado", "espejo",
        "espejos", "tocador", "tocadores", "alacena", "alacenas", "vitrina",
        "vitrinas", "rinconera", "rinconeras", "baul", "baules", "puff",
        "taburete", "taburetes", "mecedora", "mecedoras", "poltrona",
        "poltronas", "sillon", "sillones", "zapatera", "zapateras",
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


def get_business_context_words(business_type: str) -> list[str]:
    """Retorna las palabras de contexto para un tipo de negocio."""
    return BUSINESS_CONTEXT.get(business_type, [])


def extract_product_from_url(message: str, products: list) -> dict | None:
    """
    Detecta si el mensaje contiene una URL de producto de WhatsApp
    y retorna el producto correspondiente.
    URLs como: https://wa.me/p/PRODUCT_ID/NUMBER
    """
    import re
    # Buscar patrón de URL de producto de WhatsApp
    match = re.search(r'wa\.me/p/([^/\s]+)', message)
    if not match:
        return None
    product_id = match.group(1)
    for product in products:
        wa_url = product.get("whatsapp_url", "")
        if product_id in wa_url:
            return product
    return None


def is_product_query(message: str, products: list, business_type: str = "") -> bool:
    """
    Detecta si el mensaje es una consulta de producto o categoría.
    1. Detecta URLs de productos de WhatsApp
    2. Verifica contra el diccionario de contexto del tipo de negocio
    3. Verifica contra los productos reales del tenant
    """
    # URL de producto → siempre es consulta de producto
    if extract_product_from_url(message, products):
        return True

    msg_norm = normalize(message)
    msg_words = [w for w in msg_norm.split() if len(w) >= 4]

    if not msg_words:
        return False

    # Verificar contra diccionario de contexto del tipo de negocio
    # Esto va ANTES de verificar productos — da contexto general del negocio
    if business_type:
        context_words = set(get_business_context_words(business_type))
        for word in msg_words:
            # Coincidencia exacta con el diccionario
            if word in context_words:
                return True
            # Coincidencia parcial estricta — ambas palabras >= 6 letras
            for cw in context_words:
                if len(cw) >= 7 and len(word) >= 7 and (word in cw or cw in word):
                    return True

    for product in products:
        # Verificar contra categoría
        cat_norm = normalize(str(product.get("category", "")))
        if cat_norm:
            for word in msg_words:
                if word in cat_norm or cat_norm in word:
                    return True
            if max((jaro_winkler(w, cat_norm) for w in msg_words), default=0) >= 0.85:
                return True

        # Verificar contra nombre — cada palabra del nombre
        name_norm = normalize(str(product.get("name", "")))
        name_words = [w for w in name_norm.split() if len(w) >= 4]
        for word in msg_words:
            for nw in name_words:
                if word == nw or (len(word) >= 5 and word in name_norm):
                    return True
        if max((jaro_winkler(w, name_norm) for w in msg_words), default=0) >= 0.85:
            return True

    return False


@app.route('/respond-tenant', methods=['POST'])
def respond_tenant():
    """
    Genera respuesta contextualizada para un tenant específico.
    Recibe: message, tenant (config), products (lista de productos)
    """
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
        })

    # Si hay match en diccionario de contexto o productos reales → consulta_producto directo
    # Si no hay match → el clasificador decide, pero si clasifica como consulta_producto igual busca productos
    if is_product_query(message, products, tenant.get("business_type", "")):
        intent, confidence = 'consulta_producto', 0.99
    else:
        try:
            intent, confidence = classify(message)
        except Exception:
            intent, confidence = 'unknown', 0.0

    response = build_tenant_response(intent, confidence, message, tenant, products)

    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': round(confidence, 4),
    })


@app.route('/extract-delivery', methods=['POST'])
def extract_delivery_endpoint():
    """
    Extrae datos de entrega de un mensaje de texto.
    Usado por el bot de tenants para procesar datos de envio.
    """
    data = request.get_json() or {}
    message = data.get('message', '')

    if not message:
        return jsonify({"success": False, "error": "Mensaje vacio"})

    result = extract_delivery(message)
    info = result.get('info', {})

    # Extraer dirección del texto si el ML no la encontró estructuralmente
    address = info.get('address') if info else None
    if not address and not result.get('error'):
        address = message.strip()

    # Solo fallamos si no hay absolutamente nada
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

# v5.0.0 — modelo completo con fuzzy search, aliases colombianos,
# respuestas empáticas y análisis de fallos
if __name__ == '__main__':
    app.run(debug=True, port=5000)