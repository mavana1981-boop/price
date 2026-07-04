import os, re, json, base64, logging, requests
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO

from flask import (Flask, render_template_string, request, redirect,
                   url_for, flash, jsonify, session)
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler

# ── Config ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'priceland-dev-key-2026')

_raw_db_url = os.environ.get('DATABASE_URL', '')
if _raw_db_url:
    # Normaliza para pg8000 (driver puro Python, sem libpq)
    DATABASE_URL = re.sub(r'^postgres(ql)?://', 'postgresql+pg8000://', _raw_db_url)
    # pg8000 no Railway: desabilita SSL via connect_args
    _USE_PG = True
else:
    DATABASE_URL = 'sqlite:///priceland.db'
    _USE_PG = False
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
if _USE_PG:
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'connect_args': {'ssl_context': None},
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GROQ_API_KEY   = os.environ.get('GROQ_API_KEY', '')
SERPER_API_KEY = os.environ.get('SERPER_API_KEY', '')
CF_ACCOUNT_ID  = os.environ.get('CF_ACCOUNT_ID', '')
CF_API_TOKEN   = os.environ.get('CF_API_TOKEN', '')

# ── Cadeia de IA — mesmo padrão do Pauta Plenário ──────────────────────────
GEMINI_MODEL = 'gemini-2.0-flash'
GEMINI_PREFERENCIA = [
    'gemini-2.5-flash', 'gemini-2.5-flash-lite',
    'gemini-2.0-flash', 'gemini-2.0-flash-lite',
    'gemini-1.5-flash', 'gemini-1.5-flash-8b',
]
_gemini_modelo_cache = {'modelo': None}

def detectar_modelo_gemini(key):
    if _gemini_modelo_cache['modelo']:
        return _gemini_modelo_cache['modelo']
    try:
        r = requests.get(
            'https://generativelanguage.googleapis.com/v1beta/models?key=' + key,
            timeout=8)
        if r.ok:
            disponiveis = [m['name'].split('/')[-1] for m in r.json().get('models',[])
                           if 'generateContent' in m.get('supportedGenerationMethods',[])]
            for pref in GEMINI_PREFERENCIA:
                if pref in disponiveis:
                    _gemini_modelo_cache['modelo'] = pref
                    logger.info(f'Gemini selecionado: {pref}')
                    return pref
            if disponiveis:
                _gemini_modelo_cache['modelo'] = disponiveis[0]
                return disponiveis[0]
    except Exception as e:
        logger.warning(f'detectar_modelo_gemini: {e}')
    return GEMINI_MODEL

def gemini_post(prompt, max_tokens=1500, temperatura=0.3, tentativas=3):
    import time
    key = os.environ.get('GEMINI_API_KEY', '')
    if not key: raise Exception('GEMINI_API_KEY não configurada')
    modelo = detectar_modelo_gemini(key)
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={key}'
    payload = {'contents':[{'parts':[{'text':prompt}]}],
               'generationConfig':{'maxOutputTokens':max_tokens,'temperature':temperatura}}
    for i in range(tentativas):
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 429:
                time.sleep(5*(i+1)); continue
            if r.status_code == 404:
                _gemini_modelo_cache['modelo'] = None
                modelo = detectar_modelo_gemini(key)
                url = f'https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={key}'
                continue
            r.raise_for_status()
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429 and i < tentativas-1:
                time.sleep(5*(i+1)); continue
            raise
    raise Exception('Gemini indisponível após retries')

def gemini_vision_post(image_bytes, prompt, mime_type='image/jpeg'):
    key = os.environ.get('GEMINI_API_KEY', '')
    if not key: raise Exception('GEMINI_API_KEY não configurada')
    modelo = detectar_modelo_gemini(key)
    img_b64 = base64.b64encode(image_bytes).decode()
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={key}'
    body = {'contents':[{'parts':[
        {'inline_data':{'mime_type':mime_type,'data':img_b64}},
        {'text':prompt}
    ]}],'generationConfig':{'maxOutputTokens':2000,'temperature':0.2}}
    r = requests.post(url, json=body, timeout=30)
    r.raise_for_status()
    return r.json()['candidates'][0]['content']['parts'][0]['text']

def groq_post(prompt, max_tokens=1500, temperatura=0.3):
    key = os.environ.get('GROQ_API_KEY', '')
    if not key: raise Exception('GROQ_API_KEY não configurada')
    r = requests.post('https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},
        json={'model':'llama-3.3-70b-versatile',
              'messages':[{'role':'user','content':prompt}],
              'max_tokens':max_tokens,'temperature':temperatura},
        timeout=30)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

def cloudflare_post(prompt, max_tokens=1500, temperatura=0.3):
    acc = os.environ.get('CF_ACCOUNT_ID','')
    tok = os.environ.get('CF_API_TOKEN','')
    if not acc or not tok: raise Exception('CF não configurado')
    url = f'https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/meta/llama-3.1-70b-instruct'
    r = requests.post(url,
        headers={'Authorization':f'Bearer {tok}','Content-Type':'application/json'},
        json={'messages':[{'role':'user','content':prompt}],
              'max_tokens':max_tokens,'temperature':temperatura},
        timeout=30)
    r.raise_for_status()
    return r.json()['result']['response']

def ia_chain(prompt, max_tokens=1500, temperatura=0.3, contexto=''):
    """Cadeia tripla Groq → Gemini → Cloudflare. Mesmo padrão do Pauta Plenário."""
    erros = []

    # 1. Groq (rápido, gratuito)
    try:
        texto = groq_post(prompt, max_tokens=max_tokens, temperatura=temperatura)
        if texto and texto.strip():
            logger.info(f'ia_chain [{contexto}]: Groq OK')
            return texto, 'groq'
    except Exception as e:
        logger.warning(f'ia_chain [{contexto}]: Groq falhou — {e}')
        erros.append(f'Groq: {e}')

    # 2. Gemini (fallback)
    try:
        texto = gemini_post(prompt, max_tokens=max_tokens, temperatura=temperatura, tentativas=2)
        if texto and texto.strip():
            logger.info(f'ia_chain [{contexto}]: Gemini OK')
            return texto, 'gemini'
    except Exception as e:
        logger.warning(f'ia_chain [{contexto}]: Gemini falhou — {e}')
        _gemini_modelo_cache['modelo'] = None
        erros.append(f'Gemini: {e}')

    # 3. Cloudflare (último recurso)
    try:
        texto = cloudflare_post(prompt, max_tokens=max_tokens, temperatura=temperatura)
        if texto and texto.strip():
            logger.info(f'ia_chain [{contexto}]: Cloudflare OK')
            return texto, 'cloudflare'
    except Exception as e:
        logger.warning(f'ia_chain [{contexto}]: Cloudflare falhou — {e}')
        erros.append(f'Cloudflare: {e}')

    raise Exception(f'Todas as IAs falharam: {"; ".join(erros)}')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ── Models ──────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    baskets  = db.relationship('Basket', backref='owner', lazy=True)

class Basket(db.Model):
    __tablename__ = 'baskets'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name       = db.Column(db.String(120), nullable=False, default='Minha Cesta')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    items      = db.relationship('BasketItem', backref='basket',
                                  lazy=True, cascade='all, delete-orphan')

class BasketItem(db.Model):
    __tablename__ = 'basket_items'
    id          = db.Column(db.Integer, primary_key=True)
    basket_id   = db.Column(db.Integer, db.ForeignKey('baskets.id'), nullable=False)
    name        = db.Column(db.String(200), nullable=False)
    category    = db.Column(db.String(80), default='')
    unit        = db.Column(db.String(20), default='un')
    qty         = db.Column(db.Float, default=1.0)
    target_price= db.Column(db.Float, nullable=True)  # preço alvo do usuário
    prices      = db.relationship('PriceRecord', backref='item',
                                   lazy=True, cascade='all, delete-orphan',
                                   order_by='PriceRecord.captured_at.desc()')

class PriceRecord(db.Model):
    __tablename__ = 'price_records'
    id          = db.Column(db.Integer, primary_key=True)
    item_id     = db.Column(db.Integer, db.ForeignKey('basket_items.id'), nullable=False)
    price       = db.Column(db.Float, nullable=False)
    store       = db.Column(db.String(200), default='')
    url         = db.Column(db.String(500), default='')
    source      = db.Column(db.String(50), default='manual')  # manual/ia/busca
    captured_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes       = db.Column(db.Text, default='')

@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

# ── IA helpers ──────────────────────────────────────────────────────────────
# gemini_vision e groq_chat substituídos por ia_chain e gemini_vision_post acima

def buscar_precos_web(produto):
    """Busca precos via Gemini com grounding (acesso web) ou estimativa IA."""
    resultados = []

    # 1. Tenta Gemini com Google Search grounding (acessa web real)
    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    if gemini_key:
        try:
            modelo = detectar_modelo_gemini(gemini_key)
            url = f'https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={gemini_key}'
            prompt = (f'Pesquise o preco atual de "{produto}" em supermercados brasileiros. '
                      f'Liste ate 5 resultados com preco, loja e link. '
                      f'Responda APENAS com JSON valido sem markdown: '
                      f'{{"resultados":[{{"preco":22.90,"loja":"Extra","titulo":"{produto}","url":""}}]}}')
            body = {
                'contents': [{'parts': [{'text': prompt}]}],
                'tools': [{'google_search': {}}],
                'generationConfig': {'maxOutputTokens': 500, 'temperature': 0.1}
            }
            r = requests.post(url, json=body, timeout=20)
            if r.ok:
                texto = r.json()['candidates'][0]['content']['parts'][0]['text']
                clean = re.sub(r'```.*?```', '', texto, flags=re.DOTALL).strip()
                # Tenta parsear JSON
                m = re.search(r'\{.*\}', clean, re.DOTALL)
                if m:
                    j = json.loads(m.group())
                    for it in j.get('resultados', [])[:5]:
                        p = float(it.get('preco', 0))
                        if p > 0:
                            resultados.append({
                                'price': p,
                                'store': it.get('loja', ''),
                                'url':   it.get('url', ''),
                                'title': it.get('titulo', produto),
                            })
                logger.info(f'gemini_grounding: {len(resultados)} resultados')
        except Exception as e:
            logger.warning(f'gemini_grounding: {e}')

    # 2. Fallback: estimativa IA via ia_chain
    if not resultados:
        try:
            prompt = (f'Qual o preco medio de "{produto}" em supermercados brasileiros hoje em julho 2026? '
                      f'Responda APENAS com JSON valido sem markdown: '
                      f'{{"preco_medio":5.90,"variacao_min":4.50,"variacao_max":7.20,"exemplos":[{{"loja":"Extra","preco":5.90}},{{"loja":"Atacadao","preco":4.50}}]}}')
            resp, fonte = ia_chain(prompt, max_tokens=300, temperatura=0.2, contexto='busca_preco')
            clean = re.sub(r'```.*?```', '', resp, flags=re.DOTALL).strip()
            j = json.loads(clean)
            # Usa exemplos se disponíveis
            for ex in j.get('exemplos', [])[:3]:
                p = float(ex.get('preco', 0))
                if p > 0:
                    resultados.append({
                        'price': p,
                        'store': f'{ex.get("loja","Supermercado")} (estimativa {fonte})',
                        'url': '', 'title': produto, 'notes': 'Estimativa IA'
                    })
            if not resultados:
                p = float(j.get('preco_medio', 0))
                if p > 0:
                    resultados.append({
                        'price': p,
                        'store': f'Estimativa IA ({fonte})',
                        'url': '', 'title': produto,
                        'notes': f'Variacao: R$ {j.get("variacao_min",0):.2f} - R$ {j.get("variacao_max",0):.2f}'
                    })
        except Exception as e:
            logger.warning(f'ia_chain busca_preco: {e}')

    return resultados

def job_buscar_precos():
    with app.app_context():
        try:
            # Busca itens sem preço registrado nas últimas 24h
            cutoff = datetime.utcnow() - timedelta(hours=24)
            itens = BasketItem.query.all()
            for item in itens:
                ultimo = (PriceRecord.query
                          .filter_by(item_id=item.id)
                          .filter(PriceRecord.source.in_(['busca','ia']))
                          .filter(PriceRecord.captured_at >= cutoff)
                          .first())
                if not ultimo:
                    resultados = buscar_precos_web(item.name)
                    for r in resultados[:2]:
                        if r.get('price', 0) > 0:
                            pr = PriceRecord(
                                item_id=item.id, price=r['price'],
                                store=r.get('store',''), url=r.get('url',''),
                                source='busca', notes=r.get('notes','')
                            )
                            db.session.add(pr)
            db.session.commit()
            logger.info('job_buscar_precos: concluído')
        except Exception as e:
            logger.error(f'job_buscar_precos: {e}')

scheduler = BackgroundScheduler()
scheduler.add_job(job_buscar_precos, 'interval', hours=6, id='buscar_precos')
scheduler.start()

# ── Template ─────────────────────────────────────────────────────────────────
HTML = r'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Priceland — Cesta Inteligente</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
<style>
:root{
  --verde:#0F7B3C;--verde-claro:#E8F5EE;--verde-med:#2DAA6B;
  --laranja:#F47B20;--laranja-claro:#FFF3E9;
  --cinza-escuro:#1A1A1A;--cinza:#4A4A4A;--cinza-claro:#F7F7F5;
  --borda:#E2E2DE;--branco:#FFFFFF;
  --fonte-display:'Syne',sans-serif;--fonte-corpo:'Inter',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--fonte-corpo);background:var(--cinza-claro);color:var(--cinza-escuro);min-height:100vh}

/* NAVBAR */
nav{background:var(--cinza-escuro);padding:0 24px;display:flex;align-items:center;justify-content:space-between;height:56px;position:sticky;top:0;z-index:100}
.nav-logo{font-family:var(--fonte-display);font-size:1.3rem;font-weight:800;color:var(--branco);letter-spacing:-0.5px}
.nav-logo span{color:var(--verde-med)}
.nav-links a{color:rgba(255,255,255,.7);text-decoration:none;font-size:.85rem;font-weight:500;margin-left:20px;transition:color .2s}
.nav-links a:hover{color:var(--branco)}
.nav-links .btn-sair{background:rgba(255,255,255,.1);padding:6px 14px;border-radius:6px}

/* HERO LOGIN */
.hero-login{min-height:100vh;background:var(--cinza-escuro);display:flex;align-items:center;justify-content:center;padding:20px}
.login-card{background:var(--branco);border-radius:16px;padding:40px;width:100%;max-width:400px;box-shadow:0 20px 60px rgba(0,0,0,.2)}
.login-logo{font-family:var(--fonte-display);font-size:2rem;font-weight:800;color:var(--cinza-escuro);margin-bottom:4px}
.login-logo span{color:var(--verde-med)}
.login-sub{color:var(--cinza);font-size:.9rem;margin-bottom:28px}
.form-group{margin-bottom:16px}
.form-group label{display:block;font-size:.8rem;font-weight:600;color:var(--cinza);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}
.form-group input{width:100%;padding:11px 14px;border:1.5px solid var(--borda);border-radius:8px;font-family:var(--fonte-corpo);font-size:.95rem;transition:border-color .2s}
.form-group input:focus{outline:none;border-color:var(--verde)}
.btn-primary{width:100%;padding:12px;background:var(--verde);color:#fff;border:none;border-radius:8px;font-family:var(--fonte-display);font-size:1rem;font-weight:700;cursor:pointer;transition:background .2s;letter-spacing:.3px}
.btn-primary:hover{background:#0a6030}
.btn-secondary{padding:8px 16px;background:transparent;color:var(--verde);border:1.5px solid var(--verde);border-radius:8px;font-size:.85rem;font-weight:600;cursor:pointer;transition:all .2s}
.btn-secondary:hover{background:var(--verde-claro)}
.btn-danger{padding:6px 12px;background:transparent;color:#c0392b;border:1.5px solid #c0392b;border-radius:6px;font-size:.8rem;cursor:pointer}
.login-switch{text-align:center;margin-top:16px;font-size:.85rem;color:var(--cinza)}
.login-switch a{color:var(--verde);text-decoration:none;font-weight:600}
.flash-msg{padding:10px 14px;border-radius:8px;margin-bottom:16px;font-size:.88rem;background:#FEF3CD;border:1px solid #F0C030;color:#7A5C00}

/* LAYOUT PRINCIPAL */
.container{max-width:1100px;margin:0 auto;padding:24px 20px}
.page-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;flex-wrap:gap}
.page-title{font-family:var(--fonte-display);font-size:1.6rem;font-weight:700}
.page-title span{color:var(--verde)}

/* STATS STRIP */
.stats-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}
.stat-card{background:var(--branco);border-radius:12px;padding:16px;border:1px solid var(--borda)}
.stat-val{font-family:var(--fonte-display);font-size:1.6rem;font-weight:700;color:var(--verde)}
.stat-label{font-size:.75rem;color:var(--cinza);margin-top:2px;font-weight:500}

/* CESTA */
.basket-grid{display:grid;grid-template-columns:1fr 360px;gap:20px;align-items:start}
@media(max-width:768px){.basket-grid{grid-template-columns:1fr}}

.card{background:var(--branco);border-radius:12px;border:1px solid var(--borda);overflow:hidden}
.card-header{padding:14px 18px;border-bottom:1px solid var(--borda);display:flex;align-items:center;justify-content:space-between}
.card-title{font-family:var(--fonte-display);font-size:.95rem;font-weight:700}

/* ITENS */
.item-row{padding:14px 18px;border-bottom:1px solid var(--borda);display:flex;align-items:flex-start;gap:12px}
.item-row:last-child{border-bottom:none}
.item-icon{width:36px;height:36px;background:var(--verde-claro);border-radius:8px;display:flex;align-items:center;justify-content:center;color:var(--verde);font-size:.9rem;flex-shrink:0}
.item-name{font-weight:600;font-size:.92rem;margin-bottom:2px}
.item-meta{font-size:.77rem;color:var(--cinza)}
.item-price-best{font-family:var(--fonte-display);font-size:1.1rem;font-weight:700;color:var(--verde);margin-left:auto;text-align:right;flex-shrink:0}
.item-price-store{font-size:.72rem;color:var(--cinza);margin-top:1px}
.price-badge{display:inline-block;font-size:.7rem;font-weight:700;padding:2px 7px;border-radius:4px;margin-left:6px}
.badge-ok{background:var(--verde-claro);color:var(--verde)}
.badge-alto{background:#FFEDED;color:#c0392b}
.badge-novo{background:var(--laranja-claro);color:var(--laranja)}

/* FORMULÁRIO ADICIONAR */
.add-form{padding:16px 18px}
.row-2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.form-control{width:100%;padding:9px 12px;border:1.5px solid var(--borda);border-radius:8px;font-family:var(--fonte-corpo);font-size:.88rem;transition:border-color .2s}
.form-control:focus{outline:none;border-color:var(--verde)}
.form-label{font-size:.75rem;font-weight:600;color:var(--cinza);display:block;margin-bottom:4px;text-transform:uppercase;letter-spacing:.4px}
.mt-2{margin-top:10px}

/* FOTO UPLOAD */
.upload-zone{border:2px dashed var(--borda);border-radius:10px;padding:24px;text-align:center;cursor:pointer;transition:all .2s;background:var(--cinza-claro)}
.upload-zone:hover{border-color:var(--verde);background:var(--verde-claro)}
.upload-zone i{font-size:1.8rem;color:var(--cinza);margin-bottom:8px;display:block}
.upload-zone p{font-size:.82rem;color:var(--cinza)}
.upload-zone input{display:none}

/* PRECOS HISTORICO */
.price-hist{padding:14px 18px}
.ph-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--borda);font-size:.83rem}
.ph-row:last-child{border-bottom:none}
.ph-store{font-weight:500}
.ph-price{font-family:var(--fonte-display);font-weight:700;color:var(--cinza-escuro)}
.ph-meta{font-size:.72rem;color:var(--cinza)}
.source-tag{font-size:.65rem;padding:2px 5px;border-radius:3px;font-weight:600}
.src-manual{background:#E8E8E8;color:var(--cinza)}
.src-ia{background:#E8F0FE;color:#1A73E8}
.src-busca{background:var(--verde-claro);color:var(--verde)}

/* ALERTA META */
.meta-alert{padding:10px 14px;background:var(--laranja-claro);border:1px solid var(--laranja);border-radius:8px;font-size:.82rem;color:#7A3D00;margin:10px 18px;display:flex;align-items:center;gap:8px}

/* MODAL */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:500;align-items:center;justify-content:center;padding:20px}
.modal-bg.open{display:flex}
.modal{background:var(--branco);border-radius:16px;width:100%;max-width:520px;padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.25)}
.modal-title{font-family:var(--fonte-display);font-size:1.1rem;font-weight:700;margin-bottom:16px}
.modal-footer{display:flex;gap:10px;justify-content:flex-end;margin-top:20px}

/* SPINNER */
.spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* TOAST */
.toast{position:fixed;bottom:20px;right:20px;background:var(--cinza-escuro);color:#fff;padding:12px 20px;border-radius:10px;font-size:.88rem;z-index:999;opacity:0;transform:translateY(10px);transition:all .3s}
.toast.show{opacity:1;transform:translateY(0)}

/* BARRA OTIMA */
.otima-strip{background:linear-gradient(135deg,var(--verde),var(--verde-med));border-radius:12px;padding:20px;color:#fff;margin-bottom:20px}
.otima-strip h3{font-family:var(--fonte-display);font-size:1.1rem;font-weight:700;margin-bottom:4px}
.otima-strip p{font-size:.83rem;opacity:.85}
.otima-total{font-family:var(--fonte-display);font-size:2rem;font-weight:800;margin-top:8px}

/* TAB */
.tabs{display:flex;gap:0;border-bottom:2px solid var(--borda);margin-bottom:20px}
.tab{padding:10px 20px;font-size:.88rem;font-weight:600;color:var(--cinza);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .2s}
.tab.active{color:var(--verde);border-bottom-color:var(--verde)}
.tab-content{display:none}
.tab-content.active{display:block}

/* EMPTY STATE */
.empty{text-align:center;padding:40px 20px;color:var(--cinza)}
.empty i{font-size:2.5rem;margin-bottom:12px;display:block;color:var(--borda)}
.empty p{font-size:.9rem}
</style>
</head>
<body>
{% if not current_user.is_authenticated %}
<!-- ── LOGIN / REGISTER ── -->
<div class="hero-login">
  <div class="login-card">
    <div class="login-logo">Price<span>land</span></div>
    <p class="login-sub">Cesta inteligente — compare e economize</p>
    {% for msg in get_flashed_messages() %}<div class="flash-msg">{{ msg }}</div>{% endfor %}
    {% if mode == 'register' %}
    <form method="POST" action="/register">
      <div class="form-group"><label>Usuário</label><input name="username" required></div>
      <div class="form-group"><label>Senha</label><input type="password" name="password" required></div>
      <button class="btn-primary" type="submit">Criar conta</button>
    </form>
    <div class="login-switch">Já tem conta? <a href="/login">Entrar</a></div>
    {% else %}
    <form method="POST" action="/login">
      <div class="form-group"><label>Usuário</label><input name="username" required autofocus></div>
      <div class="form-group"><label>Senha</label><input type="password" name="password" required></div>
      <button class="btn-primary" type="submit">Entrar</button>
    </form>
    <div class="login-switch">Não tem conta? <a href="/register">Criar</a></div>
    {% endif %}
  </div>
</div>

{% else %}
<!-- ── APP PRINCIPAL ── -->
<nav>
  <div class="nav-logo">Price<span>land</span></div>
  <div class="nav-links">
    <span style="color:rgba(255,255,255,.5);font-size:.82rem">{{ current_user.username }}</span>
    <a href="/logout" class="btn-sair">Sair</a>
  </div>
</nav>

<div class="container">
  <!-- Stats -->
  <div class="stats-strip">
    <div class="stat-card">
      <div class="stat-val">{{ stats.total_itens }}</div>
      <div class="stat-label">Itens na cesta</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">R$ {{ "%.2f"|format(stats.total_melhor) }}</div>
      <div class="stat-label">Total cesta ótima</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{{ stats.itens_com_preco }}</div>
      <div class="stat-label">Com preço encontrado</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{{ stats.economia }}%</div>
      <div class="stat-label">Economia vs. alvo</div>
    </div>
  </div>

  {% if stats.total_melhor > 0 %}
  <div class="otima-strip">
    <h3><i class="fas fa-trophy" style="margin-right:6px"></i>Cesta Ótima</h3>
    <p>Melhor combinação de lojas encontrada para seus itens</p>
    <div class="otima-total">R$ {{ "%.2f"|format(stats.total_melhor) }}</div>
  </div>
  {% endif %}

  <div class="basket-grid">
    <!-- Lista de itens -->
    <div>
      <div class="tabs">
        <div class="tab active" onclick="switchTab('cesta')">🛒 Cesta</div>
        <div class="tab" onclick="switchTab('foto')">📷 Enviar Foto</div>
        <div class="tab" onclick="switchTab('historico')">📊 Histórico</div>
      </div>

      <!-- TAB CESTA -->
      <div class="tab-content active" id="tab-cesta">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Itens da Cesta</span>
            <button class="btn-secondary" onclick="openModal('modal-add')">
              <i class="fas fa-plus" style="margin-right:5px"></i>Adicionar
            </button>
          </div>
          {% if basket.items %}
            {% for item in basket.items %}
            {% set melhor = item.prices|selectattr('source','!=','estimativa')|sort(attribute='price')|first if item.prices else None %}
            {% set melhor = item.prices|sort(attribute='price')|first if not melhor and item.prices else melhor %}
            <div class="item-row" id="item-{{ item.id }}">
              <div class="item-icon"><i class="fas fa-tag"></i></div>
              <div style="flex:1;min-width:0">
                <div class="item-name">{{ item.name }}
                  {% if item.target_price and melhor and melhor.price <= item.target_price %}
                    <span class="price-badge badge-ok">✓ na meta</span>
                  {% elif item.target_price and melhor and melhor.price > item.target_price %}
                    <span class="price-badge badge-alto">acima da meta</span>
                  {% endif %}
                </div>
                <div class="item-meta">{{ item.qty }} {{ item.unit }} · {{ item.category or 'Geral' }}</div>
                {% if item.target_price %}
                <div class="item-meta">Meta: R$ {{ "%.2f"|format(item.target_price) }}</div>
                {% endif %}
              </div>
              {% if melhor %}
              <div class="item-price-best">
                R$ {{ "%.2f"|format(melhor.price) }}
                <div class="item-price-store">{{ melhor.store[:25] if melhor.store else '—' }}</div>
              </div>
              {% else %}
              <div class="item-price-best" style="color:var(--cinza);font-size:.8rem">sem preço</div>
              {% endif %}
              <div style="display:flex;flex-direction:column;gap:4px;flex-shrink:0">
                <button class="btn-secondary" style="font-size:.75rem;padding:5px 8px"
                  onclick="openPrecos({{ item.id }}, '{{ item.name|replace("'","\\'")|e }}')">
                  <i class="fas fa-search-dollar"></i>
                </button>
                <button class="btn-danger" style="font-size:.75rem;padding:5px 8px"
                  onclick="deletarItem({{ item.id }})">
                  <i class="fas fa-trash"></i>
                </button>
              </div>
            </div>
            {% endfor %}
          {% else %}
          <div class="empty">
            <i class="fas fa-shopping-cart"></i>
            <p>Sua cesta está vazia.<br>Adicione itens para começar.</p>
          </div>
          {% endif %}
        </div>
      </div>

      <!-- TAB FOTO -->
      <div class="tab-content" id="tab-foto">
        <div class="card">
          <div class="card-header"><span class="card-title">Enviar Foto de Prateleira ou Panfleto</span></div>
          <div style="padding:18px">
            <p style="font-size:.85rem;color:var(--cinza);margin-bottom:14px">
              Tire uma foto de preços no supermercado ou de um panfleto de ofertas. 
              A IA vai extrair os produtos e preços automaticamente.
            </p>
            <div class="upload-zone" onclick="document.getElementById('foto-input').click()">
              <i class="fas fa-camera-retro"></i>
              <p>Clique para enviar uma foto<br><small style="opacity:.6">JPG, PNG ou WEBP · máx 10MB</small></p>
              <input type="file" id="foto-input" accept="image/*" onchange="analisarFoto(this)">
            </div>
            <div id="foto-preview" style="display:none;margin-top:14px">
              <img id="foto-img" style="max-width:100%;border-radius:8px;margin-bottom:10px">
              <div id="foto-resultado" style="font-size:.85rem"></div>
            </div>
            <div id="foto-loading" style="display:none;text-align:center;padding:20px;color:var(--cinza)">
              <i class="fas fa-spinner fa-spin" style="font-size:1.5rem;margin-bottom:8px;display:block;color:var(--verde)"></i>
              Analisando imagem com IA...
            </div>
          </div>
        </div>
      </div>

      <!-- TAB HISTORICO -->
      <div class="tab-content" id="tab-historico">
        <div class="card">
          <div class="card-header"><span class="card-title">Histórico de Preços</span></div>
          {% if basket.items %}
            {% for item in basket.items %}
            {% if item.prices %}
            <div style="padding:10px 18px 0">
              <div style="font-size:.85rem;font-weight:600">{{ item.name }}</div>
            </div>
            <div class="price-hist">
              {% for pr in item.prices[:5] %}
              <div class="ph-row">
                <div>
                  <div class="ph-store">{{ pr.store or 'Manual' }}</div>
                  <div class="ph-meta">{{ pr.captured_at.strftime('%d/%m/%Y %H:%M') }}
                    <span class="source-tag src-{{ pr.source }}">{{ pr.source }}</span>
                  </div>
                </div>
                <div class="ph-price">R$ {{ "%.2f"|format(pr.price) }}</div>
              </div>
              {% endfor %}
            </div>
            {% endif %}
            {% endfor %}
          {% else %}
          <div class="empty"><i class="fas fa-chart-line"></i><p>Nenhum histórico ainda.</p></div>
          {% endif %}
        </div>
      </div>
    </div>

    <!-- Painel lateral -->
    <div>
      <!-- Busca rápida de preços -->
      <div class="card" style="margin-bottom:16px">
        <div class="card-header"><span class="card-title">🔍 Buscar Preços Agora</span></div>
        <div class="add-form">
          <p style="font-size:.8rem;color:var(--cinza);margin-bottom:12px">
            Busca preços atualizados para um produto específico via IA e web.
          </p>
          <div class="form-group" style="margin-bottom:0">
            <label class="form-label">Produto</label>
            <input class="form-control" id="busca-produto" placeholder="Ex: arroz 5kg">
          </div>
          <button class="btn-primary" style="margin-top:10px" onclick="buscarPrecosManual()">
            <i class="fas fa-search" style="margin-right:6px"></i>Buscar
          </button>
          <div id="busca-resultado" style="margin-top:12px;font-size:.83rem"></div>
        </div>
      </div>

      <!-- Info scheduler -->
      <div class="card">
        <div class="card-header"><span class="card-title">⏰ Atualização Automática</span></div>
        <div style="padding:14px 18px;font-size:.82rem;color:var(--cinza)">
          <p>O sistema busca preços atualizados automaticamente <strong>a cada 6 horas</strong> para todos os itens da cesta.</p>
          <button class="btn-secondary" style="margin-top:12px;width:100%" onclick="buscarTodos()">
            <i class="fas fa-sync" style="margin-right:5px"></i>Atualizar todos agora
          </button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- MODAL ADICIONAR ITEM -->
<div class="modal-bg" id="modal-add">
  <div class="modal">
    <div class="modal-title"><i class="fas fa-plus-circle" style="color:var(--verde);margin-right:8px"></i>Adicionar Item</div>
    <div class="form-group">
      <label class="form-label">Nome do Produto *</label>
      <input class="form-control" id="add-nome" placeholder="Ex: Arroz Integral 5kg">
    </div>
    <div class="row-2">
      <div class="form-group">
        <label class="form-label">Categoria</label>
        <select class="form-control" id="add-cat">
          <option value="">Geral</option>
          <option>Alimentos</option><option>Bebidas</option><option>Limpeza</option>
          <option>Higiene</option><option>Hortifruti</option><option>Laticínios</option>
          <option>Carnes</option><option>Padaria</option><option>Outros</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Unidade</label>
        <select class="form-control" id="add-unit">
          <option>un</option><option>kg</option><option>g</option>
          <option>L</option><option>ml</option><option>pct</option><option>cx</option>
        </select>
      </div>
    </div>
    <div class="row-2">
      <div class="form-group">
        <label class="form-label">Quantidade</label>
        <input class="form-control" type="number" id="add-qty" value="1" min="0.1" step="0.1">
      </div>
      <div class="form-group">
        <label class="form-label">Preço alvo (R$)</label>
        <input class="form-control" type="number" id="add-target" placeholder="Opcional" step="0.01">
      </div>
    </div>
    <div class="form-group">
      <label class="form-label">Preço atual que você conhece (R$)</label>
      <input class="form-control" type="number" id="add-preco" placeholder="Opcional" step="0.01">
    </div>
    <div id="add-loading" style="display:none;font-size:.82rem;color:var(--verde)">
      <i class="fas fa-spinner fa-spin"></i> Buscando preços...
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal('modal-add')">Cancelar</button>
      <button class="btn-primary" style="width:auto;padding:10px 24px" onclick="adicionarItem()">
        <i class="fas fa-plus" style="margin-right:6px"></i>Adicionar
      </button>
    </div>
  </div>
</div>

<!-- MODAL PRECOS ITEM -->
<div class="modal-bg" id="modal-precos">
  <div class="modal">
    <div class="modal-title" id="modal-precos-title">Preços</div>
    <div id="modal-precos-content"></div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeModal('modal-precos')">Fechar</button>
      <button class="btn-primary" style="width:auto;padding:10px 20px" id="btn-buscar-item-precos">
        <i class="fas fa-search" style="margin-right:6px"></i>Buscar preços
      </button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
// ── Utilitários ──────────────────────────────────────────────────────────
function showToast(msg, ok=true){
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.style.background=ok?'#0F7B3C':'#c0392b';
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3000);
}
function openModal(id){document.getElementById(id).classList.add('open');}
function closeModal(id){document.getElementById(id).classList.remove('open');}
function switchTab(name){
  document.querySelectorAll('.tab,.tab-content').forEach(el=>{
    el.classList.remove('active');
  });
  document.querySelectorAll('.tab').forEach((t,i)=>{
    const names=['cesta','foto','historico'];
    if(names[i]===name) t.classList.add('active');
  });
  document.getElementById('tab-'+name).classList.add('active');
}

// ── Adicionar Item ───────────────────────────────────────────────────────
async function adicionarItem(){
  const nome=document.getElementById('add-nome').value.trim();
  if(!nome){alert('Digite o nome do produto.');return;}
  const loading=document.getElementById('add-loading');
  loading.style.display='block';
  try{
    const r=await fetch('/api/item',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        name:nome,category:document.getElementById('add-cat').value,
        unit:document.getElementById('add-unit').value,
        qty:parseFloat(document.getElementById('add-qty').value)||1,
        target_price:parseFloat(document.getElementById('add-target').value)||null,
        price:parseFloat(document.getElementById('add-preco').value)||null,
      })
    });
    const j=await r.json();
    if(j.ok){showToast('Item adicionado!');closeModal('modal-add');location.reload();}
    else showToast(j.error||'Erro',false);
  }catch(e){showToast('Erro de rede',false);}
  loading.style.display='none';
}

// ── Deletar Item ─────────────────────────────────────────────────────────
async function deletarItem(id){
  if(!confirm('Remover item da cesta?')) return;
  const r=await fetch(`/api/item/${id}`,{method:'DELETE'});
  const j=await r.json();
  if(j.ok){showToast('Removido');location.reload();}
  else showToast(j.error||'Erro',false);
}

// ── Buscar Preços (item específico) ──────────────────────────────────────
let _buscandoItemId=null;
function openPrecos(id, nome){
  _buscandoItemId=id;
  document.getElementById('modal-precos-title').textContent='Preços: '+nome;
  const btn=document.getElementById('btn-buscar-item-precos');
  btn.onclick=()=>buscarPrecosItem(id,nome);
  // Carrega preços já registrados
  fetch(`/api/item/${id}/prices`).then(r=>r.json()).then(j=>{
    const el=document.getElementById('modal-precos-content');
    if(!j.prices||!j.prices.length){
      el.innerHTML='<p style="color:#888;font-size:.85rem;padding:10px 0">Nenhum preço registrado ainda.</p>';
    } else {
      el.innerHTML=j.prices.map(p=>`
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee;font-size:.85rem">
          <div>
            <strong>${p.store||'Manual'}</strong>
            <div style="color:#888;font-size:.75rem">${p.captured_at} · <span class="source-tag src-${p.source}">${p.source}</span></div>
            ${p.notes?`<div style="color:#888;font-size:.75rem">${p.notes}</div>`:''}
          </div>
          <strong>R$ ${p.price.toFixed(2)}</strong>
        </div>`).join('');
    }
    // Formulário para adicionar preço manual
    el.innerHTML+=`
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid #eee">
        <div style="font-size:.78rem;font-weight:600;color:#888;text-transform:uppercase;margin-bottom:8px">Registrar preço manual</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <input class="form-control" id="mp-preco" type="number" placeholder="R$ preço" step="0.01">
          <input class="form-control" id="mp-loja" placeholder="Loja">
        </div>
        <button class="btn-secondary" style="margin-top:8px;width:100%" onclick="registrarPrecoManual(${id})">
          Registrar
        </button>
      </div>`;
  });
  openModal('modal-precos');
}

async function buscarPrecosItem(id, nome){
  const btn=document.getElementById('btn-buscar-item-precos');
  btn.disabled=true;btn.innerHTML='<span class="spinner"></span>';
  try{
    const r=await fetch(`/api/item/${id}/buscar`,{method:'POST'});
    const j=await r.json();
    if(j.ok){showToast(`${j.count} preço(s) encontrado(s)`);closeModal('modal-precos');location.reload();}
    else showToast(j.error||'Nenhum preço encontrado',false);
  }catch(e){showToast('Erro',false);}
  btn.disabled=false;btn.innerHTML='<i class="fas fa-search" style="margin-right:6px"></i>Buscar preços';
}

async function registrarPrecoManual(id){
  const preco=parseFloat(document.getElementById('mp-preco').value);
  const loja=document.getElementById('mp-loja').value.trim();
  if(!preco||preco<=0){alert('Digite um preço válido.');return;}
  const r=await fetch(`/api/item/${id}/price`,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({price:preco,store:loja,source:'manual'})});
  const j=await r.json();
  if(j.ok){showToast('Preço registrado');closeModal('modal-precos');location.reload();}
  else showToast(j.error||'Erro',false);
}

// ── Busca Rápida (painel lateral) ────────────────────────────────────────
async function buscarPrecosManual(){
  const produto=document.getElementById('busca-produto').value.trim();
  if(!produto) return;
  const div=document.getElementById('busca-resultado');
  div.innerHTML='<i class="fas fa-spinner fa-spin" style="color:var(--verde)"></i> Buscando...';
  try{
    const r=await fetch('/api/buscar_produto',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({produto})});
    const j=await r.json();
    if(j.resultados&&j.resultados.length){
      div.innerHTML=j.resultados.map(p=>`
        <div style="padding:6px 0;border-bottom:1px solid #eee">
          <div style="font-weight:600;font-size:.82rem">${p.title||produto}</div>
          <div style="display:flex;justify-content:space-between;font-size:.8rem">
            <span style="color:#888">${p.store||'—'}</span>
            <strong style="color:var(--verde)">R$ ${p.price.toFixed(2)}</strong>
          </div>
          ${p.notes?`<div style="font-size:.73rem;color:#888">${p.notes}</div>`:''}
        </div>`).join('');
    } else {
      div.innerHTML='<span style="color:#888">Nenhum resultado encontrado.</span>';
    }
  }catch(e){div.innerHTML='<span style="color:#c0392b">Erro na busca.</span>';}
}

// ── Atualizar Todos ──────────────────────────────────────────────────────
async function buscarTodos(){
  showToast('Buscando preços para todos os itens...');
  const r=await fetch('/api/buscar_todos',{method:'POST'});
  const j=await r.json();
  if(j.ok){showToast(`Atualizado: ${j.count} preço(s) encontrado(s)`);setTimeout(()=>location.reload(),1500);}
  else showToast(j.error||'Erro',false);
}

// ── Analisar Foto ────────────────────────────────────────────────────────
async function analisarFoto(input){
  if(!input.files[0]) return;
  const file=input.files[0];
  const reader=new FileReader();
  reader.onload=async(e)=>{
    document.getElementById('foto-img').src=e.target.result;
    document.getElementById('foto-preview').style.display='block';
    document.getElementById('foto-loading').style.display='block';
    document.getElementById('foto-resultado').innerHTML='';
    const formData=new FormData();
    formData.append('foto',file);
    try{
      const r=await fetch('/api/analisar_foto',{method:'POST',body:formData});
      const j=await r.json();
      document.getElementById('foto-loading').style.display='none';
      if(j.produtos&&j.produtos.length){
        document.getElementById('foto-resultado').innerHTML=`
          <div style="font-weight:600;margin-bottom:10px;color:var(--verde)">
            <i class="fas fa-check-circle"></i> ${j.produtos.length} produto(s) identificado(s)
          </div>`+
          j.produtos.map(p=>`
            <div style="display:flex;align-items:center;justify-content:space-between;padding:8px;background:var(--cinza-claro);border-radius:8px;margin-bottom:6px">
              <div>
                <div style="font-weight:600;font-size:.88rem">${p.nome}</div>
                <div style="font-size:.75rem;color:#888">${p.loja||'Local'}</div>
              </div>
              <div style="display:flex;align-items:center;gap:8px">
                <strong style="color:var(--verde)">R$ ${parseFloat(p.preco).toFixed(2)}</strong>
                <button class="btn-secondary" style="font-size:.75rem;padding:4px 10px"
                  onclick="adicionarDaFoto('${p.nome.replace(/'/g,"\\'")}',${p.preco})">
                  + Cesta
                </button>
              </div>
            </div>`).join('');
      } else {
        document.getElementById('foto-resultado').innerHTML=
          '<span style="color:#888;font-size:.85rem">Não foi possível identificar preços na imagem.</span>';
      }
    }catch(err){
      document.getElementById('foto-loading').style.display='none';
      document.getElementById('foto-resultado').innerHTML=
        '<span style="color:#c0392b;font-size:.85rem">Erro ao analisar imagem.</span>';
    }
  };
  reader.readAsDataURL(file);
}

async function adicionarDaFoto(nome, preco){
  const r=await fetch('/api/item',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:nome,price:preco,source:'ia'})});
  const j=await r.json();
  if(j.ok){showToast('Item adicionado à cesta!');location.reload();}
  else showToast(j.error||'Erro',false);
}

// Fecha modal clicando fora
document.querySelectorAll('.modal-bg').forEach(bg=>{
  bg.addEventListener('click',e=>{if(e.target===bg)bg.classList.remove('open');});
});
</script>
{% endif %}
</body>
</html>'''

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if not current_user.is_authenticated:
        return render_template_string(HTML, mode='login')
    basket = Basket.query.filter_by(user_id=current_user.id).first()
    if not basket:
        basket = Basket(user_id=current_user.id, name='Minha Cesta')
        db.session.add(basket); db.session.commit()
    # Stats
    total_melhor = 0
    itens_com_preco = 0
    for item in basket.items:
        if item.prices:
            itens_com_preco += 1
            melhor = min(item.prices, key=lambda p: p.price)
            total_melhor += melhor.price * item.qty
    economia = 0
    if basket.items:
        total_alvo = sum((it.target_price or 0)*it.qty for it in basket.items if it.target_price)
        if total_alvo > 0 and total_melhor > 0:
            economia = round(max(0, (1 - total_melhor/total_alvo)*100))
    stats = dict(total_itens=len(basket.items), total_melhor=total_melhor,
                 itens_com_preco=itens_com_preco, economia=economia)
    return render_template_string(HTML, basket=basket, stats=stats)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = User.query.filter_by(username=request.form['username']).first()
        if u and check_password_hash(u.password, request.form['password']):
            login_user(u); return redirect('/')
        flash('Usuário ou senha incorretos')
    return render_template_string(HTML, mode='login')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        if User.query.filter_by(username=username).first():
            flash('Usuário já existe')
            return render_template_string(HTML, mode='register')
        u = User(username=username,
                 password=generate_password_hash(request.form['password']))
        db.session.add(u); db.session.commit()
        login_user(u); return redirect('/')
    return render_template_string(HTML, mode='register')

@app.route('/logout')
@login_required
def logout():
    logout_user(); return redirect('/login')

# ── API routes ───────────────────────────────────────────────────────────────
def get_basket():
    b = Basket.query.filter_by(user_id=current_user.id).first()
    if not b:
        b = Basket(user_id=current_user.id)
        db.session.add(b); db.session.commit()
    return b

@app.route('/api/item', methods=['POST'])
@login_required
def api_add_item():
    data = request.get_json()
    basket = get_basket()
    item = BasketItem(
        basket_id=basket.id,
        name=data.get('name',''),
        category=data.get('category',''),
        unit=data.get('unit','un'),
        qty=float(data.get('qty',1)),
        target_price=data.get('target_price'),
    )
    db.session.add(item); db.session.flush()
    # Preço inicial se informado
    preco = data.get('price')
    if preco:
        pr = PriceRecord(item_id=item.id, price=float(preco),
                         store='Informado', source=data.get('source','manual'))
        db.session.add(pr)
    db.session.commit()
    # Busca preços automática em background (sem bloquear)
    try:
        resultados = buscar_precos_web(item.name)
        for r in resultados[:3]:
            if r.get('price',0) > 0:
                pr2 = PriceRecord(item_id=item.id, price=r['price'],
                    store=r.get('store',''), url=r.get('url',''),
                    source='busca', notes=r.get('notes',''))
                db.session.add(pr2)
        db.session.commit()
    except Exception as e:
        logger.warning(f'busca auto: {e}')
    return jsonify({'ok': True, 'id': item.id})

@app.route('/api/item/<int:item_id>', methods=['DELETE'])
@login_required
def api_delete_item(item_id):
    item = BasketItem.query.get_or_404(item_id)
    if item.basket.user_id != current_user.id:
        return jsonify({'error':'Sem permissão'}), 403
    db.session.delete(item); db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/item/<int:item_id>/prices')
@login_required
def api_item_prices(item_id):
    item = BasketItem.query.get_or_404(item_id)
    if item.basket.user_id != current_user.id:
        return jsonify({'error':'Sem permissão'}), 403
    return jsonify({'prices':[{
        'price': p.price, 'store': p.store, 'url': p.url,
        'source': p.source, 'notes': p.notes,
        'captured_at': p.captured_at.strftime('%d/%m/%Y %H:%M')
    } for p in item.prices[:20]]})

@app.route('/api/item/<int:item_id>/price', methods=['POST'])
@login_required
def api_add_price(item_id):
    item = BasketItem.query.get_or_404(item_id)
    if item.basket.user_id != current_user.id:
        return jsonify({'error':'Sem permissão'}), 403
    data = request.get_json()
    pr = PriceRecord(item_id=item_id, price=float(data['price']),
                     store=data.get('store',''), source=data.get('source','manual'))
    db.session.add(pr); db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/item/<int:item_id>/buscar', methods=['POST'])
@login_required
def api_buscar_item(item_id):
    item = BasketItem.query.get_or_404(item_id)
    if item.basket.user_id != current_user.id:
        return jsonify({'error':'Sem permissão'}), 403
    resultados = buscar_precos_web(item.name)
    count = 0
    for r in resultados:
        if r.get('price',0) > 0:
            pr = PriceRecord(item_id=item_id, price=r['price'],
                store=r.get('store',''), url=r.get('url',''),
                source='busca', notes=r.get('notes',''))
            db.session.add(pr); count += 1
    db.session.commit()
    return jsonify({'ok': count > 0, 'count': count,
                    'error': 'Nenhum preço encontrado' if count == 0 else None})

@app.route('/api/buscar_produto', methods=['POST'])
@login_required
def api_buscar_produto():
    data = request.get_json()
    produto = data.get('produto','').strip()
    if not produto:
        return jsonify({'resultados':[]})
    resultados = buscar_precos_web(produto)
    return jsonify({'resultados': resultados})

@app.route('/api/buscar_todos', methods=['POST'])
@login_required
def api_buscar_todos():
    basket = get_basket()
    count = 0
    for item in basket.items:
        resultados = buscar_precos_web(item.name)
        for r in resultados[:2]:
            if r.get('price',0) > 0:
                pr = PriceRecord(item_id=item.id, price=r['price'],
                    store=r.get('store',''), url=r.get('url',''),
                    source='busca', notes=r.get('notes',''))
                db.session.add(pr); count += 1
    db.session.commit()
    return jsonify({'ok': True, 'count': count})

@app.route('/api/analisar_foto', methods=['POST'])
@login_required
def api_analisar_foto():
    if 'foto' not in request.files:
        return jsonify({'error':'Nenhuma foto enviada'}), 400
    foto = request.files['foto']
    # Reduz imagem para economizar tokens
    img = Image.open(foto.stream)
    img.thumbnail((1200, 1200), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, 'JPEG', quality=85)
    img_bytes = buf.getvalue()

    prompt = """Analise esta imagem de prateleira de supermercado ou panfleto de ofertas.
Extraia TODOS os produtos e preços visíveis.
Responda SOMENTE com JSON válido neste formato:
{"produtos": [{"nome": "Arroz Tipo 1 5kg", "preco": 22.90, "loja": "Supermercado X"}, ...]}
Se não encontrar produtos com preço, responda: {"produtos": []}
Seja preciso com os preços. Inclua a marca quando visível."""

    resultado = None
    try:
        resultado = gemini_vision_post(img_bytes, prompt)
    except Exception as e:
        logger.warning(f'gemini_vision_post falhou: {e}')
        # Groq não tem visão — tenta descrever o que pode com ia_chain
        try:
            resultado_txt, _ = ia_chain(
                f'Liste produtos típicos de supermercado com preços aproximados em BRL, '
                f'em formato JSON: {{"produtos":[{{"nome":"..","preco":0.0,"loja":""}}]}}',
                max_tokens=500, contexto='foto_fallback')
            resultado = resultado_txt
        except Exception:
            pass
    if not resultado:
        return jsonify({'produtos': [], 'error': 'IA de visão não disponível'})

    try:
        clean = re.sub(r'```(?:json)?|```', '', resultado).strip()
        j = json.loads(clean)
        return jsonify(j)
    except Exception:
        return jsonify({'produtos': [], 'raw': resultado[:500]})

@app.errorhandler(404)
def e404(e): return jsonify({'error':'Não encontrado'}), 404

@app.errorhandler(500)
def e500(e):
    logger.error(f'500: {e}')
    return jsonify({'error':'Erro interno'}), 500

# ── Init DB — roda sempre ao importar (Gunicorn não executa __main__) ────────
def init_db():
    with app.app_context():
        db.create_all()
        logger.info('DB inicializado')

# Inicializa imediatamente ao importar o módulo
init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=False)
