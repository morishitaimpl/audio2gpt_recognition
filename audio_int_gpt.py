import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
from openai import OpenAI
import requests
import base64
import json
import io
import os
import re
import time
from dotenv import load_dotenv
from urllib.parse import quote
from typing import Optional

# ==========================
# .env 読み込み & API設定
# ==========================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# ==========================
# 設定
# ==========================
DURATION = 5  # 録音時間（秒）
SAMPLE_RATE = 44000 #(Hz)

def record_audio(duration: int = DURATION, fs: int = SAMPLE_RATE):
    st.info("🎙️ 録音中...静かにしてお待ちください（5秒間）")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    st.success("✅ 録音完了！")
    return audio

# ==========================
# JSON抽出ユーティリティ
# ==========================
def _extract_json(text: str):
    if not isinstance(text, str):
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return None

# ==========================
# GPT-4o 音声解析
# ==========================
def analyze_bird_sound(audio_bytes: bytes):
    st.info("🧠 GPT-4oで解析中...")

    prompt = """
あなたは鳥の鳴き声識別の専門家です。入力音声のスペクトル的特徴を厳密に観察し、その鳴き声に該当する鳥種を推定します。
必ず以下の手順で内省し、最後の出力は指定のJSONのみを返してください。

[利用できる手掛かり（与えられた場合のみ使う）]
- 地域（国/都道府県/緯度帯）、生息環境（森林/水辺/都市公園/農地）
- 季節・月・時間帯、録音長、録音機材、背景雑音の種類
- 既知の出現頻度（地域・季節ごとの一般的な出現傾向）

[観察・分析手順（簡潔な内部チェックリスト）]
1) 時間—周波数特徴：主成分帯域(kHz)、倍音有無、ホイッスル/雑音系、周波数変化（上昇/下降/トリル/プラトー）、 syllable反復速度(回/秒)。
2) 時系列パターン：呼び数、間隔、リズム、フレーズ長。
3) ノイズ評価：SNR/風・人声・車音の混入。短すぎる/断片的なら低信頼。
4) 候補の絞り込み：上記特徴と地域・季節の整合性でTop3を内部で比較（ただし出力は最良1種のみ）。
5) 信頼度の較正：音質・持続時間・パターン一致度から0.0–1.0で数値化。閾値 <0.5 なら species を "unknown" とする。

[注意事項]
- 推測での命名をしない。「聞き分けが困難」な場合は unknown を許可。
- JSON 以外の文字・説明・コードブロックやバッククォートを出力しない。
- 学名/和名の揺れは一般的な和名に正規化する（例：ウグイス）。
- 同定困難時は、どの追加情報（地域/季節/長めの録音など）が有用かを description に簡潔に記す。


余計な文章は出さないでください。

{
  "species": "鳥の日本語名",
  "confidence": 0.0〜1.0の信頼度,
  "description": "その鳥の特徴を日本語で簡潔に説明"
}
"""

    try:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        resp = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=300,
        )

        content = resp.choices[0].message.content
        if isinstance(content, list):
            parts = [p.get("text") for p in content if isinstance(p, dict) and "text" in p]
            content = parts[0] if parts else ""

        result = _extract_json(content)
        if not result:
            raise ValueError("モデル出力からJSONを抽出できません: " + str(content))

        species = result.get("species", "").strip()
        confidence = float(result.get("confidence", 0.0))
        description = result.get("description", "").strip()

        return {"species": species, "confidence": confidence, "description": description}

    except Exception as e:
        st.error(f"❌ GPT解析中にエラーが発生しました: {e}")
        return None

# ==========================
# Wikipedia画像検索
# ==========================
def get_wikipedia_image(species_name: str) -> Optional[str]:
    """
    Wikipedia REST APIからサムネイル画像URLを取得する（ja優先→enフォールバック）。
    ・種名はそのまま使用し、各言語のSummary APIに問い合わせ
    ・サムネイル→オリジナル画像の順に採用
    ・見つからない場合は None を返す
    """
    def _request_with_retries(url: str, headers: dict, retries: int = 2, timeout: float = 8.0) -> Optional[requests.Response]:
        for attempt in range(retries + 1):
            try:
                res = requests.get(url, headers=headers, timeout=timeout)
                if res.status_code == 200:
                    return res
            except Exception:
                pass
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
        return None

    def _get_from_summary(title: str, lang: str, headers: dict) -> Optional[str]:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
        res = _request_with_retries(url, headers)
        if not res:
            return None
        try:
            data = res.json()
        except Exception:
            return None
        thumb = (data.get("thumbnail") or {}).get("source")
        if thumb:
            return thumb
        original = (data.get("originalimage") or {}).get("source")
        return original

    def _get_from_media_list(title: str, lang: str, headers: dict) -> Optional[str]:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/media-list/{quote(title, safe='')}"
        res = _request_with_retries(url, headers)
        if not res:
            return None
        try:
            data = res.json()
        except Exception:
            return None
        items = data.get("items") or []
        # 優先: original source がある画像 → 最大解像度のsrcset
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                continue
            original = (item.get("original") or {}).get("source")
            if original:
                return original
            srcset = item.get("srcset") or []
            best = None
            best_w = -1
            for s in srcset:
                src = s.get("src")
                width = s.get("width") or 0
                if src and isinstance(width, int) and width > best_w:
                    best = src
                    best_w = width
            if best:
                return best
        return None

    def _get_image_from_lang(title: str, lang: str) -> Optional[str]:
        headers = {
            "User-Agent": "audio_to_gpt/1.0 (+https://example.com)"
        }
        # 1) summary → 2) media-list の順で試す
        img = _get_from_summary(title, lang, headers)
        if img:
            return img
        return _get_from_media_list(title, lang, headers)

    # 日本語→英語の順で探索
    for lang in ("ja", "en"):
        img = _get_image_from_lang(species_name, lang)
        if img:
            return img
    return None

# ==========================
# DuckDuckGo画像検索（フォールバック）
# ==========================
def get_duckduckgo_image(query: str) -> Optional[str]:
    """
    DuckDuckGoの非公開画像APIを利用して、最初の画像URLを取得する。
    1) 検索ページから vqd トークンを取得
    2) i.js? 画像エンドポイントで結果を取得
    取得できない場合は None
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        }
        # 1) vqd取得
        q = quote(query, safe="")
        search_url = f"https://duckduckgo.com/?q={q}&iax=images&ia=images"
        r1 = requests.get(search_url, headers=headers, timeout=8)
        if r1.status_code != 200:
            return None

        # vqd をHTMLから抽出（複数のパターンを試す）
        m = re.search(r"vqd=['\"]([\w-]+)['\"]", r1.text)
        if not m:
            # 代替パターン
            m = re.search(r'\"vqd\":\"([\w-]+)\"', r1.text)
        if not m:
            return None
        vqd = m.group(1)

        # 2) 画像JSON取得
        api_url = f"https://duckduckgo.com/i.js?l=ja-jp&o=json&q={q}&vqd={vqd}&f=,,,&p=1"
        r2 = requests.get(api_url, headers=headers, timeout=8)
        if r2.status_code != 200:
            return None
        data = r2.json()
        results = data.get("results") or []
        if not results:
            return None
        # 最初の画像URL
        first = results[0]
        image_url = first.get("image") or first.get("thumbnail")
        return image_url
    except Exception:
        return None


# ==========================
# 画像取得ラッパー（Wikipedia→DuckDuckGo）
# ==========================
def get_bird_image(species_name: str) -> Optional[str]:
    img = get_wikipedia_image(species_name)
    if img:
        return img
    return get_duckduckgo_image(species_name)

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="鳴き声で鳥を特定するAI", page_icon="🪶", layout="wide")
st.title("🪶 鳥の鳴き声で種類名を特定するAIエージェント")

st.markdown("""
このアプリは、デバイスのマイクで録音した鳴き声をGPTモデルに解析させ、どの鳥かを推定し画像を取得して表示します。
""")

if "recorded_audio" not in st.session_state:
    st.session_state["recorded_audio"] = None

col1, col2 = st.columns(2)

with col1:
    if st.button("🎙️ 録音を開始（5秒間）"):
        audio = record_audio()
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(tmpfile.name, SAMPLE_RATE, audio)
        st.session_state["recorded_audio"] = tmpfile.name
    # 録音済みであれば常にプレビュー表示（再実行後も保持）
    if st.session_state["recorded_audio"]:
        st.audio(st.session_state["recorded_audio"], format="audio/wav")

with col2:
    if st.session_state["recorded_audio"]:
        if st.button("🔍 鳴き声を解析"):
            with open(st.session_state["recorded_audio"], "rb") as f:
                audio_bytes = f.read()

            result = analyze_bird_sound(audio_bytes)
            if result:
                st.subheader(f"🐦 {result['species']}（信頼度: {result['confidence']:.2f}）")
                st.write(result["description"])


                image_url = get_bird_image(result["species"])
                if image_url:
                    st.image(image_url, caption=result["species"], use_column_width=True)
                else:
                    st.warning("画像が見つかりませんでした。検索キーワードを工夫してください。")