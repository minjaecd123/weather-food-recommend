import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os, json, pickle
from datetime import datetime, timedelta, date
from sklearn.preprocessing import LabelEncoder
from urllib.parse import quote
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt
import re
import time
st.set_page_config(page_title="날씨 기반 음식 추천", layout="wide")

#다크모드일때 
is_dark = st.get_option("theme.base") == "dark"
text_color = "#fff" if is_dark else "#000"
bg_color = "#333" if is_dark else "#f8f8f8"
accent_color = "#4dabf7" if is_dark else "#1f77b4"


st.markdown("""
    <style>
    .food-card img {
        width: 100% !important;
        height: auto;
        border-radius: 10px;
    }
    .food-card {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin: 5px;
        text-align: center;
    }
    details summary::marker { display: none; }
    summary {
        cursor: pointer;
        font-weight: bold;
        font-size: 14px;
        margin: 8px 0 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 30px;'>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            🍱 날씨 기반 음식 추천
</div>
""", unsafe_allow_html=True)


groupname_map = {
    "Noodles": "면요리", "RiceDishes": "밥/죽/덮밥", "StirFryGrill": "볶음/구이",
    "BrunchSalad": "브런치/샐러드", "SideDish": "안주/보양식", "SoupStew": "찌개/국/탕"
}

STATION_COORDS = {
    "서울": (37.5665, 126.9780), "수원": (37.2636, 127.0286),
    "강릉": (37.7519, 128.8761), "청주": (36.6424, 127.4890),
    "대전": (36.3504, 127.3845), "광주": (35.1595, 126.8526),
    "대구": (35.8714, 128.6014), "부산": (35.1796, 129.0756),
    "제주": (33.4996, 126.5312)
}

sky_map = {"1": "맑음", "3": "구름 많음", "4": "흐림"}
pty_map = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈", "4": "소나기"}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat, d_lon = map(radians, [lat2 - lat1, lon2 - lon1])
    lat1, lat2 = map(radians, [lat1, lat2])
    a = sin(d_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(d_lon / 2)**2
    return 2 * R * asin(sqrt(a))

def find_nearest_station(lat, lon):
    return min(STATION_COORDS, key=lambda city: haversine(lat, lon, *STATION_COORDS[city]))

@st.cache_data
def load_material_map():
    df = pd.read_csv("./data/food_database.csv")
    return dict(zip(df["CKG_NM"], df["CKG_MTRL_CN"]))

material_map = load_material_map()

def load_food_mapping():
    df = pd.read_csv("./data/food_database.csv")
    df = df[df["CKG_GROUP"].notna()]
    df["Group_Eng"] = df["CKG_GROUP"].map({
        "면요리": "Noodles", "밥/죽/덮밥": "RiceDishes", "볶음/구이": "StirFryGrill",
        "브런치/샐러드": "BrunchSalad", "안주/보양식": "SideDish", "찌개/국/탕": "SoupStew"
    })
    return df.groupby("Group_Eng")["CKG_NM"].apply(list).to_dict()

food_dict = load_food_mapping()

with open("./data/food_description_map.pkl", "rb") as f:
    food_description_map = pickle.load(f)

def clean_material_text(text):
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[▣●★※•◆▶▷→⇨→★]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^]]*\]", "", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9,.\s]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

WEATHER_CACHE_FILE = "weather_cache.json"

def load_weather_cache():
    return json.load(open(WEATHER_CACHE_FILE)) if os.path.exists(WEATHER_CACHE_FILE) else {}

def save_weather_cache(data):
    with open(WEATHER_CACHE_FILE, "w") as f:
        json.dump(data, f)

def fetch_weather(service_key, target_date):
    today = datetime.today().date()
    is_today = target_date == today
    base_time = (datetime.now() - timedelta(minutes=40)).strftime('%H00') if is_today else "0500"
    base_date = datetime.now().strftime('%Y%m%d')
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/' + ('getUltraSrtNcst' if is_today else 'getVilageFcst')
    params = {
        'serviceKey': service_key, 'pageNo': '1', 'numOfRows': '1000',
        'dataType': 'JSON', 'base_date': base_date, 'base_time': base_time,
        'nx': 60, 'ny': 127
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        try:
            items = res.json()['response']['body']['items']['item']
            if is_today:
                return {i['category']: float(i['obsrValue']) for i in items if 'obsrValue' in i}
            df = pd.DataFrame(items)
            df['fcst_datetime'] = pd.to_datetime(df['fcstDate'] + df['fcstTime'], format='%Y%m%d%H%M')
            target_time = datetime.combine(target_date, datetime.strptime("1500", "%H%M").time())
            df = df[df['fcst_datetime'] == target_time]
            return {row['category']: float(row['fcstValue']) for _, row in df.iterrows()}
        except:
            return None
    return None


# 사용자 입력 및 지도 선택
left, right = st.columns([1, 7])

with left:
    st.markdown("### 👤 입력 정보")
    gender = st.selectbox("성별", ["남성", "여성"])
    age_group = st.selectbox("연령대", ["청년층", "중년층", "장년층"])
    selected_date = st.date_input("날짜 선택", value=date.today(), min_value=date.today(), max_value=date.today()+timedelta(days=3))

    st.markdown("### 🗺  위치 선택")
    map_center = [36.5, 127.8]
    m = folium.Map(location=map_center, zoom_start=6)
    if "map_click" not in st.session_state:
        st.session_state.map_click = None
    if st.session_state.get("map_click"):
        clicked = st.session_state["map_click"]
        folium.Marker(location=[clicked["lat"], clicked["lng"]], icon=folium.Icon(color="red", icon="flag")).add_to(m)
    map_result = st_folium(m, height=300, width=360, returned_objects=["last_clicked"])
    if map_result.get("last_clicked"):
        st.session_state["map_click"] = map_result["last_clicked"]
    clicked = st.session_state.map_click
    city = find_nearest_station(clicked["lat"], clicked["lng"]) if clicked else "서울"


with right:
    if st.button("📊 음식 추천 받기", use_container_width=True):
        cache = load_weather_cache()
        key = f"{city}_{selected_date.strftime('%Y-%m-%d')}"
        weather = cache.get(key)
        if not weather:
            weather = fetch_weather(st.secrets["KMA_API_KEY"], selected_date)
            if weather:
                cache[key] = weather
                save_weather_cache(cache)

        temp = weather.get("T1H", weather.get("TMP", 20)) if weather else 20
        humidity = weather.get("REH", 50) if weather else 50
        wind = weather.get("WSD", 2) if weather else 2
        rain = weather.get("RN1", 0) if weather else 0
        sky = sky_map.get(str(int(weather.get("SKY", 1))), "정보 없음") if weather else "정보 없음"
        pty = pty_map.get(str(int(weather.get("PTY", 0))), "정보 없음") if weather else "정보 없음"

        # 🔹 날씨 스타일 먼저 추가
        st.markdown("""
        <style>
        div[data-testid="column"] {
            padding: 0 4px !important;  /* 좌우 여백 줄이기 */
            margin: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # 🔹 제목
        st.markdown(f"""
        <div style='text-align:center; margin-bottom: 10px;'>
            <div style='text-align:center; font-size: 30px; font-weight: bold; '>{selected_date.strftime('%Y-%m-%d')}</div>
        </div>
        """, unsafe_allow_html=True)
        # 🔹 기온 습도 풍속 강수량 
        st.markdown(f"""
        <style>
        .weather-grid {{
            display: flex;
            justify-content: center;
            gap: 10px;  /* 여기서 카드 간격 조절 */
            margin-top: 15px;
            margin-bottom: 10px;
        }}
        .weather-card {{
            flex: 1;
            background: {bg_color}; 
            color: {text_color};        
            background: #f8f8f8;
            padding: 10px 14px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0px 0px 4px rgba(0,0,0,0.05);
            font-size: 16px;
            font-weight: bold;
            min-width: 120px;
        }}
        </style>

        <div class="weather-grid">
            <div class="weather-card">🌡 기온<br>{temp:.1f}°C</div>
            <div class="weather-card">💧 습도<br>{humidity:.0f}%</div>
            <div class="weather-card">🌬 풍속<br>{wind:.1f} m/s</div>
            <div class="weather-card">☔ 강수량<br>{rain:.1f} mm</div>
        </div>
        """, unsafe_allow_html=True)

        # 🔹 하늘상태 강수형태 
        st.markdown(f"""
        <div style='text-align:center; margin-top: 10px; font-size: 20px; font-weight: bold;'>
            ☁️ 하늘 상태: <b>{sky}</b> &nbsp;&nbsp; 🌧️ 강수형태: <b>{pty}</b>
        </div>
        """, unsafe_allow_html=True)

        now = datetime.combine(selected_date, datetime.min.time())

        input_data = pd.DataFrame([{
            "Gender": LabelEncoder().fit(["남성", "여성"]).transform([gender])[0],
            "Age_Group": LabelEncoder().fit(["청년층", "중년층", "장년층"]).transform([age_group])[0],
            "Region": LabelEncoder().fit(list(STATION_COORDS.keys())).transform([city])[0],
            "TA_AVG": temp, "HM_AVG": humidity, "WS_AVG": wind, "RN_DAY": rain,
            "Month_sin": np.sin(2 * np.pi * now.month / 12),
            "Month_cos": np.cos(2 * np.pi * now.month / 12),
            "Day_sin": np.sin(2 * np.pi * now.day / 31),
            "Day_cos": np.cos(2 * np.pi * now.day / 31),
            "is_weekend": now.weekday() >= 5
        }])

        model_folder = "./Models_LGBM_Streamlit_Best_Tuned"
        predictions = {}
        for file in os.listdir(model_folder):
            if file.endswith(".pkl"):
                model = joblib.load(os.path.join(model_folder, file))
                group = file.replace(".pkl", "")
                try:
                    predictions[group] = model.predict(input_data)[0]
                except:
                    continue

        # 추천 결과 출력
        top_3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        st.markdown("""
        <div style='text-align:center; font-size: 30px; font-weight: bold; margin-top: 30px; margin-bottom: 10px;'>
            🍽 추천 음식 Top 3
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(3, gap="small")

        for idx, (group_eng, _) in enumerate(top_3):
            with cols[idx]:
                group_kor = groupname_map.get(group_eng, group_eng)
                examples = food_dict.get(group_eng, [])
                food = np.random.choice(examples, size=1)[0] if examples else "추천 없음"
                query = quote(food)
                search_url = f"https://www.google.com/search?tbm=isch&q={query}"

                headers = {"User-Agent": "Mozilla/5.0"}
                try:
                    res = requests.get(search_url, headers=headers)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    img_tag = next((img for img in soup.find_all("img") if img.get("src", "").startswith("http")), None)
                    img_src = img_tag.get("src") if img_tag else ""
                    img = Image.open(BytesIO(requests.get(img_src).content)).resize((200, 200)) if img_src else None
                except:
                    img = None

                img_src = ""
                if img_tag and img_tag.get("src", "").startswith("http"):
                    img_src = img_tag.get("src")

                materials = clean_material_text(material_map.get(food, "재료 정보가 없습니다."))
                description_text = food_description_map.get(food, f"{food}는 계절과 날씨에 어울리는 음식이에요.")

                card_html = f"""
                <div class='food-card' style="min-height: 320px; display: flex; flex-direction: column; justify-content: space-between;">
                    {'<img src="' + img_src + '" style="width:100%; border-radius:10px; height:200px; object-fit:cover; margin-bottom: 8px;" />' if img_src else ''}
                    <p style="margin-top: 0; margin-bottom: 12px; font-size: 16px; text-align: center;"><strong>{food}</strong></p>
                    <div style="margin-top:auto; text-align: center;">
                        <details class="mat">
                            <summary>📦 재료 보기 ▼</summary>
                            <p>{materials}</p>
                        </details>
                        <details class="desc">
                            <summary>📖 음식 설명 보기 ▼</summary>
                            <p>{description_text}</p>
                        </details>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                materials = clean_material_text(material_map.get(food, "재료 정보가 없습니다."))
                description_text = food_description_map.get(food, f"{food}는 계절과 날씨에 어울리는 음식이에요.")

                st.markdown(f'''''', unsafe_allow_html=True)
                #----------------
                #축하 애니매이션
                #st.success("추천완료")
                #st.balloons() 
                #---------------- 

        # ✅ 네이버 오픈 API 로고 + 출처 푸터 (중첩 없이)
        st.markdown("""
        <hr style="margin-top: 2em;">

        <!-- 로고 이미지 가로 배치 -->
        <div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
            <a href="https://developers.naver.com" target="_blank">
                <img src="https://blog.kakaocdn.net/dn/sqK1R/btsEP7laotN/FPuL86cYe1FANmpIxKAZC1/img.png" alt="NAVER 오픈 API" width="120">
            </a>
            <a href="https://www.10000recipe.com/" target="_blank">
                <img src="https://recipe1.ezmember.co.kr/img/logo4.png" alt="만개의레시피" width="130">
            </a>
            <a href="https://data.kma.go.kr/cmmn/main.do" target="_blank">
                <img src="data:image/png;base64,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" alt="만개의레시피" width="130">
            </a>
        </div>

        <!-- 출처 설명 -->
        <div style="text-align: center; font-size: 0.85em; color: gray; margin-top: 10px;">
        📄 본 서비스는 기상청 공공데이터, 네이버 자료, <strong>만개의레시피</strong>를 바탕으로 개발되었습니다.<br>
        모델은 Scikit-learn 기반 LGBMClassifier를 사용하였으며, 이미지는 Google 이미지 검색을 통해 참조합니다.<br>
        © 2024 My Weather Food Recommender
        </div>
        """, unsafe_allow_html=True)                
