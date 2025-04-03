
import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta

STATION_COORDS = {
    "서울": (60, 127), "수원": (60, 120), "강릉": (92, 131),
    "청주": (69, 106), "대전": (67, 100), "광주": (58, 74),
    "대구": (89, 90), "부산": (98, 76), "제주": (52, 38)
}

def fetch_daily_forecast_for_city(city, nx, ny, api_key, target_date):
    base_date = target_date.strftime("%Y%m%d")
    base_time = "0500"  # 단기예보 고정 시간
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": api_key,
        "pageNo": "1",
        "numOfRows": "1000",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }

    res = requests.get(url, params=params)
    if res.status_code == 200:
        try:
            items = res.json()["response"]["body"]["items"]["item"]
            df = pd.DataFrame(items)
            df["fcst_datetime"] = pd.to_datetime(df["fcstDate"] + df["fcstTime"], format="%Y%m%d%H%M")
            # 가장 가까운 시간 1건만 수집 (예: 15시 또는 가장 가까운 것)
            target_time = datetime.combine(target_date, datetime.strptime("1500", "%H%M").time())
            available_times = df["fcst_datetime"].unique()
            if len(available_times) == 0:
                return None
            nearest_time = min(available_times, key=lambda x: abs(x - target_time))
            df = df[df["fcst_datetime"] == nearest_time]
            result = {row["category"]: float(row["fcstValue"]) for _, row in df.iterrows()}
            return result
        except Exception as e:
            print(f"❌ {city} 파싱 오류: {e}")
            return None
    else:
        print(f"❌ {city} API 오류: {res.status_code}")
        return None

def save_forecasts(api_key):
    today = datetime.now().date()
    cache_file = "weather_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    for city, (nx, ny) in STATION_COORDS.items():
        for i in range(4):  # 오늘 + 3일
            target_date = today + timedelta(days=i)
            key = f"{city}_{target_date.strftime('%Y-%m-%d')}"
            if key in cache:
                print(f"✅ 이미 저장됨: {key}")
                continue
            result = fetch_daily_forecast_for_city(city, nx, ny, api_key, target_date)
            if result:
                cache[key] = result
                print(f"✅ 저장 완료: {key}")
            else:
                print(f"⚠️ 수집 실패: {key}")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


# 아래처럼 함수화만 남깁니다:
def run_weather_fetcher(api_key):
    save_forecasts(api_key)
