## 📌 프로젝트명: AI 기반 맞춤형 음식 추천 시스템

### 🥗 프로젝트 개요
기상 정보(기온, 습도, 강수량 등)와 사용자 특성(성별, 연령대)을 기반으로 머신러닝 모델을 활용해 개인화된 음식 추천을 제공하는 시스템입니다.  
해당 프로젝트는 사용자의 상황에 최적화된 음식 추천을 통해 선택의 어려움을 줄이고, 더욱 만족스러운 식사 선택을 도와주는 것을 목표로 합니다.

### 🔍 주요 기능
- 실시간 날씨 데이터(기온, 강수, 습도 등) 반영
- 사용자 입력 정보 기반 추천
- 머신러닝 모델 (XGBoost) 기반 음식 그룹 분류
- 음식 이미지, 설명과 함께 추천 결과 제공
- Streamlit 기반 웹 애플리케이션 구현

### 🛠️ 사용 기술
| 분야 | 사용 기술 |
|------|-----------|
| 언어 | Python |
| 프레임워크 | Streamlit |
| 머신러닝 | XGBoost, PowerTransformer, KMeans |
| 시각화 | matplotlib, seaborn |
| 데이터 전처리 | pandas, scikit-learn |
| 기타 | 기상청 API, JSON 캐싱 |

### 📊 모델 성능
- 음식 그룹 분류 정확도: **66% (XGBoost 기준)**
- 로그 변환, 그룹별 모델링, 파생변수 생성 등 다양한 개선 전략 시도

### 📁 폴더 구조 예시
```
📦 음식추천시스템/
├─ app.py              # Streamlit 메인 실행 파일
├─ model/
│  ├─ model.pkl        # 훈련된 머신러닝 모델
│  └─ preprocessor.pkl # 전처리기
├─ data/
│  └─ food_weather.csv # 기상 + 음식 데이터
├─ utils/
│  └─ weather_api.py   # 기상청 API 호출 및 캐싱
├─ assets/
│  └─ food_images/     # 음식 이미지
├─ weather_cache.json  # API 캐시 데이터
└─ README.md
```

### 💡 프로젝트 주요 화면
![image](https://github.com/user-attachments/assets/e9ba7407-a57b-4356-8e09-7c44fa3a607a)


### 🔮 향후 발전 방향
- 더 다양한 사용자 특성 반영 (활동량, 건강 상태 등)
- 레시피 연동 및 쇼핑 연계 기능 추가
- 추천 알고리즘 다양화 (딥러닝, 협업 필터링 등)

### 👥 팀원
- **김OO**: 데이터 수집 및 전처리
- **이OO**: 모델 설계 및 성능 개선
- **송OO**: 웹앱 개발 및 배포

### 📎 참고
- 기상청 OpenAPI: https://www.data.go.kr/data/15084084/openapi.do
