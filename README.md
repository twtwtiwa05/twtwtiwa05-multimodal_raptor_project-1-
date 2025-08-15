#  Multi-modal RAPTOR Project (Python)

##  Overview
본 프로젝트는 **GTFS, OSM, GBFS 데이터를 통합**하여 **Python 기반 Multi-modal RAPTOR 알고리즘**을 구현한 연구입니다.  
서울 **강남구**를 실증 지역으로 설정하여, 출발지/도착지/출발시간 입력 시 **대중교통+도보+자전거**를 결합한 최적 경로를 탐색하고 지도에 시각화합니다.(시각화 부분 개발중) 
특히, 기존 OpenTripPlanner(OTP)에서 미지원되던 **GBFS(실시간 공유자전거)** 및 **GTFS-flex(탄력형 버스)** 데이터 처리 확장을 목표로 합니다.

---

##  Features

### 1. GTFS Loader (`GTFSLOADER.py`)
- GTFS 데이터를 Python 환경에서 로드
- 정류장, 노선, 운행 시간표 등 핵심 데이터 추출
- 도로망(Shapefile)과의 매핑 가능
- 데이터 통계 분석 기능 포함

### 2. Data Preparation (`part1_data_loader.py`)
- GTFS + 따릉이(GBFS 샘플) + 도보 네트워크(OSM) 통합
- RAPTOR 알고리즘 입력 형식에 맞춘 데이터 전처리
- NetworkX 기반 도보 연결망 생성

### 3. Routing (`part2_raptor_algorithm.py`)
- 출발지·도착지·출발시간 입력 → RAPTOR 알고리즘 실행
- Pareto Optimal 경로 탐색 (최단 도착 시간, 최소 환승)
- Multi-modal 확장 가능 구조

### 4. Visualization (`part3_visualization.py`)-> 미완성(추후에 보강 예정)
- RAPTOR 탐색 결과를 지도(Folium 등)에 시각화
- 경로, 정류장, 이동 수단 구분 표시
- haversine distance 기반 직선거리 계산

---

##  Project Structure
📁 project_root/
├── GTFSLOADER.py # GTFS 데이터 로더

├── part1_data_loader.py # 데이터 통합 및 RAPTOR 준비

├── part2_raptor_algorithm.py # 경로 탐색 실행

├── part3_visualization.py # 결과 시각화

├── gangnam_raptor_visualization_results

├── gangnam_multimodal_raptor_data_with_real_roads

├── output_integrated_transport_data

├── road_data

├── test_results


---

##  How It Works
1. **데이터 로딩** → `GTFSLOADER.py`로 GTFS 및 도로망 데이터 로드
2. **데이터 준비** → `part1_data_loader.py`로 Multi-modal RAPTOR 데이터 구조 생성
3. **경로 탐색** → `part2_raptor_algorithm.py`로 출발지/도착지 기반 경로 계산
4. **시각화** → `part3_visualization.py`로 지도에 경로 출력

---
## Data
드라이브 들어가서 직접 다운로드
https://drive.google.com/drive/folders/185xZBXGMi3Q2Y9vv62KYzppTPKD_cGWR?usp=sharing




##  Next Steps
- 시각화 part3 강력하게 재수정
- GBFS 실시간 공유자전거 데이터 연동
- GTFS-flex 기반 탄력형 버스 경로 반영
- Multi-criteria RAPTOR (시간 + 비용 + 환승) 확장
- 강남구 실증 분석 및 성능 비교 보고서 작성

---



