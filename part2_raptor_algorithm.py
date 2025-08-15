"""
강남구 Multi-modal RAPTOR 알고리즘 v3.0 완전판 - Part 1/2

프로젝트 지식 기반 정확한 RAPTOR 알고리즘 구현
- GTFS + 따릉이 GBFS + 실제 도로망 통합
- 완전한 멀티모달 라우팅 (도보, 자전거, 대중교통)
- Pareto 최적화 및 다중 기준 경로 선택
- 실제 도로망 기반 정확한 이동시간 계산
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import pickle
import json
import math
import bisect
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 핵심 데이터 구조 정의
# =============================================================================

@dataclass
class Stop:
    """정류장 정보"""
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    available_routes: List[str] = field(default_factory=list)
    is_major_station: bool = False

@dataclass
class Route:
    """노선 정보"""
    route_id: str
    route_name: str
    route_type: int  # 1: 지하철, 3: 버스
    route_color: str
    stop_pattern: List[str]  # 정류장 순서
    base_fare: float

@dataclass
class Trip:
    """운행 정보"""
    trip_id: str
    route_id: str
    service_id: str
    direction_id: int
    stop_times: List['StopTime'] = field(default_factory=list)

@dataclass
class StopTime:
    """정차 시간"""
    stop_id: str
    arrival_time: int  # 분 단위
    departure_time: int
    stop_sequence: int
    
@dataclass
class BikeStation:
    """따릉이 대여소"""
    station_id: str
    name: str
    lat: float
    lon: float
    bikes_available: int = 10  # 기본값
    docks_available: int = 10

@dataclass
class RoadSegment:
    """도로 구간"""
    from_node: Tuple[float, float]
    to_node: Tuple[float, float]
    length_km: float
    road_type: str
    walk_time: float
    bike_time: float

@dataclass
class Journey:
    """완전한 여행 경로"""
    total_time: int
    total_distance: float
    total_cost: float
    total_transfers: int
    departure_time: int
    arrival_time: int
    segments: List[Dict]
    route_coordinates: List[Tuple[float, float]] = field(default_factory=list)
    pareto_rank: int = 1
    journey_type: str = "mixed"  # walk, bike, transit, mixed

@dataclass
class RaptorLabel:
    """RAPTOR 레이블 (각 정류장의 최적 도착 정보)"""
    arrival_time: int = float('inf')
    transfers: int = 0
    cost: float = 0.0
    parent_stop: Optional[str] = None
    trip_id: Optional[str] = None
    route_id: Optional[str] = None
    access_mode: str = 'walk'
    round_number: int = 0
    boarding_time: int = 0

# =============================================================================
# 메인 RAPTOR 엔진 클래스
# =============================================================================

class GangnamMultiModalRAPTOR:
    """강남구 Multi-modal RAPTOR 엔진 v3.0"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
        # 기본 데이터
        self.stops: Dict[str, Stop] = {}
        self.routes: Dict[str, Route] = {}
        self.trips: Dict[str, Trip] = {}
        self.bike_stations: Dict[str, BikeStation] = {}
        self.road_graph: nx.Graph = None
        
        # RAPTOR 최적화 구조
        self.route_to_trips: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # (trip_id, first_departure)
        self.stop_to_routes: Dict[str, List[str]] = defaultdict(list)
        self.transfers: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # (stop_id, transfer_time)
        
        # 실제 도로망 인덱스
        self.spatial_index = {}  # 빠른 공간 검색을 위한 인덱스
        
        # 상수
        self.WALK_SPEED = 4.5  # km/h
        self.BIKE_SPEED = 12.0  # km/h
        self.MAX_WALK_TIME = 15  # 분
        self.MAX_BIKE_TIME = 20  # 분
        self.TRANSFER_TIME = 3  # 분
        self.MAX_ROUNDS = 5
        self.BIKE_RENTAL_TIME = 2  # 대여/반납 시간
        
        # 요금 정보
        self.BASE_TRANSIT_FARE = 1370  # 지하철
        self.BASE_BUS_FARE = 1200     # 버스
        self.BIKE_BASE_FARE = 1000    # 따릉이 30분
        self.TRANSFER_DISCOUNT = 300
        
        print("🚀 강남구 Multi-modal RAPTOR 엔진 v3.0 초기화")
        self._load_all_data()
    
    def _load_all_data(self):
        """모든 데이터 로드"""
        print("📊 데이터 로딩 시작...")
        
        try:
            # 1. Part1에서 생성된 RAPTOR 구조 로드
            self._load_raptor_structures()
            
            # 2. 기본 CSV 데이터 로드
            self._load_csv_data()
            
            # 3. 실제 도로망 데이터 로드
            self._load_road_network()
            
            # 4. 최적화된 구조 구축
            self._build_optimized_structures()
            
            print("✅ 데이터 로딩 완료!")
            self._print_system_summary()
            
        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_raptor_structures(self):
        """Part1에서 생성된 RAPTOR 구조 로드"""
        raptor_file = self.data_path / 'gangnam_raptor_structures.pkl'
        
        if raptor_file.exists():
            try:
                with open(raptor_file, 'rb') as f:
                    raptor_data = pickle.load(f)
                    
                # RAPTOR 구조 로드
                route_patterns = raptor_data.get('route_patterns', {})
                stop_routes = raptor_data.get('stop_routes', {})
                trip_schedules = raptor_data.get('trip_schedules', {})
                transfers = raptor_data.get('transfers', {})
                
                # 형식 변환
                for route_id, pattern in route_patterns.items():
                    if pattern:
                        self.routes[route_id] = Route(
                            route_id=route_id,
                            route_name=f"노선_{route_id}",
                            route_type=1,  # 기본값
                            route_color="#0066CC",
                            stop_pattern=pattern,
                            base_fare=1370
                        )
                
                for stop_id, routes in stop_routes.items():
                    self.stop_to_routes[stop_id] = routes
                
                for trip_id, schedule in trip_schedules.items():
                    if schedule and len(schedule) > 0:
                        stop_times = []
                        for entry in schedule:
                            if isinstance(entry, dict):
                                stop_times.append(StopTime(
                                    stop_id=entry['stop_id'],
                                    arrival_time=entry.get('arrival', 0),
                                    departure_time=entry.get('departure', 0),
                                    stop_sequence=entry.get('sequence', 0)
                                ))
                        
                        # 첫 출발시간 계산
                        first_departure = stop_times[0].departure_time if stop_times else 0
                        
                        # Trip 객체 생성
                        if stop_times:
                            route_id = None
                            # route_id 찾기
                            for rid, pattern in route_patterns.items():
                                if pattern and stop_times[0].stop_id in pattern:
                                    route_id = rid
                                    break
                            
                            self.trips[trip_id] = Trip(
                                trip_id=trip_id,
                                route_id=route_id or "unknown",
                                service_id="default",
                                direction_id=0,
                                stop_times=stop_times
                            )
                            
                            # route_to_trips 구조 구축
                            if route_id:
                                self.route_to_trips[route_id].append((trip_id, first_departure))
                
                # 환승 정보 로드
                for stop_id, transfer_list in transfers.items():
                    if transfer_list:
                        for transfer_stop, transfer_time in transfer_list:
                            self.transfers[stop_id].append((transfer_stop, transfer_time))
                
                print(f"   ✅ RAPTOR 구조 로드: {len(self.routes)}개 노선, {len(self.trips)}개 trips")
                
            except Exception as e:
                print(f"   ⚠️ RAPTOR 구조 로드 실패: {e}")
        else:
            print(f"   ⚠️ RAPTOR 구조 파일 없음: {raptor_file}")
    
    def _load_csv_data(self):
        """CSV 데이터 로드"""
        print("   📂 CSV 데이터 로딩...")
        
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
        
        # 정류장 데이터
        stops_file = self.data_path / 'gangnam_stops.csv'
        if stops_file.exists():
            for encoding in encodings:
                try:
                    stops_df = pd.read_csv(stops_file, encoding=encoding)
                    for _, row in stops_df.iterrows():
                        if pd.notna(row['stop_lat']) and pd.notna(row['stop_lon']):
                            self.stops[row['stop_id']] = Stop(
                                stop_id=row['stop_id'],
                                stop_name=row.get('stop_name', f"정류장_{row['stop_id']}"),
                                stop_lat=row['stop_lat'],
                                stop_lon=row['stop_lon'],
                                available_routes=self.stop_to_routes.get(row['stop_id'], [])
                            )
                    print(f"     ✅ 정류장: {len(self.stops)}개 ({encoding})")
                    break
                except UnicodeDecodeError:
                    continue
        
        # 노선 데이터 보완
        routes_file = self.data_path / 'gangnam_routes.csv'
        if routes_file.exists():
            for encoding in encodings:
                try:
                    routes_df = pd.read_csv(routes_file, encoding=encoding)
                    for _, row in routes_df.iterrows():
                        route_id = row['route_id']
                        if route_id in self.routes:
                            # 기존 Route 정보 업데이트
                            self.routes[route_id].route_name = row.get('route_short_name', route_id)
                            self.routes[route_id].route_type = row.get('route_type', 3)
                            self.routes[route_id].base_fare = 1370 if row.get('route_type', 3) == 1 else 1200
                            
                            # 노선 색상 설정
                            if row.get('route_type', 3) == 1:  # 지하철
                                if '2' in str(row.get('route_short_name', '')):
                                    self.routes[route_id].route_color = "#00A84D"  # 2호선
                                elif '7' in str(row.get('route_short_name', '')):
                                    self.routes[route_id].route_color = "#996600"  # 7호선
                                elif '9' in str(row.get('route_short_name', '')):
                                    self.routes[route_id].route_color = "#D4003B"  # 9호선
                                else:
                                    self.routes[route_id].route_color = "#0052A4"
                            else:  # 버스
                                self.routes[route_id].route_color = "#53B332"
                    
                    print(f"     ✅ 노선 정보 업데이트: {len(self.routes)}개")
                    break
                except UnicodeDecodeError:
                    continue
        
        # 따릉이 데이터
        bike_file = self.data_path / 'gangnam_bike_stations.csv'
        if bike_file.exists():
            for encoding in encodings:
                try:
                    bike_df = pd.read_csv(bike_file, encoding=encoding)
                    for _, row in bike_df.iterrows():
                        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                            self.bike_stations[str(row['station_id'])] = BikeStation(
                                station_id=str(row['station_id']),
                                name=row.get('address1', f"대여소_{row['station_id']}"),
                                lat=row['latitude'],
                                lon=row['longitude'],
                                bikes_available=10,  # 실제 API에서 가져올 수 있음
                                docks_available=10
                            )
                    print(f"     ✅ 따릉이: {len(self.bike_stations)}개소 ({encoding})")
                    break
                except UnicodeDecodeError:
                    continue
    
    def _load_road_network(self):
        """실제 도로망 데이터 로드"""
        print("   🛣️ 도로망 데이터 로딩...")
        
        # NetworkX 그래프 로드 시도
        graph_files = [
            self.data_path / 'gangnam_road_graph.pkl',
            self.data_path / 'gangnam_road_graph.gpickle'
        ]
        
        for graph_file in graph_files:
            if graph_file.exists():
                try:
                    if graph_file.suffix == '.pkl':
                        with open(graph_file, 'rb') as f:
                            self.road_graph = pickle.load(f)
                    else:
                        self.road_graph = nx.read_gpickle(graph_file)
                    
                    print(f"     ✅ 도로 그래프: {self.road_graph.number_of_nodes():,}개 노드, {self.road_graph.number_of_edges():,}개 엣지")
                    break
                except Exception as e:
                    print(f"     ⚠️ {graph_file.name} 로드 실패: {e}")
        
        # 그래프를 로드하지 못한 경우 기본 그리드 생성
        if self.road_graph is None:
            print("     🔧 기본 도로 네트워크 생성...")
            self._create_basic_road_network()
    
    def _create_basic_road_network(self):
        """기본 도로 네트워크 생성 (그래프가 없는 경우)"""
        self.road_graph = nx.Graph()
        
        # 강남구 범위
        lat_min, lat_max = 37.46, 37.55
        lon_min, lon_max = 127.00, 127.14
        
        # 기본 그리드 생성 (100m 간격)
        grid_size = 0.001  # 약 100m
        
        nodes = []
        for lat in np.arange(lat_min, lat_max, grid_size):
            for lon in np.arange(lon_min, lon_max, grid_size):
                nodes.append((lat, lon))
        
        # 그래프에 노드 추가
        self.road_graph.add_nodes_from(nodes)
        
        # 인접한 노드들 간 엣지 생성
        for i, (lat1, lon1) in enumerate(nodes):
            for lat2, lon2 in nodes[i+1:]:
                distance = self._haversine_distance(lat1, lon1, lat2, lon2)
                if distance <= 0.15:  # 150m 이내 연결
                    walk_time = (distance / self.WALK_SPEED) * 60
                    bike_time = (distance / self.BIKE_SPEED) * 60
                    
                    self.road_graph.add_edge(
                        (lat1, lon1), (lat2, lon2),
                        distance=distance,
                        walk_time=walk_time,
                        bike_time=bike_time
                    )
        
        print(f"     ✅ 기본 그리드 생성: {self.road_graph.number_of_nodes():,}개 노드, {self.road_graph.number_of_edges():,}개 엣지")
    
    def _build_optimized_structures(self):
        """최적화된 구조 구축"""
        print("   ⚡ 최적화 구조 구축...")
        
        # route_to_trips 시간순 정렬
        for route_id in self.route_to_trips:
            self.route_to_trips[route_id].sort(key=lambda x: x[1])
        
        # 공간 인덱스 구축 (빠른 주변 검색용)
        self._build_spatial_index()
        
        # 환승 정보 최적화
        self._optimize_transfers()
        
        print("     ✅ 최적화 완료")
    
    def _build_spatial_index(self):
        """공간 인덱스 구축"""
        # 정류장 공간 인덱스
        self.spatial_index['stops'] = {}
        for stop_id, stop in self.stops.items():
            lat_key = int(stop.stop_lat * 1000)  # 0.001도 단위
            lon_key = int(stop.stop_lon * 1000)
            key = (lat_key, lon_key)
            if key not in self.spatial_index['stops']:
                self.spatial_index['stops'][key] = []
            self.spatial_index['stops'][key].append(stop_id)
        
        # 따릉이 공간 인덱스
        self.spatial_index['bikes'] = {}
        for station_id, station in self.bike_stations.items():
            lat_key = int(station.lat * 1000)
            lon_key = int(station.lon * 1000)
            key = (lat_key, lon_key)
            if key not in self.spatial_index['bikes']:
                self.spatial_index['bikes'][key] = []
            self.spatial_index['bikes'][key].append(station_id)
    
    def _optimize_transfers(self):
        """환승 정보 최적화"""
        # 거리 기반 환승 시간 재계산
        optimized_transfers = defaultdict(list)
        
        for stop_id in self.stops:
            nearby_stops = self._find_nearby_stops(stop_id, max_distance=0.3)  # 300m
            
            for nearby_stop_id, distance in nearby_stops:
                if nearby_stop_id != stop_id:
                    # 환승 시간 계산 (거리 기반)
                    transfer_time = max(2, min(8, int(distance * 1000 / 80)))  # 80m/분 보행속도
                    optimized_transfers[stop_id].append((nearby_stop_id, transfer_time))
        
        self.transfers = optimized_transfers
    
    def _find_nearby_stops(self, stop_id: str, max_distance: float = 0.5) -> List[Tuple[str, float]]:
        """주변 정류장 찾기"""
        if stop_id not in self.stops:
            return []
        
        origin_stop = self.stops[stop_id]
        nearby_stops = []
        
        for other_stop_id, other_stop in self.stops.items():
            if other_stop_id != stop_id:
                distance = self._haversine_distance(
                    origin_stop.stop_lat, origin_stop.stop_lon,
                    other_stop.stop_lat, other_stop.stop_lon
                )
                if distance <= max_distance:
                    nearby_stops.append((other_stop_id, distance))
        
        return sorted(nearby_stops, key=lambda x: x[1])
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """하버사인 공식으로 거리 계산 (km)"""
        R = 6371  # 지구 반지름 (km)
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

    # =============================================================================
    # 메인 라우팅 인터페이스
    # =============================================================================
    
    def find_routes(self, origin_lat: float, origin_lon: float,
                    dest_lat: float, dest_lon: float,
                    departure_time: str = "08:30",
                    max_routes: int = 5,
                    include_bike: bool = True,
                    user_preferences: Dict = None) -> List[Journey]:
        """
        멀티모달 경로 탐색 메인 함수
        
        Args:
            origin_lat, origin_lon: 출발지 좌표
            dest_lat, dest_lon: 목적지 좌표
            departure_time: 출발시간 ("HH:MM" 형식)
            max_routes: 최대 경로 수
            include_bike: 따릉이 포함 여부
            user_preferences: 사용자 선호도
        
        Returns:
            List[Journey]: 최적 경로 리스트
        """
        print(f"\n🎯 강남구 Multi-modal 경로 탐색 v3.0")
        print(f"   출발지: ({origin_lat:.6f}, {origin_lon:.6f})")
        print(f"   목적지: ({dest_lat:.6f}, {dest_lon:.6f})")
        print(f"   출발시간: {departure_time}")
        
        dep_time_minutes = self._parse_time_to_minutes(departure_time)
        
        # 기본 선호도 설정
        if user_preferences is None:
            user_preferences = {
                'time_weight': 0.5,      # 시간 중요도
                'cost_weight': 0.2,      # 비용 중요도 
                'transfer_weight': 0.3,  # 환승 중요도
                'max_walk_time': 15,     # 최대 도보시간
                'max_bike_time': 20,     # 최대 자전거시간
                'prefer_subway': True,   # 지하철 선호
                'avoid_bus': False       # 버스 회피
            }
        
        all_journeys = []
        
        # 1. 도보 전용 경로
        print("   🚶 도보 경로 탐색...")
        walk_journeys = self._find_walk_only_routes(
            origin_lat, origin_lon, dest_lat, dest_lon, dep_time_minutes
        )
        all_journeys.extend(walk_journeys)
        
        # 2. 따릉이 전용 경로 (요청시)
        if include_bike:
            print("   🚲 따릉이 경로 탐색...")
            bike_journeys = self._find_bike_only_routes(
                origin_lat, origin_lon, dest_lat, dest_lon, dep_time_minutes
            )
            all_journeys.extend(bike_journeys)
        
        # 3. 대중교통 기반 경로
        print("   🚇 대중교통 경로 탐색...")
        transit_journeys = self._find_transit_routes(
            origin_lat, origin_lon, dest_lat, dest_lon, 
            dep_time_minutes, include_bike, user_preferences
        )
        all_journeys.extend(transit_journeys)
        
        # 4. 혼합 경로 (따릉이 + 대중교통)
        if include_bike:
            print("   🔄 혼합 경로 탐색...")
            mixed_journeys = self._find_mixed_routes(
                origin_lat, origin_lon, dest_lat, dest_lon,
                dep_time_minutes, user_preferences
            )
            all_journeys.extend(mixed_journeys)
        
        # 5. Pareto 최적화 및 순위 매기기
        print("   ⚖️ Pareto 최적화...")
        optimized_journeys = self._pareto_optimize(all_journeys, user_preferences)
        
        # 6. 다양성 확보 및 최종 선택
        final_journeys = self._diversify_routes(optimized_journeys, max_routes)
        
        print(f"🎉 총 {len(final_journeys)}개 최적 경로 발견")
        return final_journeys
    
    def _parse_time_to_minutes(self, time_str: str) -> int:
        """시간 문자열을 분으로 변환"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return hour * 60 + minute
        except:
            return 8 * 60  # 기본값: 08:00
    
    def _print_system_summary(self):
        """시스템 요약 출력"""
        print(f"\n📊 시스템 요약:")
        print(f"   정류장: {len(self.stops):,}개")
        print(f"   노선: {len(self.routes):,}개")
        print(f"   운행: {len(self.trips):,}개")
        print(f"   따릉이: {len(self.bike_stations):,}개소")
        
        if self.road_graph:
            print(f"   도로 노드: {self.road_graph.number_of_nodes():,}개")
            print(f"   도로 엣지: {self.road_graph.number_of_edges():,}개")
        
        # 환승 정보
        total_transfers = sum(len(transfers) for transfers in self.transfers.values())
        print(f"   환승 연결: {total_transfers:,}개")
        
        # 노선별 통계
        subway_routes = sum(1 for r in self.routes.values() if r.route_type == 1)
        bus_routes = len(self.routes) - subway_routes
        print(f"   지하철: {subway_routes}개 노선")
        print(f"   버스: {bus_routes}개 노선")  
    
    """
강남구 Multi-modal RAPTOR 알고리즘 v3.0 완전판 - Part 2/2

실제 라우팅 구현 및 최적화 함수들
- 도보/자전거/대중교통 각 모드별 라우팅
- 핵심 RAPTOR 알고리즘 구현
- Pareto 최적화 및 경로 다양성 확보
- 실제 도로망 기반 정확한 시간 계산
"""

    # =============================================================================
    # 각 교통수단별 라우팅 함수들
    # =============================================================================
    
    def _find_walk_only_routes(self, origin_lat: float, origin_lon: float,
                              dest_lat: float, dest_lon: float, 
                              dep_time: int) -> List[Journey]:
        """도보 전용 경로"""
        journeys = []
        
        # 직선거리 계산
        distance = self._haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)
        
        if distance <= 2.0:  # 2km 이내만 도보 추천
            # 실제 도로망 경로 계산
            walk_time, walk_distance, coordinates = self._calculate_road_route(
                origin_lat, origin_lon, dest_lat, dest_lon, 'walk'
            )
            
            if walk_time <= self.MAX_WALK_TIME:
                journey = Journey(
                    total_time=int(walk_time),
                    total_distance=walk_distance,
                    total_cost=0.0,
                    total_transfers=0,
                    departure_time=dep_time,
                    arrival_time=dep_time + int(walk_time),
                    journey_type="walk",
                    route_coordinates=coordinates,
                    segments=[{
                        'mode': 'walk',
                        'from': '출발지',
                        'to': '목적지',
                        'duration': int(walk_time),
                        'distance_km': walk_distance,
                        'cost': 0,
                        'route_info': f'도보 {int(walk_time)}분',
                        'coordinates': coordinates
                    }]
                )
                journeys.append(journey)
        
        return journeys
    
    def _find_bike_only_routes(self, origin_lat: float, origin_lon: float,
                              dest_lat: float, dest_lon: float,
                              dep_time: int) -> List[Journey]:
        """따릉이 전용 경로"""
        journeys = []
        
        # 출발지 근처 따릉이 대여소 찾기
        origin_stations = self._find_nearby_bike_stations(origin_lat, origin_lon, 0.5)  # 500m
        dest_stations = self._find_nearby_bike_stations(dest_lat, dest_lon, 0.5)
        
        if not origin_stations or not dest_stations:
            return journeys
        
        print(f"       따릉이 대여소: 출발지 {len(origin_stations)}개, 목적지 {len(dest_stations)}개")
        
        # 최적 대여소 조합 찾기 (최대 3개씩만)
        for start_station_id, start_dist in origin_stations[:3]:
            start_station = self.bike_stations[start_station_id]
            
            for end_station_id, end_dist in dest_stations[:3]:
                if start_station_id == end_station_id:  # 같은 대여소는 제외
                    continue
                    
                end_station = self.bike_stations[end_station_id]
                
                # 1. 출발지 → 대여소 (도보)
                walk_to_start_time, walk_to_start_dist, coords1 = self._calculate_road_route(
                    origin_lat, origin_lon, start_station.lat, start_station.lon, 'walk'
                )
                
                # 2. 대여소 → 대여소 (자전거)
                bike_time, bike_dist, coords2 = self._calculate_road_route(
                    start_station.lat, start_station.lon, 
                    end_station.lat, end_station.lon, 'bike'
                )
                
                # 3. 대여소 → 목적지 (도보)
                walk_to_dest_time, walk_to_dest_dist, coords3 = self._calculate_road_route(
                    end_station.lat, end_station.lon, dest_lat, dest_lon, 'walk'
                )
                
                total_time = (walk_to_start_time + self.BIKE_RENTAL_TIME + 
                             bike_time + self.BIKE_RENTAL_TIME + walk_to_dest_time)
                
                # 최소 시간 보장 (너무 짧으면 비현실적)
                walk_to_start_time = max(2, walk_to_start_time)
                bike_time = max(5, bike_time)
                walk_to_dest_time = max(2, walk_to_dest_time)
                total_time = max(10, total_time)  # 최소 10분
                
                if total_time <= 45:  # 45분 이내만
                    bike_cost = self._calculate_bike_cost(bike_time)
                    total_distance = max(0.5, walk_to_start_dist + bike_dist + walk_to_dest_dist)
                    
                    # 전체 경로 좌표
                    all_coordinates = coords1 + coords2 + coords3
                    
                    # 대여소 이름 정리 (인코딩 문제 해결)
                    start_name = self._clean_station_name(start_station.name)
                    end_name = self._clean_station_name(end_station.name)
                    
                    journey = Journey(
                        total_time=int(total_time),
                        total_distance=total_distance,
                        total_cost=bike_cost,
                        total_transfers=0,
                        departure_time=dep_time,
                        arrival_time=dep_time + int(total_time),
                        journey_type="bike",
                        route_coordinates=all_coordinates,
                        segments=[
                            {
                                'mode': 'walk',
                                'from': '출발지',
                                'to': start_name,
                                'duration': int(walk_to_start_time),
                                'distance_km': walk_to_start_dist,
                                'cost': 0,
                                'route_info': '도보 (따릉이 대여소)',
                                'coordinates': coords1
                            },
                            {
                                'mode': 'bike_rental',
                                'from': start_name,
                                'to': start_name,
                                'duration': self.BIKE_RENTAL_TIME,
                                'cost': 0,
                                'route_info': '따릉이 대여'
                            },
                            {
                                'mode': 'bike',
                                'from': start_name,
                                'to': end_name,
                                'duration': int(bike_time),
                                'distance_km': bike_dist,
                                'cost': bike_cost,
                                'route_info': f'따릉이 {int(bike_time)}분',
                                'coordinates': coords2
                            },
                            {
                                'mode': 'bike_return',
                                'from': end_name,
                                'to': end_name,
                                'duration': self.BIKE_RENTAL_TIME,
                                'cost': 0,
                                'route_info': '따릉이 반납'
                            },
                            {
                                'mode': 'walk',
                                'from': end_name,
                                'to': '목적지',
                                'duration': int(walk_to_dest_time),
                                'distance_km': walk_to_dest_dist,
                                'cost': 0,
                                'route_info': '도보',
                                'coordinates': coords3
                            }
                        ]
                    )
                    journeys.append(journey)
        
        print(f"       생성된 따릉이 경로: {len(journeys)}개")
        return journeys
    
    def _clean_station_name(self, name: str) -> str:
        """대여소 이름 정리"""
        if not name or pd.isna(name):
            return "따릉이 대여소"
        
        name_str = str(name)
        
        # 인코딩 문제가 있는 경우
        if any(ord(c) > 127 for c in name_str if c.isprintable()):
            try:
                # UTF-8로 다시 디코딩 시도
                clean_name = name_str.encode('cp949').decode('utf-8')
            except:
                return "따릉이 대여소"
        else:
            clean_name = name_str
        
        # 길이 제한
        if len(clean_name) > 30:
            clean_name = clean_name[:30] + "..."
        
        return clean_name
    
    def _find_transit_routes(self, origin_lat: float, origin_lon: float,
                            dest_lat: float, dest_lon: float,
                            dep_time: int, include_bike: bool,
                            preferences: Dict) -> List[Journey]:
        """대중교통 기반 경로 (RAPTOR 알고리즘 사용)"""
        
        # 1. 접근 가능한 정류장들 찾기
        access_stops = self._find_access_stops(
            origin_lat, origin_lon, include_bike, preferences
        )
        
        egress_stops = self._find_access_stops(
            dest_lat, dest_lon, include_bike, preferences
        )
        
        if not access_stops or not egress_stops:
            return []
        
        print(f"     접근 정류장: {len(access_stops)}개, 하차 정류장: {len(egress_stops)}개")
        
        # 2. 핵심 RAPTOR 알고리즘 실행
        raptor_results = self._execute_raptor_algorithm(
            access_stops, egress_stops, dep_time, preferences
        )
        
        # 3. RAPTOR 결과를 Journey 객체로 변환
        journeys = []
        for result in raptor_results:
            journey = self._reconstruct_journey_from_raptor(
                result, origin_lat, origin_lon, dest_lat, dest_lon, dep_time
            )
            if journey:
                journeys.append(journey)
        
        return journeys
    
    def _find_mixed_routes(self, origin_lat: float, origin_lon: float,
                          dest_lat: float, dest_lon: float,
                          dep_time: int, preferences: Dict) -> List[Journey]:
        """혼합 경로 (따릉이 + 대중교통)"""
        journeys = []
        
        # 출발지 근처 따릉이 대여소
        origin_bike_stations = self._find_nearby_bike_stations(origin_lat, origin_lon, 0.8)
        
        # 각 대여소에서 대중교통역까지 이동 후 대중교통 이용
        for station_id, dist in origin_bike_stations[:5]:  # 상위 5개만
            station = self.bike_stations[station_id]
            
            # 대여소 근처 대중교통 정류장 찾기
            nearby_stops = self._find_nearby_stops_from_point(
                station.lat, station.lon, max_distance=0.3
            )
            
            for stop_id, stop_dist in nearby_stops[:3]:
                if stop_id not in self.stops:
                    continue
                
                stop = self.stops[stop_id]
                
                # 1단계: 출발지 → 따릉이 대여소 (도보)
                walk1_time, walk1_dist, coords1 = self._calculate_road_route(
                    origin_lat, origin_lon, station.lat, station.lon, 'walk'
                )
                
                # 2단계: 대여소 → 지하철역 (따릉이)
                bike_time, bike_dist, coords2 = self._calculate_road_route(
                    station.lat, station.lon, stop.stop_lat, stop.stop_lon, 'bike'
                )
                
                # 3단계: 지하철역 → 목적지 근처역 (대중교통)
                transit_options = self._find_simple_transit_route(
                    stop_id, dest_lat, dest_lon, 
                    dep_time + int(walk1_time + self.BIKE_RENTAL_TIME + bike_time + self.BIKE_RENTAL_TIME)
                )
                
                for transit_result in transit_options:
                    total_time = (walk1_time + self.BIKE_RENTAL_TIME + bike_time + 
                                 self.BIKE_RENTAL_TIME + transit_result['duration'] + 
                                 transit_result['egress_time'])
                    
                    if total_time <= 60:  # 1시간 이내
                        bike_cost = self._calculate_bike_cost(bike_time)
                        transit_cost = transit_result.get('cost', self.BASE_TRANSIT_FARE)
                        
                        journey = Journey(
                            total_time=int(total_time),
                            total_distance=walk1_dist + bike_dist + transit_result.get('distance', 3.0),
                            total_cost=bike_cost + transit_cost,
                            total_transfers=1 + transit_result.get('transfers', 0),
                            departure_time=dep_time,
                            arrival_time=dep_time + int(total_time),
                            journey_type="mixed",
                            segments=[
                                {
                                    'mode': 'walk',
                                    'from': '출발지',
                                    'to': station.name,
                                    'duration': int(walk1_time),
                                    'distance_km': walk1_dist,
                                    'cost': 0,
                                    'route_info': '도보 (따릉이 대여소)',
                                    'coordinates': coords1
                                },
                                {
                                    'mode': 'bike',
                                    'from': station.name,
                                    'to': stop.stop_name,
                                    'duration': int(bike_time) + 2 * self.BIKE_RENTAL_TIME,
                                    'distance_km': bike_dist,
                                    'cost': bike_cost,
                                    'route_info': f'따릉이 {int(bike_time)}분',
                                    'coordinates': coords2
                                },
                                {
                                    'mode': 'transit',
                                    'from': stop.stop_name,
                                    'to': transit_result['dest_stop_name'],
                                    'duration': transit_result['duration'],
                                    'cost': transit_cost,
                                    'route_info': transit_result['route_info'],
                                    'route_id': transit_result.get('route_id'),
                                    'route_color': transit_result.get('route_color', '#0066CC')
                                },
                                {
                                    'mode': 'walk',
                                    'from': transit_result['dest_stop_name'],
                                    'to': '목적지',
                                    'duration': int(transit_result['egress_time']),
                                    'cost': 0,
                                    'route_info': '도보'
                                }
                            ]
                        )
                        journeys.append(journey)
        
        return journeys
    
    # =============================================================================
    # 핵심 RAPTOR 알고리즘 구현
    # =============================================================================
    
    def _execute_raptor_algorithm(self, access_stops: List[Dict], 
                                 egress_stops: List[Dict],
                                 dep_time: int, preferences: Dict) -> List[Dict]:
        """핵심 RAPTOR 알고리즘 실행"""
        
        print(f"     🔄 RAPTOR 알고리즘 시작...")
        
        # 초기화
        best_labels = {}  # stop_id -> RaptorLabel
        marked_stops = set()
        
        # Round 0: Access stops 초기화
        for access in access_stops:
            stop_id = access['stop_id']
            arrival_time = dep_time + access['access_time']
            
            best_labels[stop_id] = RaptorLabel(
                arrival_time=arrival_time,
                transfers=0,
                cost=0.0,
                access_mode=access['mode'],
                round_number=0
            )
            marked_stops.add(stop_id)
        
        print(f"       Round 0: {len(marked_stops)}개 접근점 초기화")
        
        # Rounds 1 to MAX_ROUNDS
        for round_num in range(1, self.MAX_ROUNDS + 1):
            if not marked_stops:
                break
            
            new_marked_stops = set()
            
            # Route Scanning
            scanned_routes = set()
            for stop_id in marked_stops:
                for route_id in self.stop_to_routes.get(stop_id, []):
                    if route_id not in scanned_routes:
                        scanned_routes.add(route_id)
                        new_stops = self._scan_route(
                            route_id, marked_stops, best_labels, dep_time, round_num
                        )
                        new_marked_stops.update(new_stops)
            
            # Transfer Processing
            transfer_stops = self._process_transfers(best_labels, round_num)
            new_marked_stops.update(transfer_stops)
            
            marked_stops = new_marked_stops
            print(f"       Round {round_num}: {len(marked_stops)}개 정류장 업데이트")
            
            if not marked_stops:
                break
        
        # 결과 수집
        results = []
        for egress in egress_stops:
            stop_id = egress['stop_id']
            if stop_id in best_labels:
                label = best_labels[stop_id]
                egress_time = egress.get('egress_time', egress.get('access_time', 5))  # 수정: 키 오류 방지
                total_time = (label.arrival_time - dep_time) + egress_time
                
                results.append({
                    'dest_stop_id': stop_id,
                    'dest_stop_name': egress.get('stop_name', f'정류장_{stop_id}'),
                    'arrival_time': label.arrival_time,
                    'total_time': total_time,
                    'transfers': label.transfers,
                    'cost': label.cost,
                    'trip_id': label.trip_id,
                    'route_id': label.route_id,
                    'egress_time': egress_time,
                    'egress_mode': egress.get('mode', 'walk')
                })
        
        print(f"     ✅ RAPTOR 완료: {len(results)}개 경로 발견")
        return results
    
    def _scan_route(self, route_id: str, marked_stops: Set[str],
                   best_labels: Dict[str, RaptorLabel], dep_time: int,
                   round_num: int) -> Set[str]:
        """개별 노선 스캔"""
        
        if route_id not in self.routes:
            return set()
        
        route = self.routes[route_id]
        new_stops = set()
        
        # 이 노선의 최적 trip 찾기
        best_trip = self._find_best_trip_for_route(
            route_id, marked_stops, best_labels, dep_time
        )
        
        if not best_trip:
            return new_stops
        
        trip_id, boarding_stop_id, boarding_time = best_trip
        boarding_label = best_labels[boarding_stop_id]
        
        # Trip의 모든 후속 정류장 업데이트
        if trip_id in self.trips:
            trip = self.trips[trip_id]
            boarding_found = False
            
            for stop_time in trip.stop_times:
                # 탑승 정류장 찾기
                if stop_time.stop_id == boarding_stop_id:
                    boarding_found = True
                    continue
                
                # 탑승 이후 정류장들만 처리
                if not boarding_found:
                    continue
                
                # 도착시간이 탑승시간보다 이후인지 확인
                if stop_time.arrival_time <= boarding_time:
                    continue
                
                # 새로운 레이블 계산
                new_arrival = stop_time.arrival_time
                new_transfers = boarding_label.transfers + 1
                new_cost = boarding_label.cost + route.base_fare
                
                # 환승할인 적용
                if boarding_label.transfers > 0:
                    new_cost -= self.TRANSFER_DISCOUNT
                
                # 기존 레이블과 비교
                if self._is_label_better(new_arrival, new_transfers, new_cost,
                                       best_labels.get(stop_time.stop_id)):
                    
                    best_labels[stop_time.stop_id] = RaptorLabel(
                        arrival_time=new_arrival,
                        transfers=new_transfers,
                        cost=new_cost,
                        parent_stop=boarding_stop_id,
                        trip_id=trip_id,
                        route_id=route_id,
                        access_mode='transit',
                        round_number=round_num,
                        boarding_time=boarding_time
                    )
                    new_stops.add(stop_time.stop_id)
        
        return new_stops
    
    def _find_best_trip_for_route(self, route_id: str, marked_stops: Set[str],
                                 best_labels: Dict[str, RaptorLabel],
                                 dep_time: int) -> Optional[Tuple[str, str, int]]:
        """노선에서 최적 trip 찾기"""
        
        best_option = None
        earliest_departure = float('inf')
        
        # 이 노선의 정류장들 중 마킹된 것들
        route_marked_stops = []
        if route_id in self.routes:
            for stop_id in self.routes[route_id].stop_pattern:
                if stop_id in marked_stops:
                    route_marked_stops.append(stop_id)
        
        # 각 마킹된 정류장에서 탑승 가능한 trip 찾기
        for stop_id in route_marked_stops:
            label = best_labels[stop_id]
            earliest_board_time = label.arrival_time + 1  # 1분 여유
            
            # 이 노선의 trips 중에서 적합한 것 찾기
            for trip_id, first_dep in self.route_to_trips.get(route_id, []):
                if trip_id in self.trips:
                    trip = self.trips[trip_id]
                    
                    # 이 정류장에서의 출발시간 찾기
                    for stop_time in trip.stop_times:
                        if (stop_time.stop_id == stop_id and 
                            stop_time.departure_time >= earliest_board_time):
                            
                            if stop_time.departure_time < earliest_departure:
                                earliest_departure = stop_time.departure_time
                                best_option = (trip_id, stop_id, stop_time.departure_time)
                            break
        
        return best_option
    
    def _process_transfers(self, best_labels: Dict[str, RaptorLabel],
                          round_num: int) -> Set[str]:
        """환승 처리"""
        new_stops = set()
        
        # 현재 레이블의 복사본으로 안전한 iteration
        current_labels = dict(best_labels)
        
        for stop_id, label in current_labels.items():
            if stop_id in self.transfers:
                for transfer_stop_id, transfer_time in self.transfers[stop_id]:
                    new_arrival = label.arrival_time + transfer_time
                    
                    # 기존 레이블과 비교
                    if self._is_label_better(new_arrival, label.transfers, label.cost,
                                           best_labels.get(transfer_stop_id)):
                        
                        best_labels[transfer_stop_id] = RaptorLabel(
                            arrival_time=new_arrival,
                            transfers=label.transfers,
                            cost=label.cost,
                            parent_stop=stop_id,
                            trip_id=label.trip_id,
                            route_id=label.route_id,
                            access_mode=label.access_mode,
                            round_number=round_num,
                            boarding_time=label.boarding_time
                        )
                        new_stops.add(transfer_stop_id)
        
        return new_stops
    
    def _is_label_better(self, new_arrival: int, new_transfers: int, new_cost: float,
                        existing_label: Optional[RaptorLabel]) -> bool:
        """레이블 개선 여부 판단 (Pareto 비교)"""
        if existing_label is None:
            return True
        
        # 단순히 도착시간이 빠르면 좋음
        if new_arrival < existing_label.arrival_time:
            return True
        
        # 도착시간이 같으면 환승횟수와 비용 고려
        if new_arrival == existing_label.arrival_time:
            if new_transfers < existing_label.transfers:
                return True
            if new_transfers == existing_label.transfers and new_cost < existing_label.cost:
                return True
        
        # 도착시간이 5분 이내 차이면 환승이 적은 것 선호
        if (new_arrival <= existing_label.arrival_time + 5 and 
            new_transfers < existing_label.transfers):
            return True
        
        return False
    
    # =============================================================================
    # 유틸리티 함수들
    # =============================================================================
    
    def _find_access_stops(self, lat: float, lon: float, include_bike: bool,
                          preferences: Dict) -> List[Dict]:
        """접근 가능한 정류장들 찾기"""
        access_stops = []
        
        # 도보 접근 정류장
        nearby_stops = self._find_nearby_stops_from_point(lat, lon, 
                                                         preferences.get('max_walk_time', 15) / 60 * self.WALK_SPEED)
        
        for stop_id, distance in nearby_stops:
            if stop_id in self.stops:
                stop = self.stops[stop_id]
                walk_time = (distance / self.WALK_SPEED) * 60
                
                if walk_time <= preferences.get('max_walk_time', 15):
                    access_stops.append({
                        'stop_id': stop_id,
                        'stop_name': stop.stop_name,
                        'access_time': int(walk_time),
                        'egress_time': int(walk_time),  # 추가: egress_time 키 추가
                        'mode': 'walk',
                        'distance': distance
                    })
        
        # 따릉이 접근 정류장 (요청시)
        if include_bike:
            bike_stations = self._find_nearby_bike_stations(lat, lon, 0.5)
            
            for station_id, station_dist in bike_stations[:3]:  # 상위 3개만
                station = self.bike_stations[station_id]
                
                # 대여소 근처 정류장들
                station_nearby_stops = self._find_nearby_stops_from_point(
                    station.lat, station.lon, 0.3
                )
                
                for stop_id, stop_dist in station_nearby_stops[:2]:
                    if stop_id in self.stops:
                        stop = self.stops[stop_id]
                        
                        # 총 접근시간 = 도보(대여소) + 대여 + 자전거(정류장) + 반납
                        walk_to_station = (station_dist / self.WALK_SPEED) * 60
                        bike_to_stop = (stop_dist / self.BIKE_SPEED) * 60
                        total_time = walk_to_station + self.BIKE_RENTAL_TIME + bike_to_stop + self.BIKE_RENTAL_TIME
                        
                        if total_time <= preferences.get('max_bike_time', 20):
                            access_stops.append({
                                'stop_id': stop_id,
                                'stop_name': stop.stop_name,
                                'access_time': int(total_time),
                                'egress_time': int(total_time),  # 추가: egress_time 키 추가
                                'mode': 'bike',
                                'distance': station_dist + stop_dist,
                                'bike_station': station.name
                            })
        
        return sorted(access_stops, key=lambda x: x['access_time'])
    
    def _find_nearby_stops_from_point(self, lat: float, lon: float, 
                                     max_distance: float) -> List[Tuple[str, float]]:
        """특정 지점 근처 정류장 찾기"""
        nearby = []
        
        for stop_id, stop in self.stops.items():
            distance = self._haversine_distance(lat, lon, stop.stop_lat, stop.stop_lon)
            if distance <= max_distance:
                nearby.append((stop_id, distance))
        
        return sorted(nearby, key=lambda x: x[1])
    
    def _find_nearby_bike_stations(self, lat: float, lon: float,
                                  max_distance: float) -> List[Tuple[str, float]]:
        """근처 따릉이 대여소 찾기"""
        nearby = []
        
        for station_id, station in self.bike_stations.items():
            distance = self._haversine_distance(lat, lon, station.lat, station.lon)
            if distance <= max_distance:
                nearby.append((station_id, distance))
        
        return sorted(nearby, key=lambda x: x[1])
    
    def _calculate_road_route(self, start_lat: float, start_lon: float,
                             end_lat: float, end_lon: float,
                             mode: str) -> Tuple[float, float, List[Tuple[float, float]]]:
        """실제 도로망 기반 경로 계산"""
        
        if self.road_graph is None:
            # 그래프가 없으면 직선거리 * 보정계수
            distance = self._haversine_distance(start_lat, start_lon, end_lat, end_lon)
            road_distance = distance * 1.3  # 30% 우회
            
            if mode == 'walk':
                time_minutes = (road_distance / self.WALK_SPEED) * 60
            else:  # bike
                time_minutes = (road_distance / self.BIKE_SPEED) * 60
            
            coordinates = [(start_lat, start_lon), (end_lat, end_lon)]
            return time_minutes, road_distance, coordinates
        
        # 실제 그래프 사용
        try:
            # 가장 가까운 노드 찾기
            start_node = self._find_nearest_node(start_lat, start_lon)
            end_node = self._find_nearest_node(end_lat, end_lon)
            
            if start_node and end_node:
                # 최단 경로 계산
                if mode == 'walk':
                    path = nx.shortest_path(self.road_graph, start_node, end_node, 
                                          weight='walk_time')
                    time_minutes = nx.shortest_path_length(self.road_graph, start_node, end_node,
                                                         weight='walk_time')
                else:  # bike
                    path = nx.shortest_path(self.road_graph, start_node, end_node,
                                          weight='bike_time')
                    time_minutes = nx.shortest_path_length(self.road_graph, start_node, end_node,
                                                         weight='bike_time')
                
                # 실제 거리 계산
                total_distance = 0
                for i in range(len(path) - 1):
                    edge_data = self.road_graph.get_edge_data(path[i], path[i+1])
                    if edge_data and 'distance' in edge_data:
                        total_distance += edge_data['distance']
                
                coordinates = [(lat, lon) for lat, lon in path]
                return time_minutes, total_distance, coordinates
        
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        # 실패시 직선거리 사용
        distance = self._haversine_distance(start_lat, start_lon, end_lat, end_lon)
        road_distance = distance * 1.3
        
        if mode == 'walk':
            time_minutes = (road_distance / self.WALK_SPEED) * 60
        else:
            time_minutes = (road_distance / self.BIKE_SPEED) * 60
        
        coordinates = [(start_lat, start_lon), (end_lat, end_lon)]
        return time_minutes, road_distance, coordinates
    
    def _find_nearest_node(self, lat: float, lon: float) -> Optional[Tuple[float, float]]:
        """가장 가까운 그래프 노드 찾기"""
        if not self.road_graph:
            return None
        
        min_distance = float('inf')
        nearest_node = None
        
        # 샘플링으로 성능 최적화
        nodes = list(self.road_graph.nodes())
        sample_size = min(1000, len(nodes))
        sample_nodes = nodes[::len(nodes)//sample_size] if len(nodes) > sample_size else nodes
        
        for node in sample_nodes:
            distance = self._haversine_distance(lat, lon, node[0], node[1])
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _calculate_bike_cost(self, bike_time_minutes: float) -> float:
        """따릉이 요금 계산"""
        if bike_time_minutes <= 30:
            return self.BIKE_BASE_FARE
        else:
            extra_time = bike_time_minutes - 30
            extra_periods = math.ceil(extra_time / 30)
            return self.BIKE_BASE_FARE + (extra_periods * 1000)
    
    def _find_simple_transit_route(self, origin_stop_id: str, dest_lat: float, dest_lon: float,
                                  departure_time: int) -> List[Dict]:
        """간단한 대중교통 경로 찾기"""
        results = []
        
        if origin_stop_id not in self.stops:
            return results
        
        # 목적지 근처 정류장들
        dest_stops = self._find_nearby_stops_from_point(dest_lat, dest_lon, 0.8)
        
        for dest_stop_id, dest_distance in dest_stops[:5]:
            if dest_stop_id not in self.stops:
                continue
            
            dest_stop = self.stops[dest_stop_id]
            egress_time = (dest_distance / self.WALK_SPEED) * 60
            
            # 간단한 최단경로 (직접 연결 또는 1회 환승)
            routes = self._find_direct_routes(origin_stop_id, dest_stop_id)
            
            for route_info in routes:
                if route_info['arrival_time'] >= departure_time:
                    travel_time = route_info['arrival_time'] - departure_time
                    
                    results.append({
                        'dest_stop_id': dest_stop_id,
                        'dest_stop_name': dest_stop.stop_name,
                        'duration': travel_time,
                        'egress_time': egress_time,
                        'transfers': route_info.get('transfers', 0),
                        'cost': route_info.get('cost', self.BASE_TRANSIT_FARE),
                        'route_info': route_info.get('route_name', '대중교통'),
                        'route_id': route_info.get('route_id'),
                        'route_color': route_info.get('route_color', '#0066CC'),
                        'distance': route_info.get('distance', 3.0)
                    })
        
        return sorted(results, key=lambda x: x['duration'])[:3]
    
    def _find_direct_routes(self, origin_stop_id: str, dest_stop_id: str) -> List[Dict]:
        """두 정류장간 직접 연결 노선 찾기"""
        routes = []
        
        origin_routes = set(self.stop_to_routes.get(origin_stop_id, []))
        dest_routes = set(self.stop_to_routes.get(dest_stop_id, []))
        
        # 공통 노선 (직접 연결)
        common_routes = origin_routes.intersection(dest_routes)
        
        for route_id in common_routes:
            if route_id in self.routes:
                route = self.routes[route_id]
                
                # 노선 패턴에서 순서 확인
                try:
                    origin_idx = route.stop_pattern.index(origin_stop_id)
                    dest_idx = route.stop_pattern.index(dest_stop_id)
                    
                    if dest_idx > origin_idx:  # 올바른 방향
                        # 예상 소요시간 (역 수 * 2분)
                        station_count = dest_idx - origin_idx
                        estimated_time = station_count * 2
                        
                        routes.append({
                            'route_id': route_id,
                            'route_name': route.route_name,
                            'route_color': route.route_color,
                            'transfers': 0,
                            'cost': route.base_fare,
                            'arrival_time': int(time.time() / 60) + estimated_time,  # 임시
                            'distance': station_count * 0.8  # 역간 평균 거리 추정
                        })
                except ValueError:
                    continue
        
        return routes
    
    def _reconstruct_journey_from_raptor(self, raptor_result: Dict, 
                                        origin_lat: float, origin_lon: float,
                                        dest_lat: float, dest_lon: float,
                                        dep_time: int) -> Optional[Journey]:
        """RAPTOR 결과를 Journey 객체로 변환"""
        
        if not raptor_result.get('trip_id') or not raptor_result.get('route_id'):
            return None
        
        route_id = raptor_result['route_id']
        route = self.routes.get(route_id)
        
        if not route:
            return None
        
        # 실제 노선명 가져오기
        route_name = self._get_clean_route_name(route)
        
        # 접근 및 하차 시간 계산
        access_time = max(3, raptor_result.get('total_time', 20) - raptor_result.get('arrival_time', dep_time) + dep_time)
        egress_time = raptor_result.get('egress_time', 3)
        
        total_time = raptor_result['total_time']
        total_cost = raptor_result['cost']
        transfers = raptor_result['transfers']
        
        # 실제 거리 추정 (삼성역-강남역 약 4km)
        estimated_distance = self._haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon) * 1.2
        
        # 세그먼트 구성
        segments = []
        
        # 1. 접근 세그먼트
        segments.append({
            'mode': 'walk',
            'from': '출발지',
            'to': '탑승역',
            'duration': max(3, access_time),
            'distance_km': round(estimated_distance * 0.1, 1),  # 전체의 10%
            'cost': 0,
            'route_info': '도보 접근'
        })
        
        # 2. 대중교통 세그먼트
        transit_time = max(5, raptor_result['arrival_time'] - dep_time - access_time)
        segments.append({
            'mode': 'transit',
            'from': '탑승역',
            'to': raptor_result['dest_stop_name'],
            'duration': transit_time,
            'distance_km': round(estimated_distance * 0.8, 1),  # 전체의 80%
            'cost': total_cost,
            'route_info': route_name,
            'route_id': route_id,
            'route_color': route.route_color,
            'route_type': route.route_type
        })
        
        # 3. 하차 세그먼트
        segments.append({
            'mode': 'walk',
            'from': raptor_result['dest_stop_name'],
            'to': '목적지',
            'duration': max(2, int(egress_time)),
            'distance_km': round(estimated_distance * 0.1, 1),  # 전체의 10%
            'cost': 0,
            'route_info': '도보'
        })
        
        # 좌표 생성 (간단화)
        coordinates = [
            (origin_lat, origin_lon),
            (dest_lat, dest_lon)
        ]
        
        return Journey(
            total_time=max(10, total_time),  # 최소 10분
            total_distance=round(estimated_distance, 1),
            total_cost=total_cost,
            total_transfers=transfers,
            departure_time=dep_time,
            arrival_time=dep_time + max(10, total_time),
            journey_type="transit",
            route_coordinates=coordinates,
            segments=segments
        )
    
    def _get_clean_route_name(self, route: Route) -> str:
        """깨끗한 노선명 반환"""
        if not route:
            return "대중교통"
        
        route_name = route.route_name
        
        # 지하철 노선명 정리
        if route.route_type == 1:  # 지하철
            if '2' in route_name or '2호선' in route_name:
                return "지하철 2호선"
            elif '7' in route_name or '7호선' in route_name:
                return "지하철 7호선"
            elif '9' in route_name or '9호선' in route_name:
                return "지하철 9호선"
            elif '분당' in route_name:
                return "분당선"
            elif '신분당' in route_name:
                return "신분당선"
            else:
                return f"지하철 {route_name}"
        else:  # 버스
            # 숫자만 추출
            import re
            numbers = re.findall(r'\d+', route_name)
            if numbers:
                return f"{numbers[0]}번 버스"
            else:
                return f"{route_name} 버스"
    
    # =============================================================================
    # Pareto 최적화 및 경로 다양성
    # =============================================================================
    
    def _pareto_optimize(self, journeys: List[Journey], preferences: Dict) -> List[Journey]:
        """Pareto 최적화"""
        if not journeys:
            return []
        
        print(f"     ⚖️ Pareto 최적화: {len(journeys)}개 경로 입력")
        
        # 1. 교통수단별 그룹화
        groups = {
            'walk': [],
            'bike': [],
            'transit': [],
            'mixed': []
        }
        
        for journey in journeys:
            groups[journey.journey_type].append(journey)
        
        pareto_optimal = []
        
        # 2. 각 그룹에서 Pareto 최적 선택
        for group_type, group_journeys in groups.items():
            if not group_journeys:
                continue
            
            # 시간 기준 최고
            best_time = min(group_journeys, key=lambda x: x.total_time)
            pareto_optimal.append(best_time)
            
            # 비용 기준 최고 (다른 경로인 경우)
            best_cost = min(group_journeys, key=lambda x: x.total_cost)
            if best_cost != best_time:
                pareto_optimal.append(best_cost)
            
            # 환승 기준 최고 (다른 경로인 경우)
            best_transfer = min(group_journeys, key=lambda x: x.total_transfers)
            if best_transfer not in [best_time, best_cost]:
                pareto_optimal.append(best_transfer)
        
        # 3. 중복 제거
        unique_journeys = []
        for journey in pareto_optimal:
            is_duplicate = False
            for existing in unique_journeys:
                if self._are_journeys_similar(journey, existing):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_journeys.append(journey)
        
        # 4. 다중기준 점수 계산
        for journey in unique_journeys:
            journey.pareto_rank = self._calculate_multi_criteria_score(journey, preferences)
        
        print(f"     ✅ Pareto 결과: {len(unique_journeys)}개 경로")
        return sorted(unique_journeys, key=lambda x: x.pareto_rank)
    
    def _are_journeys_similar(self, journey1: Journey, journey2: Journey) -> bool:
        """두 경로가 유사한지 판단"""
        time_diff = abs(journey1.total_time - journey2.total_time)
        cost_diff = abs(journey1.total_cost - journey2.total_cost)
        transfer_diff = abs(journey1.total_transfers - journey2.total_transfers)
        
        return (time_diff <= 5 and cost_diff <= 200 and transfer_diff <= 1 and
                journey1.journey_type == journey2.journey_type)
    
    def _calculate_multi_criteria_score(self, journey: Journey, preferences: Dict) -> float:
        """다중기준 점수 계산 (낮을수록 좋음)"""
        time_weight = preferences.get('time_weight', 0.5)
        cost_weight = preferences.get('cost_weight', 0.2)
        transfer_weight = preferences.get('transfer_weight', 0.3)
        
        # 정규화를 위한 기준값
        max_time = 60  # 60분
        max_cost = 3000  # 3000원
        max_transfers = 3  # 3회
        
        time_score = min(journey.total_time / max_time, 1.0)
        cost_score = min(journey.total_cost / max_cost, 1.0)
        transfer_score = min(journey.total_transfers / max_transfers, 1.0)
        
        total_score = (time_score * time_weight + 
                      cost_score * cost_weight + 
                      transfer_score * transfer_weight)
        
        return total_score
    
    def _diversify_routes(self, journeys: List[Journey], max_routes: int) -> List[Journey]:
        """경로 다양성 확보"""
        if len(journeys) <= max_routes:
            return journeys
        
        diversified = []
        
        # 1. 각 교통수단별 최고 경로 보장
        type_best = {}
        for journey in journeys:
            if journey.journey_type not in type_best:
                type_best[journey.journey_type] = journey
            elif journey.pareto_rank < type_best[journey.journey_type].pareto_rank:
                type_best[journey.journey_type] = journey
        
        diversified.extend(type_best.values())
        
        # 2. 나머지 슬롯을 점수순으로 채움
        remaining_slots = max_routes - len(diversified)
        remaining_journeys = [j for j in journeys if j not in diversified]
        
        diversified.extend(remaining_journeys[:remaining_slots])
        
        return sorted(diversified, key=lambda x: x.pareto_rank)
    
    # =============================================================================
    # 결과 출력 및 시각화 준비
    # =============================================================================
    
    def print_journey_summary(self, journeys: List[Journey]):
        """경로 요약 출력"""
        if not journeys:
            print("❌ 경로를 찾을 수 없습니다.")
            return
        
        print(f"\n🎉 총 {len(journeys)}개 최적 경로:")
        print("=" * 80)
        
        for i, journey in enumerate(journeys, 1):
            print(f"\n{'='*20} 경로 {i} ({'⭐' * min(3, 4-journey.pareto_rank)}) {'='*20}")
            print(f"🚶‍♂️ 교통수단: {self._get_transport_emoji(journey.journey_type)} {journey.journey_type.upper()}")
            print(f"⏱️  총 소요시간: {journey.total_time//60}시간 {journey.total_time%60}분")
            print(f"💰 총 요금: {journey.total_cost:,.0f}원")
            print(f"🔄 환승횟수: {journey.total_transfers}회")
            print(f"📏 총 거리: {journey.total_distance:.1f}km")
            print(f"🕐 출발: {self._minutes_to_time(journey.departure_time)} → 도착: {self._minutes_to_time(journey.arrival_time)}")
            
            print(f"\n📍 상세 경로:")
            for j, segment in enumerate(journey.segments, 1):
                duration_str = f"{segment['duration']}분"
                cost_str = f" ({segment.get('cost', 0):,.0f}원)" if segment.get('cost', 0) > 0 else ""
                distance_str = f" {segment.get('distance_km', 0):.1f}km" if segment.get('distance_km', 0) > 0 else ""
                
                mode_emoji = self._get_mode_emoji(segment['mode'])
                print(f"  {j}. {mode_emoji} {segment['route_info']}: {segment['from']} → {segment['to']}")
                print(f"     소요시간: {duration_str}{cost_str}{distance_str}")
            
            print("-" * 80)
    
    def _get_transport_emoji(self, journey_type: str) -> str:
        """교통수단 이모지"""
        emoji_map = {
            'walk': '🚶‍♂️',
            'bike': '🚲',
            'transit': '🚇',
            'mixed': '🔄'
        }
        return emoji_map.get(journey_type, '🚌')
    
    def _get_mode_emoji(self, mode: str) -> str:
        """이동수단 이모지"""
        emoji_map = {
            'walk': '🚶‍♂️',
            'bike': '🚲',
            'bike_rental': '🔄',
            'bike_return': '🔄',
            'transit': '🚇',
            'bus': '🚌',
            'subway': '🚇'
        }
        return emoji_map.get(mode, '🚌')
    
    def _minutes_to_time(self, minutes: int) -> str:
        """분을 시간 문자열로 변환"""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def get_journey_geojson(self, journeys: List[Journey]) -> Dict:
        """경로를 GeoJSON 형식으로 변환 (시각화용)"""
        features = []
        
        colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF']
        
        for i, journey in enumerate(journeys):
            color = colors[i % len(colors)]
            
            if journey.route_coordinates and len(journey.route_coordinates) > 1:
                # 경로 라인
                line_feature = {
                    "type": "Feature",
                    "properties": {
                        "journey_id": i + 1,
                        "journey_type": journey.journey_type,
                        "total_time": journey.total_time,
                        "total_cost": journey.total_cost,
                        "total_transfers": journey.total_transfers,
                        "color": color,
                        "weight": 5,
                        "opacity": 0.8
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat] for lat, lon in journey.route_coordinates]
                    }
                }
                features.append(line_feature)
            
            # 세그먼트별 포인트들
            for j, segment in enumerate(journey.segments):
                if 'coordinates' in segment and segment['coordinates']:
                    for k, (lat, lon) in enumerate(segment['coordinates']):
                        point_feature = {
                            "type": "Feature",
                            "properties": {
                                "journey_id": i + 1,
                                "segment_id": j,
                                "point_id": k,
                                "mode": segment['mode'],
                                "route_info": segment.get('route_info', ''),
                                "marker_color": color,
                                "marker_size": "small" if k > 0 and k < len(segment['coordinates'])-1 else "medium"
                            },
                            "geometry": {
                                "type": "Point",
                                "coordinates": [lon, lat]
                            }
                        }
                        features.append(point_feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def save_results(self, journeys: List[Journey], output_path: str):
        """결과 저장"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # 1. JSON 형태로 저장
        results_data = []
        for i, journey in enumerate(journeys):
            journey_data = {
                'journey_id': i + 1,
                'journey_type': journey.journey_type,
                'total_time': journey.total_time,
                'total_distance': journey.total_distance,
                'total_cost': journey.total_cost,
                'total_transfers': journey.total_transfers,
                'departure_time': journey.departure_time,
                'arrival_time': journey.arrival_time,
                'pareto_rank': journey.pareto_rank,
                'segments': journey.segments,
                'route_coordinates': journey.route_coordinates
            }
            results_data.append(journey_data)
        
        with open(output_dir / 'journey_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 2. GeoJSON 저장
        geojson_data = self.get_journey_geojson(journeys)
        with open(output_dir / 'journey_routes.geojson', 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2)
        
        # 3. 요약 통계 저장
        summary = {
            'total_journeys': len(journeys),
            'journey_types': {
                journey_type: sum(1 for j in journeys if j.journey_type == journey_type)
                for journey_type in ['walk', 'bike', 'transit', 'mixed']
            },
            'avg_time': sum(j.total_time for j in journeys) / len(journeys) if journeys else 0,
            'avg_cost': sum(j.total_cost for j in journeys) / len(journeys) if journeys else 0,
            'avg_transfers': sum(j.total_transfers for j in journeys) / len(journeys) if journeys else 0
        }
        
        with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 결과 저장 완료: {output_dir}/")


# =============================================================================
# 사용 예제
# =============================================================================

if __name__ == "__main__":
    print("🚀 강남구 Multi-modal RAPTOR v3.0 테스트")
    print("=" * 60)
    
    # 데이터 경로 설정
    data_path = "C:\\Users\\sec\\Desktop\\kim\\학회\\GTFS\\code\\multimodal_raptor_project\\gangnam_multimodal_raptor_data_with_real_roads"

    try:
        # RAPTOR 엔진 초기화
        raptor = GangnamMultiModalRAPTOR(data_path)
        
        # 테스트 시나리오 1: 삼성역 → 강남역
        print("\n📍 테스트 시나리오 1: 삼성역 → 강남역")
        origin_lat, origin_lon = 37.51579174292475, 127.02039435436643  # 삼성역 근처
        dest_lat, dest_lon = 37.49985645759325, 127.04146988383535      # 강남역 근처
        
        journeys = raptor.find_routes(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            departure_time="09:30",
            max_routes=5,
            include_bike=True,
            user_preferences={
                'time_weight': 0.6,
                'cost_weight': 0.2,
                'transfer_weight': 0.2,
                'max_walk_time': 12,
                'max_bike_time': 18
            }
        )
        
        # 결과 출력
        raptor.print_journey_summary(journeys)
        
        # 결과 저장
        raptor.save_results(journeys, "test_results")
        
        print("\n🎯 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()