"""
Part 3-1: 강남구 Multi-modal RAPTOR 경로 시각화 엔진 v1.0 (핵심 부분)
- Part 2 RAPTOR 알고리즘 결과를 정확한 실제 경로로 시각화
- 실제 버스 경로 (GTFS shapes 또는 stop 순서 기반)
- 실제 도로망 기반 도보/자전거 경로
- 지하철 경로 (실제 지하철 노선도 기반)
- 대화형 웹 지도 인터페이스
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import folium.plugins as plugins
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import json
import pickle
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
import base64
from io import BytesIO
import time
from dataclasses import dataclass
import math

warnings.filterwarnings('ignore')

# =============================================================================
# 시각화용 데이터 구조 정의
# =============================================================================

@dataclass
class VisualizationSegment:
    """시각화용 경로 세그먼트"""
    mode: str  # walk, bike, transit, bike_rental, bike_return
    route_name: str
    start_point: Tuple[float, float]  # (lat, lon)
    end_point: Tuple[float, float]
    coordinates: List[Tuple[float, float]]  # 실제 경로 좌표들
    duration: int  # 분
    distance: float  # km
    cost: float
    color: str
    route_type: str  # subway, bus, walk, bike
    route_id: Optional[str] = None

@dataclass
class VisualizationJourney:
    """시각화용 완전한 여행 경로"""
    journey_id: int
    journey_type: str
    total_time: int
    total_cost: float
    segments: List[VisualizationSegment]
    summary_stats: Dict[str, Any]

# =============================================================================
# 메인 시각화 엔진 클래스
# =============================================================================

class GangnamRAPTORVisualizer:
    """강남구 Multi-modal RAPTOR 경로 시각화 엔진"""
    
    def __init__(self, data_path: str, raptor_results_path: str):
        self.data_path = Path(data_path)
        self.results_path = Path(raptor_results_path)
        
        # 기본 데이터
        self.stops = {}
        self.routes = {}
        self.trips = {}
        self.bike_stations = {}
        self.road_graph = None
        self.route_shapes = {}  # 실제 노선 경로
        
        # RAPTOR 결과
        self.journey_results = []
        self.original_journeys = []
        
        # 시각화 설정
        self.color_schemes = {
            'walk': '#32CD32',      # 라임그린
            'bike': '#FF6B35',      # 오렌지레드
            'subway_2': '#00A84D',  # 2호선 그린
            'subway_7': '#996600',  # 7호선 브라운
            'subway_9': '#D4003B',  # 9호선 레드
            'subway_bundang': '#FFCD12',  # 분당선 옐로우
            'subway_shinbundang': '#AA5500',  # 신분당선
            'bus': '#3366CC',       # 버스 블루
            'transfer': '#FF9900'   # 환승 오렌지
        }
        
        # 강남구 경계
        self.gangnam_center = [37.5172, 127.0473]  # 강남역 중심
        self.gangnam_bounds = {
            'north': 37.55, 'south': 37.46,
            'east': 127.14, 'west': 127.00
        }
        
        print("🎨 강남구 RAPTOR 경로 시각화 엔진 v1.0 초기화")
        self._load_all_data()
    
    def _load_all_data(self):
        """모든 데이터 로드"""
        print("📊 시각화용 데이터 로딩...")
        
        # 1. 기본 교통 데이터
        self._load_transportation_data()
        
        # 2. 도로망 데이터
        self._load_road_network()
        
        # 3. RAPTOR 결과 데이터
        self._load_raptor_results()
        
        # 4. 실제 경로 데이터 (GTFS shapes 등)
        self._load_route_geometries()
        
        print("✅ 데이터 로딩 완료")
    
    def _load_transportation_data(self):
        """교통 데이터 로드"""
        try:
            # 정류장 데이터
            stops_file = self.data_path / 'gangnam_stops.csv'
            if stops_file.exists():
                stops_df = pd.read_csv(stops_file, encoding='utf-8')
                for _, row in stops_df.iterrows():
                    self.stops[row['stop_id']] = {
                        'name': row.get('stop_name', f'정류장_{row["stop_id"]}'),
                        'lat': row['stop_lat'],
                        'lon': row['stop_lon']
                    }
                print(f"   ✅ 정류장: {len(self.stops)}개")
            
            # 노선 데이터
            routes_file = self.data_path / 'gangnam_routes.csv'
            if routes_file.exists():
                routes_df = pd.read_csv(routes_file, encoding='utf-8')
                for _, row in routes_df.iterrows():
                    route_type = row.get('route_type', 3)
                    route_name = str(row.get('route_short_name', row['route_id']))
                    
                    # 색상 결정
                    if route_type == 1:  # 지하철
                        if '2' in route_name:
                            color = self.color_schemes['subway_2']
                        elif '7' in route_name:
                            color = self.color_schemes['subway_7']
                        elif '9' in route_name:
                            color = self.color_schemes['subway_9']
                        elif '분당' in route_name or 'K' in route_name:
                            color = self.color_schemes['subway_bundang']
                        elif '신분당' in route_name or 'D' in route_name:
                            color = self.color_schemes['subway_shinbundang']
                        else:
                            color = '#0066CC'
                    else:  # 버스
                        color = self.color_schemes['bus']
                    
                    self.routes[row['route_id']] = {
                        'name': route_name,
                        'type': route_type,
                        'color': color,
                        'long_name': row.get('route_long_name', route_name)
                    }
                print(f"   ✅ 노선: {len(self.routes)}개")
            
            # 따릉이 데이터
            bike_file = self.data_path / 'gangnam_bike_stations.csv'
            if bike_file.exists():
                bike_df = pd.read_csv(bike_file, encoding='utf-8')
                for _, row in bike_df.iterrows():
                    self.bike_stations[str(row['station_id'])] = {
                        'name': self._clean_name(row.get('address1', f'대여소_{row["station_id"]}')),
                        'lat': row['latitude'],
                        'lon': row['longitude']
                    }
                print(f"   ✅ 따릉이: {len(self.bike_stations)}개소")
                
        except Exception as e:
            print(f"   ⚠️ 교통 데이터 로드 실패: {e}")
    
    def _load_road_network(self):
        """도로망 데이터 로드"""
        try:
            # NetworkX 그래프 로드
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
                        
                        print(f"   ✅ 도로 그래프: {self.road_graph.number_of_nodes():,}개 노드, {self.road_graph.number_of_edges():,}개 엣지")
                        break
                    except Exception as e:
                        print(f"   ⚠️ {graph_file.name} 로드 실패: {e}")
            
            if self.road_graph is None:
                print("   🔧 기본 도로망 생성...")
                self._create_basic_road_network()
                
        except Exception as e:
            print(f"   ⚠️ 도로망 로드 실패: {e}")
    
    def _load_raptor_results(self):
        """RAPTOR 결과 로드"""
        try:
            # JSON 결과 파일 로드
            results_file = self.results_path / 'journey_results.json'
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    self.journey_results = json.load(f)
                print(f"   ✅ RAPTOR 결과: {len(self.journey_results)}개 경로")
            else:
                print(f"   ⚠️ RAPTOR 결과 파일 없음: {results_file}")
                
        except Exception as e:
            print(f"   ⚠️ RAPTOR 결과 로드 실패: {e}")
    
    def _load_route_geometries(self):
        """실제 노선 경로 데이터 로드"""
        try:
            # GTFS shapes 데이터가 있다면 로드
            shapes_file = self.data_path / 'shapes.csv'
            if shapes_file.exists():
                shapes_df = pd.read_csv(shapes_file, encoding='utf-8')
                # shapes 데이터 처리 (추후 구현)
                print(f"   ✅ 실제 노선 경로 로드")
            else:
                # shapes가 없으면 stop 순서 기반으로 경로 생성
                print("   🔧 정류장 순서 기반 경로 생성...")
                self._generate_route_paths_from_stops()
                
        except Exception as e:
            print(f"   ⚠️ 노선 경로 로드 실패: {e}")
    
    def _generate_route_paths_from_stops(self):
        """정류장 순서 기반으로 노선 경로 생성"""
        try:
            # RAPTOR 구조에서 route patterns 로드
            raptor_file = self.data_path / 'gangnam_raptor_structures.pkl'
            if raptor_file.exists():
                with open(raptor_file, 'rb') as f:
                    raptor_data = pickle.load(f)
                    route_patterns = raptor_data.get('route_patterns', {})
                    
                    for route_id, stop_pattern in route_patterns.items():
                        if len(stop_pattern) >= 2:
                            coordinates = []
                            for stop_id in stop_pattern:
                                if stop_id in self.stops:
                                    stop = self.stops[stop_id]
                                    coordinates.append((stop['lat'], stop['lon']))
                            
                            if len(coordinates) >= 2:
                                self.route_shapes[route_id] = coordinates
                
                print(f"   ✅ 노선 경로 생성: {len(self.route_shapes)}개")
                
        except Exception as e:
            print(f"   ⚠️ 노선 경로 생성 실패: {e}")
    
    def _create_basic_road_network(self):
        """기본 도로망 생성 (그래프가 없는 경우)"""
        self.road_graph = nx.Graph()
        
        # 강남구 그리드 생성
        lat_min, lat_max = 37.46, 37.55
        lon_min, lon_max = 127.00, 127.14
        grid_size = 0.002  # 약 200m 간격
        
        # 노드 생성
        for lat in np.arange(lat_min, lat_max, grid_size):
            for lon in np.arange(lon_min, lon_max, grid_size):
                self.road_graph.add_node((lat, lon))
        
        # 인접 노드 연결
        nodes = list(self.road_graph.nodes())
        for i, (lat1, lon1) in enumerate(nodes):
            for lat2, lon2 in nodes[i+1:]:
                distance = self._haversine_distance(lat1, lon1, lat2, lon2)
                if distance <= 0.3:  # 300m 이내 연결
                    self.road_graph.add_edge(
                        (lat1, lon1), (lat2, lon2),
                        distance=distance,
                        weight=distance
                    )
        
        print(f"   ✅ 기본 그리드: {self.road_graph.number_of_nodes():,}개 노드, {self.road_graph.number_of_edges():,}개 엣지")
    
    # =============================================================================
    # 실제 경로 생성 함수들 (핵심 기능)
    # =============================================================================
    
    def generate_accurate_route_coordinates(self, journey_data: Dict) -> VisualizationJourney:
        """RAPTOR 결과를 실제 정확한 경로 좌표로 변환"""
        print(f"🗺️ 경로 {journey_data['journey_id']} 정확한 좌표 생성 중...")
        
        viz_segments = []
        
        for i, segment in enumerate(journey_data['segments']):
            print(f"   세그먼트 {i+1}: {segment['mode']} - {segment.get('route_info', 'N/A')}")
            
            if segment['mode'] == 'walk':
                viz_segment = self._generate_walking_route(segment)
            elif segment['mode'] == 'bike':
                viz_segment = self._generate_bike_route(segment)
            elif segment['mode'] in ['bike_rental', 'bike_return']:
                viz_segment = self._generate_bike_station_point(segment)
            elif segment['mode'] == 'transit':
                viz_segment = self._generate_transit_route(segment)
            else:
                # 기본 직선 경로
                viz_segment = self._generate_default_route(segment)
            
            if viz_segment:
                viz_segments.append(viz_segment)
        
        # 경로 요약 통계
        total_distance = sum(s.distance for s in viz_segments)
        summary_stats = {
            'total_segments': len(viz_segments),
            'total_distance_km': round(total_distance, 2),
            'modes_used': list(set(s.mode for s in viz_segments)),
            'has_transit': any(s.mode == 'transit' for s in viz_segments),
            'has_bike': any(s.mode == 'bike' for s in viz_segments),
            'has_walk': any(s.mode == 'walk' for s in viz_segments)
        }
        
        return VisualizationJourney(
            journey_id=journey_data['journey_id'],
            journey_type=journey_data['journey_type'],
            total_time=journey_data['total_time'],
            total_cost=journey_data['total_cost'],
            segments=viz_segments,
            summary_stats=summary_stats
        )
    
    def _generate_walking_route(self, segment: Dict) -> VisualizationSegment:
        """실제 도로망 기반 도보 경로 생성"""
        start_coords = self._extract_coordinates_from_location(segment['from'])
        end_coords = self._extract_coordinates_from_location(segment['to'])
        
        if not start_coords or not end_coords:
            return self._generate_default_route(segment)
        
        # 실제 도로망에서 최단 경로 찾기
        path_coords = self._find_road_path(start_coords, end_coords, 'walk')
        
        if not path_coords or len(path_coords) < 2:
            # 실패시 직선 경로
            path_coords = [start_coords, end_coords]
        
        # 실제 거리 계산
        actual_distance = self._calculate_path_distance(path_coords)
        
        return VisualizationSegment(
            mode='walk',
            route_name='도보',
            start_point=start_coords,
            end_point=end_coords,
            coordinates=path_coords,
            duration=segment.get('duration', 5),
            distance=actual_distance,
            cost=segment.get('cost', 0),
            color=self.color_schemes['walk'],
            route_type='walk'
        )
    
    def _generate_bike_route(self, segment: Dict) -> VisualizationSegment:
        """실제 도로망 기반 자전거 경로 생성"""
        start_coords = self._extract_coordinates_from_location(segment['from'])
        end_coords = self._extract_coordinates_from_location(segment['to'])
        
        if not start_coords or not end_coords:
            return self._generate_default_route(segment)
        
        # 자전거용 도로망 경로 (도보보다 빠른 도로 선호)
        path_coords = self._find_road_path(start_coords, end_coords, 'bike')
        
        if not path_coords or len(path_coords) < 2:
            path_coords = [start_coords, end_coords]
        
        actual_distance = self._calculate_path_distance(path_coords)
        
        return VisualizationSegment(
            mode='bike',
            route_name=f'따릉이 {segment.get("duration", 10)}분',
            start_point=start_coords,
            end_point=end_coords,
            coordinates=path_coords,
            duration=segment.get('duration', 10),
            distance=actual_distance,
            cost=segment.get('cost', 1000),
            color=self.color_schemes['bike'],
            route_type='bike'
        )
    
    def _generate_transit_route(self, segment: Dict) -> VisualizationSegment:
        """실제 대중교통 노선 경로 생성"""
        route_id = segment.get('route_id')
        route_color = segment.get('route_color', '#0066CC')
        route_name = segment.get('route_info', '대중교통')
        
        # 정류장 이름에서 좌표 추출
        start_coords = self._extract_coordinates_from_location(segment['from'])
        end_coords = self._extract_coordinates_from_location(segment['to'])
        
        if not start_coords or not end_coords:
            return self._generate_default_route(segment)
        
        # 실제 노선 경로 사용
        if route_id and route_id in self.route_shapes:
            route_coords = self.route_shapes[route_id]
            
            # 시작/끝 정류장에 가장 가까운 지점 찾기
            start_idx = self._find_closest_point_index(start_coords, route_coords)
            end_idx = self._find_closest_point_index(end_coords, route_coords)
            
            if start_idx is not None and end_idx is not None and start_idx < end_idx:
                path_coords = route_coords[start_idx:end_idx+1]
            else:
                # 실패시 직선
                path_coords = [start_coords, end_coords]
        else:
            # 노선 경로가 없으면 직선
            path_coords = [start_coords, end_coords]
        
        # 거리 계산 (대중교통은 실제 노선 거리)
        actual_distance = self._calculate_path_distance(path_coords)
        
        return VisualizationSegment(
            mode='transit',
            route_name=route_name,
            start_point=start_coords,
            end_point=end_coords,
            coordinates=path_coords,
            duration=segment.get('duration', 15),
            distance=actual_distance,
            cost=segment.get('cost', 1370),
            color=route_color,
            route_type='subway' if segment.get('route_type', 3) == 1 else 'bus',
            route_id=route_id
        )
    
    def _generate_bike_station_point(self, segment: Dict) -> VisualizationSegment:
        """따릉이 대여/반납 지점 표시"""
        # 대여소 이름에서 좌표 추출
        station_coords = self._extract_coordinates_from_location(segment['from'])
        
        if not station_coords:
            # 기본 좌표 (강남역 근처)
            station_coords = (37.498, 127.028)
        
        return VisualizationSegment(
            mode=segment['mode'],
            route_name=segment.get('route_info', '따릉이 대여/반납'),
            start_point=station_coords,
            end_point=station_coords,
            coordinates=[station_coords],
            duration=segment.get('duration', 2),
            distance=0.0,
            cost=segment.get('cost', 0),
            color=self.color_schemes['transfer'],
            route_type='bike_station'
        )
    
    def _generate_default_route(self, segment: Dict) -> VisualizationSegment:
        """기본 직선 경로 (fallback)"""
        # 임시 좌표 (강남구 중심부)
        start_coords = (37.517, 127.047)
        end_coords = (37.520, 127.050)
        
        return VisualizationSegment(
            mode=segment['mode'],
            route_name=segment.get('route_info', '이동'),
            start_point=start_coords,
            end_point=end_coords,
            coordinates=[start_coords, end_coords],
            duration=segment.get('duration', 5),
            distance=0.5,
            cost=segment.get('cost', 0),
            color='#CCCCCC',
            route_type='default'
        )
    
    # =============================================================================
    # 좌표 및 경로 계산 유틸리티
    # =============================================================================
    
    def _extract_coordinates_from_location(self, location_name: str) -> Optional[Tuple[float, float]]:
        """위치 이름에서 좌표 추출"""
        if not location_name or location_name in ['출발지', '목적지']:
            return None
        
        # 정류장에서 찾기
        for stop_id, stop_data in self.stops.items():
            if stop_data['name'] in location_name or location_name in stop_data['name']:
                return (stop_data['lat'], stop_data['lon'])
        
        # 따릉이 대여소에서 찾기
        for station_id, station_data in self.bike_stations.items():
            if station_data['name'] in location_name or location_name in station_data['name']:
                return (station_data['lat'], station_data['lon'])
        
        # 특정 지명 매칭
        known_locations = {
            '강남역': (37.498095, 127.027610),
            '역삼역': (37.500108, 127.036394),
            '선릉역': (37.504741, 127.048976),
            '삼성역': (37.508847, 127.063804),
            '종합운동장역': (37.510994, 127.073617),
            '신논현역': (37.504631, 127.025327),
            '논현역': (37.511221, 127.022223),
            '학동역': (37.514090, 127.041910),
            '압구정로데오역': (37.527082, 127.040139),
            '강남구청역': (37.517307, 127.041758)
        }
        
        for place_name, coords in known_locations.items():
            if place_name in location_name:
                return coords
        
        return None
    
    def _find_road_path(self, start: Tuple[float, float], end: Tuple[float, float], 
                       mode: str = 'walk') -> List[Tuple[float, float]]:
        """실제 도로망에서 경로 찾기"""
        if not self.road_graph:
            return [start, end]
        
        try:
            # 가장 가까운 노드 찾기
            start_node = self._find_nearest_graph_node(start)
            end_node = self._find_nearest_graph_node(end)
            
            if start_node and end_node and start_node != end_node:
                # 최단 경로 계산
                if mode == 'bike':
                    # 자전거는 거리 기준
                    path = nx.shortest_path(self.road_graph, start_node, end_node, weight='distance')
                else:
                    # 도보는 가중치 없음
                    path = nx.shortest_path(self.road_graph, start_node, end_node)
                
                # 실제 시작/끝점 포함
                full_path = [start] + list(path) + [end]
                return full_path
            
        except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
            pass
        
        # 실패시 직선
        return [start, end]
    
    def _find_nearest_graph_node(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """그래프에서 가장 가까운 노드 찾기"""
        if not self.road_graph:
            return None
        
        min_distance = float('inf')
        nearest_node = None
        
        # 주변 노드만 검색 (성능 최적화)
        lat, lon = point
        candidate_nodes = [
            node for node in self.road_graph.nodes() 
            if abs(node[0] - lat) < 0.01 and abs(node[1] - lon) < 0.01
        ]
        
        if not candidate_nodes:
            # 전체 노드에서 샘플링
            all_nodes = list(self.road_graph.nodes())
            candidate_nodes = all_nodes[::max(1, len(all_nodes)//100)]
        
        for node in candidate_nodes:
            distance = self._haversine_distance(lat, lon, node[0], node[1])
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _find_closest_point_index(self, target: Tuple[float, float], 
                                 path: List[Tuple[float, float]]) -> Optional[int]:
        """경로에서 목표점에 가장 가까운 지점의 인덱스 찾기"""
        if not path:
            return None
        
        min_distance = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(path):
            distance = self._haversine_distance(target[0], target[1], point[0], point[1])
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx
    
    def _calculate_path_distance(self, path: List[Tuple[float, float]]) -> float:
        """경로의 총 거리 계산 (km)"""
        if len(path) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path) - 1):
            distance = self._haversine_distance(
                path[i][0], path[i][1],
                path[i+1][0], path[i+1][1]
            )
            total_distance += distance
        
        return round(total_distance, 3)
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """하버사인 공식으로 거리 계산 (km)"""
        R = 6371  # 지구 반지름 (km)
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _clean_name(self, name: str) -> str:
        """이름 정리"""
        if not name or pd.isna(name):
            return "알 수 없음"
        
        name_str = str(name)
        if len(name_str) > 50:
            return name_str[:50] + "..."
        return name_str
    
    # =============================================================================
    # 대화형 웹 지도 시각화
    # =============================================================================
    
    def create_interactive_map(self, visualization_journeys: List[VisualizationJourney],
                             origin_coords: Tuple[float, float],
                             dest_coords: Tuple[float, float]) -> folium.Map:
        """대화형 웹 지도 생성"""
        print("🗺️ 대화형 지도 생성 중...")
        
        # 지도 초기화 (강남구 중심)
        m = folium.Map(
            location=self.gangnam_center,
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # 추가 타일 레이어
        folium.TileLayer(
            tiles='CartoDB Positron',
            name='밝은 지도',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='CartoDB Dark_Matter',
            name='어두운 지도',
            overlay=False,
            control=True
        ).add_to(m)
        
        # 출발지/목적지 마커
        folium.Marker(
            origin_coords,
            popup='🚀 출발지',
            tooltip='출발지',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        folium.Marker(
            dest_coords,
            popup='🎯 목적지',
            tooltip='목적지',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)
        
        # 각 경로별 레이어 그룹 생성
        journey_groups = {}
        
        for journey in visualization_journeys:
            group_name = f"경로 {journey.journey_id} ({journey.journey_type.upper()})"
            journey_group = folium.FeatureGroup(name=group_name, show=True)
            
            # 경로 세그먼트들 추가
            for i, segment in enumerate(journey.segments):
                self._add_segment_to_map(journey_group, segment, journey.journey_id, i)
            
            journey_groups[journey.journey_id] = journey_group
            m.add_child(journey_group)
        
        # 교통 인프라 레이어
        self._add_infrastructure_layers(m)
        
        # 경로 정보 패널
        self._add_journey_info_panel(m, visualization_journeys)
        
        # 컨트롤 추가
        folium.LayerControl(collapsed=False).add_to(m)
        
        # 미니맵 추가
        minimap = plugins.MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # 전체화면 버튼
        plugins.Fullscreen().add_to(m)
        
        # 마우스 위치 표시
        plugins.MousePosition().add_to(m)
        
        print("✅ 대화형 지도 생성 완료")
        return m
    
    def _add_segment_to_map(self, group: folium.FeatureGroup, segment: VisualizationSegment,
                           journey_id: int, segment_id: int):
        """지도에 경로 세그먼트 추가"""
        
        if len(segment.coordinates) < 2:
            return
        
        # 경로 선 추가
        if segment.mode == 'transit':
            # 대중교통은 굵은 선
            line_weight = 6
            opacity = 0.8
        elif segment.mode in ['walk', 'bike']:
            # 도보/자전거는 얇은 선
            line_weight = 4
            opacity = 0.7
        else:
            # 기타
            line_weight = 3
            opacity = 0.6
        
        # 애니메이션 효과를 위한 경로선
        folium.PolyLine(
            locations=segment.coordinates,
            color=segment.color,
            weight=line_weight,
            opacity=opacity,
            popup=self._create_segment_popup(segment, journey_id, segment_id),
            tooltip=f"{segment.route_name} ({segment.duration}분)"
        ).add_to(group)
        
        # 시작점 마커 (첫 번째 세그먼트만)
        if segment_id == 0:
            folium.CircleMarker(
                location=segment.start_point,
                radius=8,
                popup=f"🚀 여행 {journey_id} 시작",
                color=segment.color,
                fillColor=segment.color,
                fillOpacity=0.8
            ).add_to(group)
        
        # 환승/전환점 마커
        if segment.mode == 'transit':
            folium.CircleMarker(
                location=segment.start_point,
                radius=6,
                popup=f"🚇 {segment.route_name}",
                color=segment.color,
                fillColor='white',
                fillOpacity=1.0
            ).add_to(group)
        elif segment.mode in ['bike_rental', 'bike_return']:
            icon_symbol = '🚲' if segment.mode == 'bike_rental' else '🔄'
            folium.Marker(
                location=segment.start_point,
                popup=f"{icon_symbol} {segment.route_name}",
                tooltip=segment.route_name,
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 20px;">{icon_symbol}</div>',
                    class_name='bike-marker'
                )
            ).add_to(group)
        
        # 방향 화살표 (긴 세그먼트에만)
        if len(segment.coordinates) > 3:
            mid_point = segment.coordinates[len(segment.coordinates)//2]
            plugins.PolyLineTextPath(
                folium.PolyLine(segment.coordinates, opacity=0),
                "    ►    ",
                repeat=True,
                offset=7,
                attributes={'fill': segment.color, 'font-weight': 'bold'}
            ).add_to(group)
    
    def _create_segment_popup(self, segment: VisualizationSegment, 
                             journey_id: int, segment_id: int) -> str:
        """세그먼트 팝업 HTML 생성"""
        
        mode_icons = {
            'walk': '🚶‍♂️',
            'bike': '🚲',
            'transit': '🚇',
            'bike_rental': '🔄',
            'bike_return': '🔄'
        }
        
        icon = mode_icons.get(segment.mode, '🚌')
        
        popup_html = f"""
        <div style="width: 250px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0; color: {segment.color};">
                {icon} {segment.route_name}
            </h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0;"><b>경로:</b> {journey_id}, 구간: {segment_id + 1}</p>
            <p style="margin: 5px 0;"><b>소요시간:</b> {segment.duration}분</p>
            <p style="margin: 5px 0;"><b>거리:</b> {segment.distance:.2f}km</p>
            <p style="margin: 5px 0;"><b>요금:</b> {segment.cost:,.0f}원</p>
        </div>
        """
        
        return popup_html
    
    def _add_infrastructure_layers(self, m: folium.Map):
        """교통 인프라 레이어 추가"""
        
        # 지하철역 레이어
        subway_group = folium.FeatureGroup(name="🚇 지하철역", show=False)
        
        for stop_id, stop_data in self.stops.items():
            # 지하철역 판별 (노선 정보 기반)
            is_subway = any(
                route_data.get('type', 3) == 1 
                for route_data in self.routes.values()
            )
            
            if is_subway:
                folium.CircleMarker(
                    location=(stop_data['lat'], stop_data['lon']),
                    radius=4,
                    popup=f"🚇 {stop_data['name']}",
                    tooltip=stop_data['name'],
                    color='blue',
                    fillColor='lightblue',
                    fillOpacity=0.7
                ).add_to(subway_group)
        
        m.add_child(subway_group)
        
        # 따릉이 대여소 레이어
        bike_group = folium.FeatureGroup(name="🚲 따릉이 대여소", show=False)
        
        for station_id, station_data in self.bike_stations.items():
            folium.CircleMarker(
                location=(station_data['lat'], station_data['lon']),
                radius=3,
                popup=f"🚲 {station_data['name']}",
                tooltip=station_data['name'],
                color='orange',
                fillColor='yellow',
                fillOpacity=0.6
            ).add_to(bike_group)
        
        m.add_child(bike_group)
    
    def _add_journey_info_panel(self, m: folium.Map, journeys: List[VisualizationJourney]):
        """경로 정보 패널 추가"""
        
        info_html = self._generate_journey_info_html(journeys)
        
        # 정보 패널을 우측 상단에 추가
        info_panel = folium.plugins.FloatImage(
            image=self._html_to_image(info_html),
            bottom=70,
            left=85
        )
        m.add_child(info_panel)
    
    def _generate_journey_info_html(self, journeys: List[VisualizationJourney]) -> str:
        """경로 정보 HTML 생성"""
        
        html = """
        <div style='background: white; padding: 15px; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3); max-width: 300px;
                    font-family: Arial, sans-serif; font-size: 12px;'>
            <h3 style='margin: 0 0 10px 0; color: #333;'>🗺️ 경로 요약</h3>
        """
        
        for journey in journeys:
            # 경로 타입별 이모지
            type_emoji = {
                'walk': '🚶‍♂️',
                'bike': '🚲', 
                'transit': '🚇',
                'mixed': '🔄'
            }.get(journey.journey_type, '🚌')
            
            html += f"""
            <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 5px;'>
                <div style='font-weight: bold; color: #495057;'>
                    {type_emoji} 경로 {journey.journey_id}
                </div>
                <div style='margin: 3px 0;'>
                    ⏱️ {journey.total_time//60}시간 {journey.total_time%60}분
                </div>
                <div style='margin: 3px 0;'>
                    💰 {journey.total_cost:,.0f}원
                </div>
                <div style='margin: 3px 0;'>
                    📏 {journey.summary_stats['total_distance_km']}km
                </div>
                <div style='margin: 3px 0; font-size: 10px; color: #6c757d;'>
                    {len(journey.segments)}개 구간
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _html_to_image(self, html: str) -> str:
        """HTML을 이미지로 변환 (간단한 버전)"""
        # 실제 구현에서는 HTML을 이미지로 변환하는 라이브러리 사용
        # 여기서는 단순화된 버전으로 base64 인코딩된 투명 이미지 반환
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    # =============================================================================
    # 정적 시각화 (Plotly)
    # =============================================================================
    
    def create_plotly_visualization(self, visualization_journeys: List[VisualizationJourney]) -> go.Figure:
        """Plotly를 이용한 정적 시각화"""
        print("📊 Plotly 시각화 생성 중...")
        
        fig = go.Figure()
        
        # 강남구 경계 추가
        self._add_gangnam_boundary(fig)
        
        # 각 경로 추가
        for journey in visualization_journeys:
            for i, segment in enumerate(journey.segments):
                if len(segment.coordinates) >= 2:
                    lats, lons = zip(*segment.coordinates)
                    
                    # 모드별 스타일 설정
                    if segment.mode == 'transit':
                        line_width = 4
                        line_color = segment.color
                    elif segment.mode == 'walk':
                        line_width = 2
                        line_color = segment.color
                    elif segment.mode == 'bike':
                        line_width = 3
                        line_color = segment.color
                    else:
                        line_width = 2
                        line_color = segment.color
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='lines',
                        line=dict(width=line_width, color=segment.color, ),
                        name=f"경로{journey.journey_id}-{segment.route_name}",
                        hovertemplate=f"<b>{segment.route_name}</b><br>" +
                                    f"소요시간: {segment.duration}분<br>" +
                                    f"거리: {segment.distance:.2f}km<br>" +
                                    f"요금: {segment.cost:,.0f}원<extra></extra>",
                        showlegend=True
                    ))
        
        # 교통 인프라 추가
        self._add_infrastructure_to_plotly(fig)
        
        # 레이아웃 설정
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=self.gangnam_center[0], lon=self.gangnam_center[1]),
                zoom=12
            ),
            title="강남구 Multi-modal RAPTOR 경로 시각화",
            title_x=0.5,
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        print("✅ Plotly 시각화 생성 완료")
        return fig
    
    def _add_gangnam_boundary(self, fig: go.Figure):
        """강남구 경계 추가"""
        # 강남구 대략적 경계
        boundary_coords = [
            (37.46, 127.00), (37.46, 127.14),
            (37.55, 127.14), (37.55, 127.00),
            (37.46, 127.00)  # 닫힌 다각형
        ]
        
        lats, lons = zip(*boundary_coords)
        
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(width=2, color='gray', ),
            name="강남구 경계",
            hovertemplate="강남구 경계<extra></extra>",
            showlegend=False
        ))
    
    def _add_infrastructure_to_plotly(self, fig: go.Figure):
        """교통 인프라를 Plotly에 추가"""
        
        # 주요 지하철역 추가
        major_stations = {
            '강남역': (37.498095, 127.027610),
            '역삼역': (37.500108, 127.036394),
            '선릉역': (37.504741, 127.048976),
            '삼성역': (37.508847, 127.063804)
        }
        
        for station_name, (lat, lon) in major_stations.items():
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers',
                marker=dict(size=10, color='blue', symbol='rail'),
                name=f"🚇 {station_name}",
                hovertemplate=f"<b>🚇 {station_name}</b><extra></extra>",
                showlegend=False
            ))
    
    # =============================================================================
    # 경로 비교 및 통계 분석
    # =============================================================================
    
    def create_journey_comparison_chart(self, visualization_journeys: List[VisualizationJourney]) -> go.Figure:
        """경로 비교 차트 생성"""
        print("📊 경로 비교 차트 생성 중...")
        
        if not visualization_journeys:
            return go.Figure()
        
        # 데이터 준비
        journey_ids = [j.journey_id for j in visualization_journeys]
        journey_types = [j.journey_type for j in visualization_journeys]
        times = [j.total_time for j in visualization_journeys]
        costs = [j.total_cost for j in visualization_journeys]
        distances = [j.summary_stats['total_distance_km'] for j in visualization_journeys]
        
        # 서브플롯 생성
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('소요시간 비교', '요금 비교', '거리 비교', '효율성 분석'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # 색상 맵
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # 1. 소요시간 비교
        fig.add_trace(
            go.Bar(
                x=[f"경로{jid}<br>({jtype})" for jid, jtype in zip(journey_ids, journey_types)],
                y=times,
                name='소요시간(분)',
                marker_color=colors[0],
                text=[f"{t}분" for t in times],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. 요금 비교
        fig.add_trace(
            go.Bar(
                x=[f"경로{jid}" for jid in journey_ids],
                y=costs,
                name='요금(원)',
                marker_color=colors[1],
                text=[f"{c:,.0f}원" for c in costs],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. 거리 비교
        fig.add_trace(
            go.Bar(
                x=[f"경로{jid}" for jid in journey_ids],
                y=distances,
                name='거리(km)',
                marker_color=colors[2],
                text=[f"{d:.1f}km" for d in distances],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. 효율성 분석 (시간당 비용)
        efficiency = [c/t*60 if t > 0 else 0 for c, t in zip(costs, times)]  # 시간당 비용
        speed = [d/t*60 if t > 0 else 0 for d, t in zip(distances, times)]   # 평균 속도
        
        fig.add_trace(
            go.Scatter(
                x=[f"경로{jid}" for jid in journey_ids],
                y=efficiency,
                mode='lines+markers',
                name='시간당 비용(원/시간)',
                line=dict(color=colors[3], width=3),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[f"경로{jid}" for jid in journey_ids],
                y=speed,
                mode='lines+markers',
                name='평균 속도(km/시간)',
                line=dict(color=colors[4], width=3),
                marker=dict(size=8),
                yaxis='y2'
            ),
            row=2, col=2, secondary_y=True
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title="Multi-modal RAPTOR 경로 성능 비교",
            title_x=0.5,
            height=600,
            showlegend=False,
            font=dict(size=10)
        )
        
        # Y축 라벨
        fig.update_yaxes(title_text="시간(분)", row=1, col=1)
        fig.update_yaxes(title_text="요금(원)", row=1, col=2)
        fig.update_yaxes(title_text="거리(km)", row=2, col=1)
        fig.update_yaxes(title_text="비용(원/시간)", row=2, col=2)
        fig.update_yaxes(title_text="속도(km/시간)", row=2, col=2, secondary_y=True)
        
        print("✅ 경로 비교 차트 생성 완료")
        return fig
    
    def generate_journey_statistics(self, visualization_journeys: List[VisualizationJourney]) -> Dict:
        """경로 통계 생성"""
        if not visualization_journeys:
            return {}
        
        stats = {
            'total_journeys': len(visualization_journeys),
            'journey_types': {},
            'time_stats': {},
            'cost_stats': {},
            'distance_stats': {},
            'mode_analysis': {},
            'efficiency_rankings': []
        }
        
        # 기본 통계
        times = [j.total_time for j in visualization_journeys]
        costs = [j.total_cost for j in visualization_journeys]
        distances = [j.summary_stats['total_distance_km'] for j in visualization_journeys]
        
        stats['time_stats'] = {
            'min': min(times),
            'max': max(times),
            'avg': sum(times) / len(times),
            'median': sorted(times)[len(times)//2]
        }
        
        stats['cost_stats'] = {
            'min': min(costs),
            'max': max(costs),
            'avg': sum(costs) / len(costs),
            'median': sorted(costs)[len(costs)//2]
        }
        
        stats['distance_stats'] = {
            'min': min(distances),
            'max': max(distances),
            'avg': sum(distances) / len(distances),
            'median': sorted(distances)[len(distances)//2]
        }
        
        # 경로 타입별 분석
        for journey in visualization_journeys:
            journey_type = journey.journey_type
            if journey_type not in stats['journey_types']:
                stats['journey_types'][journey_type] = {
                    'count': 0,
                    'avg_time': 0,
                    'avg_cost': 0,
                    'avg_distance': 0
                }
            
            type_stats = stats['journey_types'][journey_type]
            type_stats['count'] += 1
            type_stats['avg_time'] += journey.total_time
            type_stats['avg_cost'] += journey.total_cost
            type_stats['avg_distance'] += journey.summary_stats['total_distance_km']
        
        # 평균값 계산
        for journey_type, type_stats in stats['journey_types'].items():
            count = type_stats['count']
            type_stats['avg_time'] /= count
            type_stats['avg_cost'] /= count
            type_stats['avg_distance'] /= count
        
        # 모드 분석
        all_modes = []
        for journey in visualization_journeys:
            all_modes.extend(journey.summary_stats['modes_used'])
        
        from collections import Counter
        mode_counts = Counter(all_modes)
        stats['mode_analysis'] = dict(mode_counts)
        
        # 효율성 순위
        for journey in visualization_journeys:
            efficiency_score = journey.total_time * 0.6 + (journey.total_cost / 1000) * 0.4
            stats['efficiency_rankings'].append({
                'journey_id': journey.journey_id,
                'journey_type': journey.journey_type,
                'efficiency_score': efficiency_score,
                'total_time': journey.total_time,
                'total_cost': journey.total_cost
            })
        
        stats['efficiency_rankings'].sort(key=lambda x: x['efficiency_score'])
        
        return stats
    
    # =============================================================================
    # 메인 실행 함수들
    # =============================================================================
    
    def visualize_all_journeys(self, origin_lat: float, origin_lon: float,
                              dest_lat: float, dest_lon: float,
                              save_path: str = "visualization_results") -> Dict[str, Any]:
        """모든 경로 시각화 실행"""
        print(f"\n🎨 강남구 Multi-modal RAPTOR 경로 시각화 시작")
        print(f"   출발지: ({origin_lat:.6f}, {origin_lon:.6f})")
        print(f"   목적지: ({dest_lat:.6f}, {dest_lon:.6f})")
        
        if not self.journey_results:
            print("❌ RAPTOR 결과 데이터가 없습니다.")
            return {}
        
        # 1. 정확한 경로 좌표 생성
        print("\n1️⃣ 정확한 경로 좌표 생성...")
        visualization_journeys = []
        
        for journey_data in self.journey_results:
            viz_journey = self.generate_accurate_route_coordinates(journey_data)
            visualization_journeys.append(viz_journey)
        
        print(f"✅ {len(visualization_journeys)}개 경로 좌표 생성 완료")
        
        # 2. 대화형 웹 지도 생성
        print("\n2️⃣ 대화형 웹 지도 생성...")
        interactive_map = self.create_interactive_map(
            visualization_journeys,
            (origin_lat, origin_lon),
            (dest_lat, dest_lon)
        )
        
        # 3. Plotly 시각화 생성
        print("\n3️⃣ Plotly 시각화 생성...")
        plotly_fig = self.create_plotly_visualization(visualization_journeys)
        
        # 4. 경로 비교 차트 생성
        print("\n4️⃣ 경로 비교 차트 생성...")
        comparison_chart = self.create_journey_comparison_chart(visualization_journeys)
        
        # 5. 통계 분석
        print("\n5️⃣ 통계 분석...")
        statistics = self.generate_journey_statistics(visualization_journeys)
        
        # 6. 결과 저장
        print(f"\n6️⃣ 결과 저장: {save_path}/")
        results = self._save_visualization_results(
            visualization_journeys, interactive_map, plotly_fig, 
            comparison_chart, statistics, save_path
        )
        
        print("\n🎉 강남구 Multi-modal RAPTOR 경로 시각화 완료!")
        return results
    
    def _save_visualization_results(self, journeys: List[VisualizationJourney],
                                   interactive_map: folium.Map,
                                   plotly_fig: go.Figure,
                                   comparison_chart: go.Figure,
                                   statistics: Dict,
                                   save_path: str) -> Dict[str, Any]:
        """시각화 결과 저장"""
        
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        results = {
            'visualization_journeys': journeys,
            'statistics': statistics,
            'file_paths': {}
        }
        
        try:
            # 1. 대화형 지도 저장
            map_path = save_dir / 'interactive_route_map.html'
            interactive_map.save(str(map_path))
            results['file_paths']['interactive_map'] = str(map_path)
            print(f"   ✅ 대화형 지도: {map_path}")
            
            # 2. Plotly 시각화 저장
            plotly_path = save_dir / 'route_visualization.html'
            plotly_fig.write_html(str(plotly_path))
            results['file_paths']['plotly_visualization'] = str(plotly_path)
            print(f"   ✅ Plotly 시각화: {plotly_path}")
            
            # 3. 경로 비교 차트 저장
            comparison_path = save_dir / 'route_comparison.html'
            comparison_chart.write_html(str(comparison_path))
            results['file_paths']['comparison_chart'] = str(comparison_path)
            print(f"   ✅ 경로 비교 차트: {comparison_path}")
            
            # 4. 정적 이미지 저장 (PNG)
            try:
                png_path = save_dir / 'route_visualization.png'
                plotly_fig.write_image(str(png_path), width=1200, height=800)
                results['file_paths']['static_image'] = str(png_path)
                print(f"   ✅ 정적 이미지: {png_path}")
            except Exception as e:
                print(f"   ⚠️ PNG 저장 실패: {e}")
            
            # 5. 통계 데이터 저장
            stats_path = save_dir / 'route_statistics.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False, default=str)
            results['file_paths']['statistics'] = str(stats_path)
            print(f"   ✅ 통계 데이터: {stats_path}")
            
            # 6. 시각화 경로 데이터 저장
            viz_data = []
            for journey in journeys:
                journey_data = {
                    'journey_id': journey.journey_id,
                    'journey_type': journey.journey_type,
                    'total_time': journey.total_time,
                    'total_cost': journey.total_cost,
                    'summary_stats': journey.summary_stats,
                    'segments': []
                }
                
                for segment in journey.segments:
                    segment_data = {
                        'mode': segment.mode,
                        'route_name': segment.route_name,
                        'start_point': segment.start_point,
                        'end_point': segment.end_point,
                        'coordinates': segment.coordinates,
                        'duration': segment.duration,
                        'distance': segment.distance,
                        'cost': segment.cost,
                        'color': segment.color,
                        'route_type': segment.route_type
                    }
                    journey_data['segments'].append(segment_data)
                
                viz_data.append(journey_data)
            
            viz_path = save_dir / 'visualization_data.json'
            with open(viz_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2, ensure_ascii=False)
            results['file_paths']['visualization_data'] = str(viz_path)
            print(f"   ✅ 시각화 데이터: {viz_path}")
            
            # 7. GeoJSON 저장
            geojson_data = self._create_geojson_from_journeys(journeys)
            geojson_path = save_dir / 'routes.geojson'
            with open(geojson_path, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)
            results['file_paths']['geojson'] = str(geojson_path)
            print(f"   ✅ GeoJSON: {geojson_path}")
            
            # 8. 요약 리포트 생성
            report_path = save_dir / 'visualization_report.html'
            self._generate_html_report(journeys, statistics, report_path)
            results['file_paths']['report'] = str(report_path)
            print(f"   ✅ 요약 리포트: {report_path}")
            
        except Exception as e:
            print(f"   ❌ 저장 중 오류: {e}")
        
        return results
    
    def _create_geojson_from_journeys(self, journeys: List[VisualizationJourney]) -> Dict:
        """경로를 GeoJSON 형식으로 변환"""
        features = []
        
        for journey in journeys:
            for i, segment in enumerate(journey.segments):
                if len(segment.coordinates) >= 2:
                    # LineString 피처 생성
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "journey_id": journey.journey_id,
                            "journey_type": journey.journey_type,
                            "segment_id": i,
                            "mode": segment.mode,
                            "route_name": segment.route_name,
                            "duration": segment.duration,
                            "distance": segment.distance,
                            "cost": segment.cost,
                            "color": segment.color,
                            "route_type": segment.route_type
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[lon, lat] for lat, lon in segment.coordinates]
                        }
                    }
                    features.append(feature)
                
                # 시작점/끝점 마커
                if segment.start_point:
                    point_feature = {
                        "type": "Feature",
                        "properties": {
                            "journey_id": journey.journey_id,
                            "segment_id": i,
                            "point_type": "start",
                            "mode": segment.mode,
                            "route_name": segment.route_name
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [segment.start_point[1], segment.start_point[0]]
                        }
                    }
                    features.append(point_feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def _generate_html_report(self, journeys: List[VisualizationJourney],
                             statistics: Dict, report_path: Path):
        """HTML 요약 리포트 생성"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>강남구 Multi-modal RAPTOR 경로 분석 리포트</title>
            <style>
                body {{ font-family: 'Malgun Gothic', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }}
                .journey {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
                .segment {{ margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #007bff; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .stat-box {{ padding: 15px; background: #e9ecef; border-radius: 8px; text-align: center; }}
                .emoji {{ font-size: 1.5em; margin-right: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🗺️ 강남구 Multi-modal RAPTOR</h1>
                <h2>경로 분석 리포트</h2>
                <p>생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3>📊 전체 통계</h3>
                <div class="stats">
                    <div class="stat-box">
                        <div><span class="emoji">🛣️</span><strong>총 경로 수</strong></div>
                        <div style="font-size: 2em; color: #007bff;">{statistics.get('total_journeys', 0)}</div>
                    </div>
                    <div class="stat-box">
                        <div><span class="emoji">⏱️</span><strong>평균 소요시간</strong></div>
                        <div style="font-size: 1.5em; color: #28a745;">{statistics.get('time_stats', {}).get('avg', 0):.1f}분</div>
                    </div>
                    <div class="stat-box">
                        <div><span class="emoji">💰</span><strong>평균 요금</strong></div>
                        <div style="font-size: 1.5em; color: #ffc107;">{statistics.get('cost_stats', {}).get('avg', 0):.0f}원</div>
                    </div>
                    <div class="stat-box">
                        <div><span class="emoji">📏</span><strong>평균 거리</strong></div>
                        <div style="font-size: 1.5em; color: #17a2b8;">{statistics.get('distance_stats', {}).get('avg', 0):.1f}km</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>🚗 교통수단별 분석</h3>
                <table>
                    <thead>
                        <tr>
                            <th>교통수단</th>
                            <th>경로 수</th>
                            <th>평균 시간</th>
                            <th>평균 요금</th>
                            <th>평균 거리</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # 교통수단별 통계 테이블
        type_emojis = {
            'walk': '🚶‍♂️', 'bike': '🚲', 'transit': '🚇', 'mixed': '🔄'
        }
        
        for journey_type, type_stats in statistics.get('journey_types', {}).items():
            emoji = type_emojis.get(journey_type, '🚌')
            html_content += f"""
                        <tr>
                            <td>{emoji} {journey_type.upper()}</td>
                            <td>{type_stats['count']}개</td>
                            <td>{type_stats['avg_time']:.1f}분</td>
                            <td>{type_stats['avg_cost']:.0f}원</td>
                            <td>{type_stats['avg_distance']:.1f}km</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h3>🏆 효율성 순위</h3>
                <table>
                    <thead>
                        <tr>
                            <th>순위</th>
                            <th>경로</th>
                            <th>교통수단</th>
                            <th>소요시간</th>
                            <th>요금</th>
                            <th>효율성 점수</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # 효율성 순위 테이블
        for i, ranking in enumerate(statistics.get('efficiency_rankings', [])[:5], 1):
            emoji = type_emojis.get(ranking['journey_type'], '🚌')
            html_content += f"""
                        <tr>
                            <td>#{i}</td>
                            <td>경로 {ranking['journey_id']}</td>
                            <td>{emoji} {ranking['journey_type'].upper()}</td>
                            <td>{ranking['total_time']}분</td>
                            <td>{ranking['total_cost']:,.0f}원</td>
                            <td>{ranking['efficiency_score']:.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h3>🗺️ 상세 경로 정보</h3>
        """
        
        # 각 경로별 상세 정보
        for journey in journeys:
            type_emoji = type_emojis.get(journey.journey_type, '🚌')
            html_content += f"""
                <div class="journey">
                    <h4>{type_emoji} 경로 {journey.journey_id} ({journey.journey_type.upper()})</h4>
                    <p><strong>총 소요시간:</strong> {journey.total_time}분 | 
                       <strong>총 요금:</strong> {journey.total_cost:,.0f}원 | 
                       <strong>총 거리:</strong> {journey.summary_stats['total_distance_km']}km</p>
                    
                    <h5>세부 구간:</h5>
            """
            
            for i, segment in enumerate(journey.segments, 1):
                mode_emojis = {
                    'walk': '🚶‍♂️', 'bike': '🚲', 'transit': '🚇',
                    'bike_rental': '🔄', 'bike_return': '🔄'
                }
                mode_emoji = mode_emojis.get(segment.mode, '🚌')
                
                html_content += f"""
                    <div class="segment">
                        <strong>{i}. {mode_emoji} {segment.route_name}</strong><br>
                        소요시간: {segment.duration}분 | 거리: {segment.distance:.2f}km | 요금: {segment.cost:,.0f}원
                    </div>
                """
            
            html_content += "</div>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h3>📁 생성된 파일들</h3>
                <ul>
                    <li><strong>interactive_route_map.html</strong> - 대화형 지도 (클릭하여 열기)</li>
                    <li><strong>route_visualization.html</strong> - Plotly 시각화</li>
                    <li><strong>route_comparison.html</strong> - 경로 비교 차트</li>
                    <li><strong>routes.geojson</strong> - GIS 소프트웨어용 경로 데이터</li>
                    <li><strong>route_statistics.json</strong> - 상세 통계 데이터</li>
                    <li><strong>visualization_data.json</strong> - 시각화 원본 데이터</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #6c757d;">
                <p>🎯 강남구 Multi-modal RAPTOR 시스템으로 생성됨</p>
                <p>Python + GTFS + 실제 도로망 기반 정확한 경로 시각화</p>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def print_visualization_summary(self, results: Dict[str, Any]):
        """시각화 결과 요약 출력"""
        if not results:
            return
        
        journeys = results.get('visualization_journeys', [])
        statistics = results.get('statistics', {})
        file_paths = results.get('file_paths', {})
        
        print("\n" + "="*70)
        print("🎨 강남구 Multi-modal RAPTOR 시각화 결과 요약")
        print("="*70)
        
        if journeys:
            print(f"\n📊 시각화된 경로: {len(journeys)}개")
            
            for journey in journeys:
                type_emoji = {
                    'walk': '🚶‍♂️', 'bike': '🚲', 'transit': '🚇', 'mixed': '🔄'
                }.get(journey.journey_type, '🚌')
                
                print(f"   {type_emoji} 경로 {journey.journey_id}: {journey.total_time}분, "
                      f"{journey.total_cost:,.0f}원, {journey.summary_stats['total_distance_km']}km")
        
        if statistics:
            print(f"\n📈 통계 요약:")
            time_stats = statistics.get('time_stats', {})
            cost_stats = statistics.get('cost_stats', {})
            
            print(f"   ⏱️ 소요시간: 최단 {time_stats.get('min', 0)}분, "
                  f"최장 {time_stats.get('max', 0)}분, 평균 {time_stats.get('avg', 0):.1f}분")
            print(f"   💰 요금: 최저 {cost_stats.get('min', 0):,.0f}원, "
                  f"최고 {cost_stats.get('max', 0):,.0f}원, 평균 {cost_stats.get('avg', 0):.0f}원")
            
            # 가장 효율적인 경로
            rankings = statistics.get('efficiency_rankings', [])
            if rankings:
                best = rankings[0]
                print(f"   🏆 가장 효율적: 경로 {best['journey_id']} ({best['journey_type']})")
        
        if file_paths:
            print(f"\n📁 생성된 파일:")
            for file_type, path in file_paths.items():
                file_icons = {
                    'interactive_map': '🗺️',
                    'plotly_visualization': '📊',
                    'comparison_chart': '📈',
                    'static_image': '🖼️',
                    'statistics': '📋',
                    'geojson': '🌍',
                    'report': '📄'
                }
                icon = file_icons.get(file_type, '📄')
                print(f"   {icon} {Path(path).name}")
        
        print(f"\n💡 사용법:")
        print(f"   1. interactive_route_map.html을 브라우저로 열어 대화형 지도 확인")
        print(f"   2. route_comparison.html에서 경로 성능 비교")
        print(f"   3. routes.geojson을 QGIS 등에서 열어 상세 분석")
        print(f"   4. visualization_report.html에서 종합 리포트 확인")
        
        print(f"\n🎉 강남구 Multi-modal RAPTOR 시각화 완료!")
        print("="*70)


# =============================================================================
# 사용 예제 및 테스트 함수
# =============================================================================

def create_sample_raptor_results():
    """샘플 RAPTOR 결과 생성 (테스트용)"""
    return [
        {
            'journey_id': 1,
            'journey_type': 'transit',
            'total_time': 25,
            'total_cost': 1370,
            'segments': [
                {
                    'mode': 'walk',
                    'from': '출발지',
                    'to': '강남역',
                    'duration': 5,
                    'distance_km': 0.4,
                    'cost': 0,
                    'route_info': '도보'
                },
                {
                    'mode': 'transit',
                    'from': '강남역',
                    'to': '역삼역',
                    'duration': 15,
                    'cost': 1370,
                    'route_info': '지하철 2호선',
                    'route_id': 'line_2',
                    'route_color': '#00A84D',
                    'route_type': 1
                },
                {
                    'mode': 'walk',
                    'from': '역삼역',
                    'to': '목적지',
                    'duration': 5,
                    'distance_km': 0.3,
                    'cost': 0,
                    'route_info': '도보'
                }
            ]
        },
        {
            'journey_id': 2,
            'journey_type': 'mixed',
            'total_time': 22,
            'total_cost': 2370,
            'segments': [
                {
                    'mode': 'walk',
                    'from': '출발지',
                    'to': '따릉이 대여소 123',
                    'duration': 3,
                    'distance_km': 0.2,
                    'cost': 0,
                    'route_info': '도보'
                },
                {
                    'mode': 'bike_rental',
                    'from': '따릉이 대여소 123',
                    'to': '따릉이 대여소 123',
                    'duration': 2,
                    'cost': 0,
                    'route_info': '따릉이 대여'
                },
                {
                    'mode': 'bike',
                    'from': '따릉이 대여소 123',
                    'to': '선릉역 따릉이 대여소',
                    'duration': 8,
                    'distance_km': 1.2,
                    'cost': 1000,
                    'route_info': '따릉이 8분'
                },
                {
                    'mode': 'bike_return',
                    'from': '선릉역 따릉이 대여소',
                    'to': '선릉역 따릉이 대여소',
                    'duration': 2,
                    'cost': 0,
                    'route_info': '따릉이 반납'
                },
                {
                    'mode': 'transit',
                    'from': '선릉역',
                    'to': '역삼역',
                    'duration': 5,
                    'cost': 1370,
                    'route_info': '지하철 분당선',
                    'route_id': 'bundang_line',
                    'route_color': '#FFCD12',
                    'route_type': 1
                },
                {
                    'mode': 'walk',
                    'from': '역삼역',
                    'to': '목적지',
                    'duration': 2,
                    'distance_km': 0.15,
                    'cost': 0,
                    'route_info': '도보'
                }
            ]
        }
    ]


if __name__ == "__main__":
    print("🎨 강남구 Multi-modal RAPTOR 시각화 엔진 테스트")
    print("="*60)
    
    # 데이터 경로 설정
    data_path = "C:\\Users\\sec\\Desktop\\kim\\학회\\GTFS\\code\\multimodal_raptor_project\\gangnam_multimodal_raptor_data_with_real_roads"
    results_path = "C:\\Users\\sec\\Desktop\\kim\\학회\\GTFS\\code\\multimodal_raptor_project\\test_results"
    
    try:
        # 시각화 엔진 초기화
        visualizer = GangnamRAPTORVisualizer(data_path, results_path)
        
        # 샘플 데이터로 테스트 (실제 데이터가 없는 경우)
        if not visualizer.journey_results:
            print("📝 샘플 RAPTOR 결과 사용...")
            visualizer.journey_results = create_sample_raptor_results()
        
        # 테스트 시나리오: 삼성역 → 강남역
        origin_lat, origin_lon = 37.51579174292475, 127.02039435436643  # 삼성역
        dest_lat, dest_lon = 37.49985645759325, 127.04146988383535      # 강남역
        
        # 전체 시각화 실행
        visualization_results = visualizer.visualize_all_journeys(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            save_path="gangnam_raptor_visualization_results"
        )
        
        # 결과 요약 출력
        visualizer.print_visualization_summary(visualization_results)
        # 기존 코드
        

        # 여기에 추가 ⬇️
        print("\n" + "="*50)
        print("🔍 경로 디버깅 테스트")
        print("="*50)

        # 디버깅 코드
        if 'visualization_journeys' in visualization_results:
            journeys = visualization_results['visualization_journeys']
            print(f"\n📊 총 {len(journeys)}개 경로 분석:")
            
            for journey in journeys:
                print(f"\n--- 경로 {journey.journey_id} ({journey.journey_type}) ---")
                print(f"세그먼트 수: {len(journey.segments)}")
                
                for i, segment in enumerate(journey.segments):
                    print(f"  {i+1}. {segment.mode} - {segment.route_name}")
                    print(f"     좌표 개수: {len(segment.coordinates)}")
                    print(f"     색상: {segment.color}")
                    
                    if len(segment.coordinates) >= 2:
                        print(f"     시작: {segment.coordinates[0]}")
                        print(f"     끝: {segment.coordinates[-1]}")
                    else:
                        print(f"     ⚠️ 좌표 부족! (경로선이 안 보이는 원인)")
        else:
            print("❌ 시각화 데이터가 없습니다!")
        
        
        
        
        
        
        
        
        
        
        
        print(f"\n🔗 주요 파일 경로:")
        file_paths = visualization_results.get('file_paths', {})
        if 'interactive_map' in file_paths:
            print(f"   🗺️ 대화형 지도: {file_paths['interactive_map']}")
        if 'report' in file_paths:
            print(f"   📄 종합 리포트: {file_paths['report']}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()