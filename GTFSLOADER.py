"""
KTDB GTFS 표준 기반 완전한 교통 데이터 로더
- GTFS 개별 파일 (agency, stops, routes, trips, stop_times, calendar)
- 도로망 SHP 파일 
- 데이터 통합 및 분석
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import warnings
from typing import Dict, Tuple, Optional, List
import os
import time
from datetime import datetime


warnings.filterwarnings('ignore')

class KTDBGTFSLoader:
    """KTDB GTFS 표준 기반 교통 데이터 로더"""
    
    def __init__(self, road_data_path: str, gtfs_data_path: str):
        self.road_data_path = Path(road_data_path)
        self.gtfs_data_path = Path(gtfs_data_path)
        
        # 도로망 데이터
        self.road_nodes = None
        self.road_links = None
        
        # GTFS 핵심 데이터 (KTDB 표준)
        self.agency = None          # 대중교통 기관 정보
        self.stops = None           # 정류장/역 정보  
        self.routes = None          # 노선 정보
        self.trips = None           # 운행회차 정보
        self.stop_times = None      # 정차 시간표
        self.calendar = None        # 운행 일정
        
        # 선택적 GTFS 데이터
        self.calendar_dates = None  # 예외 운행일
        self.fare_attributes = None # 요금 정보
        self.fare_rules = None      # 요금 규칙
        self.shapes = None          # 노선 형상
        self.transfers = None       # 환승 정보
        
        # 통합 분석 데이터
        self.integrated_stops = None
        self.route_analysis = None
        self.accessibility_matrix = None
        
        print("🚀 KTDB GTFS 표준 교통 데이터 로더 초기화")
        self._validate_paths()
    
    def _validate_paths(self):
        """데이터 경로 검증"""
        print(f"📂 도로망 경로: {self.road_data_path}")
        print(f"📂 GTFS 경로: {self.gtfs_data_path}")
        
        if not self.road_data_path.exists():
            print(f"⚠️ 도로망 데이터 경로 없음")
            
        if not self.gtfs_data_path.exists():
            print(f"⚠️ GTFS 데이터 경로 없음")
    
    # ========== 1. GTFS 핵심 데이터 로딩 (KTDB 표준) ==========
    def load_gtfs_data(self) -> Dict[str, bool]:
        """KTDB GTFS 표준에 따른 데이터 로딩"""
        print("\n🚇 1단계: KTDB GTFS 데이터 로딩...")
        
        results = {}
        
        # GTFS 필수 파일들 (KTDB 구축 여부 Y)
        required_files = {
            'agency': 'agency.txt',
            'stops': 'stops.txt', 
            'routes': 'routes.txt',
            'trips': 'trips.txt',
            'stop_times': 'stop_times.txt',
            'calendar': 'calendar.txt'
        }
        
        # GTFS 선택적 파일들 (KTDB 구축 여부 N)
        optional_files = {
            'calendar_dates': 'calendar_dates.txt',
            'fare_attributes': 'fare_attributes.txt',
            'fare_rules': 'fare_rules.txt',
            'shapes': 'shapes.txt',
            'transfers': 'transfers.txt'
        }
        
        # 필수 파일 로딩
        print("   📋 필수 GTFS 파일 로딩:")
        for name, filename in required_files.items():
            results[name] = self._load_gtfs_file(name, filename, required=True)
        
        # 선택적 파일 로딩
        print("\n   📋 선택적 GTFS 파일 로딩:")
        for name, filename in optional_files.items():
            results[name] = self._load_gtfs_file(name, filename, required=False)
        
        # 로딩 결과 요약
        self._print_gtfs_summary(results)
        
        return results
    
    def _load_gtfs_file(self, data_name: str, filename: str, required: bool = True) -> bool:
        """개별 GTFS 파일 로딩"""
        file_path = self.gtfs_data_path / filename
        
        try:
            if file_path.exists():
                # CSV 로딩 (다양한 인코딩 시도)
                encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        setattr(self, data_name, df)
                        
                        # 데이터 품질 확인
                        if len(df) > 0:
                            status = "✅" if required else "📄"
                            print(f"     {status} {filename}: {len(df):,}개 ({encoding})")
                            return True
                        else:
                            print(f"     ⚠️ {filename}: 빈 파일")
                            return False
                            
                    except UnicodeDecodeError:
                        continue
                        
                print(f"     ❌ {filename}: 인코딩 실패")
                return False
                
            else:
                if required:
                    print(f"     ❌ {filename}: 필수 파일 없음")
                else:
                    print(f"     ➖ {filename}: 선택 파일 없음")
                return False
                
        except Exception as e:
            print(f"     ❌ {filename}: 로딩 실패 - {str(e)[:50]}...")
            return False
    
    def _print_gtfs_summary(self, results: Dict[str, bool]):
        """GTFS 로딩 결과 요약"""
        print("\n   📊 GTFS 데이터 로딩 요약:")
        
        # 필수 파일 체크
        required = ['agency', 'stops', 'routes', 'trips', 'stop_times', 'calendar']
        required_loaded = sum(1 for key in required if results.get(key, False))
        print(f"     필수 파일: {required_loaded}/{len(required)} 개")
        
        # 선택적 파일 체크  
        optional = ['calendar_dates', 'fare_attributes', 'fare_rules', 'shapes', 'transfers']
        optional_loaded = sum(1 for key in optional if results.get(key, False))
        print(f"     선택 파일: {optional_loaded}/{len(optional)} 개")
        
        # 데이터 규모 (로딩된 것만)
        if self.stops is not None:
            print(f"     정류장/역: {len(self.stops):,}개")
        if self.routes is not None:
            print(f"     노선: {len(self.routes):,}개")
        if self.trips is not None:
            print(f"     운행회차: {len(self.trips):,}개")
        if self.stop_times is not None:
            print(f"     정차시간: {len(self.stop_times):,}개")
    
    # ========== 2. 도로망 데이터 로딩 ==========
    def load_road_network(self) -> bool:
        """도로망 데이터 로딩 (기존 성공 방법 사용)"""
        print("\n🛣️ 2단계: 도로망 데이터 로딩...")
        
        try:
            # 노드 파일 로딩
            node_patterns = ['ad0102*.shp', '*node*.shp', '*교차*.shp']
            node_file = self._find_file_by_patterns(self.road_data_path, node_patterns)
            
            if node_file:
                self.road_nodes = gpd.read_file(node_file, encoding='cp949')
                print(f"   ✅ 노드: {len(self.road_nodes):,}개")
            
            # 링크 파일 로딩 (기존 성공 방법)
            link_patterns = ['ad0022*.shp', '*link*.shp', '*도로*.shp']
            link_file = self._find_file_by_patterns(self.road_data_path, link_patterns)
            
            if link_file:
                self.road_links = gpd.read_file(str(link_file))
                print(f"   ✅ 링크: {len(self.road_links):,}개")
                
                # 도로 등급별 통계
                if 'ROAD_RANK' in self.road_links.columns:
                    self._print_road_statistics()
                
                # 총 도로 연장
                if 'LENGTH' in self.road_links.columns:
                    total_length = self.road_links['LENGTH'].sum()
                    print(f"   📏 총 도로연장: {total_length:,.1f} km")
            
            return self.road_nodes is not None and self.road_links is not None
            
        except Exception as e:
            print(f"❌ 도로망 로딩 실패: {e}")
            return False
    
    def _print_road_statistics(self):
        """도로 등급별 통계 출력"""
        road_ranks = self.road_links['ROAD_RANK'].value_counts()
        print(f"   📊 도로등급별 통계:")
        
        # KTDB 도로등급 코드
        rank_names = {
            '101': '고속도로', '102': '도시고속도로', '103': '일반국도',
            '104': '특별광역시도', '105': '국가지원지방도', 
            '106': '지방도', '107': '시군도'
        }
        
        for code, count in road_ranks.head(5).items():
            name = rank_names.get(str(code), '기타')
            print(f"     {code}({name}): {count:,}개")
    
    # ========== 3. GTFS 데이터 분석 ==========
    def analyze_gtfs_data(self):
        """GTFS 데이터 상세 분석"""
        print("\n📊 3단계: GTFS 데이터 분석...")
        
        if not self._validate_gtfs_loaded():
            return
        
        # 기관 정보 분석
        self._analyze_agency()
        
        # 정류장 분석
        self._analyze_stops()
        
        # 노선 분석  
        self._analyze_routes()
        
        # 운행 분석
        self._analyze_trips()
        
        # 시간표 분석
        self._analyze_stop_times()
    
    def _analyze_agency(self):
        """기관 정보 분석"""
        if self.agency is not None:
            print("\n   🏢 기관 정보:")
            for _, agency in self.agency.iterrows():
                print(f"     ID: {agency.get('agency_id', 'N/A')}")
                print(f"     이름: {agency.get('agency_name', 'N/A')}")
                print(f"     URL: {agency.get('agency_url', 'N/A')}")
                print(f"     시간대: {agency.get('agency_timezone', 'N/A')}")
    
    def _analyze_stops(self):
        """정류장 분석"""
        if self.stops is not None:
            print(f"\n   🚏 정류장 분석:")
            print(f"     총 정류장: {len(self.stops):,}개")
            
            # 좌표 정보 확인
            has_coords = self.stops[['stop_lat', 'stop_lon']].notna().all(axis=1).sum()
            print(f"     좌표 있음: {has_coords:,}개 ({has_coords/len(self.stops)*100:.1f}%)")
            
            # 좌표 범위 
            if has_coords > 0:
                coords_df = self.stops[['stop_lat', 'stop_lon']].dropna()
                print(f"     위도 범위: {coords_df['stop_lat'].min():.4f} ~ {coords_df['stop_lat'].max():.4f}")
                print(f"     경도 범위: {coords_df['stop_lon'].min():.4f} ~ {coords_df['stop_lon'].max():.4f}")
    
    def _analyze_routes(self):
        """노선 분석"""
        if self.routes is not None:
            print(f"\n   🚌 노선 분석:")
            print(f"     총 노선: {len(self.routes):,}개")
            
            # 노선 유형별 분석 (KTDB 기준)
            if 'route_type' in self.routes.columns:
                route_types = self.routes['route_type'].value_counts()
                print(f"     노선 유형별:")
                
                # KTDB route_type 매핑
                type_names = {
                    0: '시내/농어촌/마을버스',
                    1: '도시철도/경전철', 
                    2: '해운',
                    3: '시외버스',
                    4: '일반철도',
                    5: '공항리무진버스',
                    6: '고속철도',
                    7: '항공'
                }
                
                for route_type, count in route_types.head(8).items():
                    name = type_names.get(route_type, f'기타({route_type})')
                    print(f"       {route_type}: {name} - {count:,}개")
    
    def _analyze_trips(self):
        """운행회차 분석"""
        if self.trips is not None:
            print(f"\n   🚌 운행 분석:")
            print(f"     총 운행회차: {len(self.trips):,}개")
            
            # 노선별 운행회차
            if 'route_id' in self.trips.columns:
                trips_per_route = self.trips.groupby('route_id').size()
                print(f"     노선당 평균 운행: {trips_per_route.mean():.1f}회")
                print(f"     최대 운행 노선: {trips_per_route.max()}회")
    
    def _analyze_stop_times(self):
        """정차시간 분석"""
        if self.stop_times is not None:
            print(f"\n   ⏰ 정차시간 분석:")
            print(f"     총 정차 기록: {len(self.stop_times):,}개")
            
            # 시간 형식 확인
            if 'arrival_time' in self.stop_times.columns:
                sample_times = self.stop_times['arrival_time'].dropna().head(5)
                print(f"     시간 형식 예시: {list(sample_times)}")
    
    # ========== 4. 데이터 통합 ==========
    def integrate_transport_data(self) -> bool:
        """교통 데이터 통합"""
        print("\n🔗 4단계: 교통 데이터 통합...")
        
        if not self._validate_gtfs_loaded():
            print("   ⚠️ GTFS 데이터가 없어 통합 불가")
            return False
        
        try:
            # GTFS 정류장을 지리공간 데이터로 변환
            self._create_integrated_stops()
            
            # 노선-정류장 연결 분석
            self._analyze_route_stops()
            
            # 도로망과 연결 (도로망 데이터가 있는 경우)
            if self.road_links is not None:
                self._link_stops_to_roads()
            
            print("   ✅ 데이터 통합 완료")
            return True
            
        except Exception as e:
            print(f"❌ 데이터 통합 실패: {e}")
            return False
    
    def _create_integrated_stops(self):
        """GTFS 정류장을 지리공간 데이터로 변환"""
        if self.stops is None:
            return
        
        # 좌표가 있는 정류장만 선택
        valid_stops = self.stops.dropna(subset=['stop_lat', 'stop_lon']).copy()
        
        if len(valid_stops) > 0:
            # GeoDataFrame 생성
            self.integrated_stops = gpd.GeoDataFrame(
                valid_stops,
                geometry=gpd.points_from_xy(valid_stops.stop_lon, valid_stops.stop_lat),
                crs='EPSG:4326'
            )
            print(f"   📍 지리공간 정류장: {len(self.integrated_stops):,}개")
        else:
            print(f"   ⚠️ 좌표 정보가 있는 정류장이 없음")
    
    def _analyze_route_stops(self):
        """노선-정류장 연결 분석"""
        if self.stop_times is None or self.routes is None:
            return
        
        print("   🔍 노선-정류장 연결 분석...")
        
        # stop_times에서 노선별 정류장 추출
        if 'trip_id' in self.stop_times.columns and 'route_id' in self.trips.columns:
            # trip_id로 route_id 연결
            route_stops = self.stop_times.merge(
                self.trips[['trip_id', 'route_id']], 
                on='trip_id', 
                how='left'
            )
            
            # 노선별 정류장 수 계산
            stops_per_route = route_stops.groupby('route_id')['stop_id'].nunique()
            
            print(f"     노선당 평균 정류장: {stops_per_route.mean():.1f}개")
            print(f"     최대 정류장 노선: {stops_per_route.max()}개")
            
            # 정류장별 노선 수 
            routes_per_stop = route_stops.groupby('stop_id')['route_id'].nunique()
            print(f"     정류장당 평균 노선: {routes_per_stop.mean():.1f}개")
            
            self.route_analysis = {
                'stops_per_route': stops_per_route,
                'routes_per_stop': routes_per_stop
            }
    
    def _link_stops_to_roads(self):
        """정류장과 도로망 연결"""
        if self.integrated_stops is None or self.road_links is None:
            return
        
        print("   🔗 정류장-도로 연결 분석...")
        
        # 좌표계 통일
        if self.integrated_stops.crs != self.road_links.crs:
            stops_projected = self.integrated_stops.to_crs(self.road_links.crs)
        else:
            stops_projected = self.integrated_stops
        
        # 샘플링 (성능 고려)
        sample_size = min(1000, len(stops_projected))
        sample_stops = stops_projected.sample(sample_size, random_state=42)
        
        # 50m 버퍼로 도로 연결 찾기
        buffered_stops = sample_stops.copy()
        buffered_stops.geometry = buffered_stops.geometry.buffer(50)
        
        try:
            # 공간 조인
            stop_road_links = gpd.sjoin(
                buffered_stops,
                self.road_links,
                how='left',
                predicate='intersects'
            )
            
            connected_stops = len(stop_road_links.dropna(subset=['index_right']))
            print(f"     도로 연결 정류장: {connected_stops}/{sample_size}개 ({connected_stops/sample_size*100:.1f}%)")
            
        except Exception as e:
            print(f"     ⚠️ 공간 조인 실패: {str(e)[:50]}...")
    
    # ========== 5. 지역별 필터링 ==========
    def filter_by_region(self, region_name: str = "강남구") -> Dict:
        """지역별 데이터 필터링"""
        print(f"\n🎯 5단계: {region_name} 지역 데이터 추출...")
        
        # 강남구 대략적 경계 (더 넓은 범위로 설정)
        if region_name == "강남구":
            bounds = {
                'min_lon': 126.95, 'max_lon': 127.15,
                'min_lat': 37.45, 'max_lat': 37.57
            }
        else:
            # 서울 전체 범위
            bounds = {
                'min_lon': 126.7, 'max_lon': 127.3,
                'min_lat': 37.4, 'max_lat': 37.7
            }
        
        region_data = {}
        
        try:
            # GTFS 정류장 필터링
            if self.integrated_stops is not None:
                region_stops = self.integrated_stops[
                    (self.integrated_stops.geometry.x >= bounds['min_lon']) &
                    (self.integrated_stops.geometry.x <= bounds['max_lon']) &
                    (self.integrated_stops.geometry.y >= bounds['min_lat']) &
                    (self.integrated_stops.geometry.y <= bounds['max_lat'])
                ]
                region_data['stops'] = region_stops
                print(f"   🚏 {region_name} 정류장: {len(region_stops):,}개")
                
                # 해당 지역 노선 추출
                if len(region_stops) > 0 and self.stop_times is not None:
                    region_stop_ids = region_stops['stop_id'].tolist()
                    region_stop_times = self.stop_times[
                        self.stop_times['stop_id'].isin(region_stop_ids)
                    ]
                    
                    if self.trips is not None:
                        region_trip_ids = region_stop_times['trip_id'].unique()
                        region_trips = self.trips[
                            self.trips['trip_id'].isin(region_trip_ids)
                        ]
                        
                        region_route_ids = region_trips['route_id'].unique()
                        if self.routes is not None:
                            region_routes = self.routes[
                                self.routes['route_id'].isin(region_route_ids)
                            ]
                            region_data['routes'] = region_routes
                            print(f"   🚌 {region_name} 노선: {len(region_routes):,}개")
            
            # 도로망 필터링
            if self.road_links is not None:
                # 좌표계 변환
                road_links_4326 = self.road_links.to_crs('EPSG:4326')
                
                # 경계 박스와 교차하는 도로
                from shapely.geometry import box
                bbox = gpd.box(bounds['min_lon'], bounds['min_lat'], 
                              bounds['max_lon'], bounds['max_lat'])
                bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox], crs='EPSG:4326')
                
                region_roads = gpd.overlay(road_links_4326, bbox_gdf, how='intersection')
                region_data['roads'] = region_roads
                
                print(f"   🛣️ {region_name} 도로: {len(region_roads):,}개")
                
                if 'LENGTH' in region_roads.columns and len(region_roads) > 0:
                    region_road_length = region_roads['LENGTH'].sum()
                    print(f"   📏 {region_name} 도로연장: {region_road_length:.1f} km")
            
            return region_data
            
        except Exception as e:
            print(f"❌ 지역 필터링 실패: {e}")
            return {}
    
    # ========== 6. 결과 저장 및 요약 ==========
    def get_comprehensive_summary(self) -> Dict:
        """종합 데이터 요약"""
        summary = {
            'gtfs_data': {
                'agency': len(self.agency) if self.agency is not None else 0,
                'stops': len(self.stops) if self.stops is not None else 0,
                'routes': len(self.routes) if self.routes is not None else 0,
                'trips': len(self.trips) if self.trips is not None else 0,
                'stop_times': len(self.stop_times) if self.stop_times is not None else 0,
                'calendar': len(self.calendar) if self.calendar is not None else 0
            },
            'road_network': {
                'nodes': len(self.road_nodes) if self.road_nodes is not None else 0,
                'links': len(self.road_links) if self.road_links is not None else 0,
                'total_length_km': self.road_links['LENGTH'].sum() if self.road_links is not None and 'LENGTH' in self.road_links.columns else 0
            },
            'integration': {
                'integrated_stops': len(self.integrated_stops) if self.integrated_stops is not None else 0,
                'has_route_analysis': self.route_analysis is not None
            }
        }
        return summary
    
    def save_data(self, output_dir: str):
        """데이터 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 데이터 저장: {output_dir}/")
        
        try:
            # GTFS 데이터 저장
            gtfs_files = ['agency', 'stops', 'routes', 'trips', 'stop_times', 'calendar']
            for file_name in gtfs_files:
                data = getattr(self, file_name)
                if data is not None:
                    data.to_csv(output_path / f"{file_name}.csv", index=False, encoding='utf-8')
                    print(f"   ✅ {file_name}.csv")
            
            # 통합 정류장 저장 (지리공간 데이터)
            if self.integrated_stops is not None:
                self.integrated_stops.to_file(output_path / "integrated_stops.shp", encoding='utf-8')
                print(f"   ✅ integrated_stops.shp")
            
            # 요약 정보 저장
            summary = self.get_comprehensive_summary()
            import json
            with open(output_path / "summary.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"   ✅ summary.json")
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
    
    # ========== 유틸리티 함수들 ==========
    def _find_file_by_patterns(self, directory: Path, patterns: List[str]) -> Optional[Path]:
        """패턴으로 파일 찾기"""
        for pattern in patterns:
            files = list(directory.glob(pattern))
            if files:
                return files[0]
        return None
    
    def _validate_gtfs_loaded(self) -> bool:
        """GTFS 핵심 데이터 로딩 확인"""
        required = [self.stops, self.routes, self.trips, self.stop_times]
        loaded_count = sum(1 for data in required if data is not None)
        return loaded_count >= 2  # 최소 2개 이상의 핵심 데이터 필요


# ========== 메인 실행 코드 ==========
if __name__ == "__main__":
    print("🚀 KTDB GTFS 표준 교통 데이터 로더 시작")
    print("=" * 70)
    
    # 데이터 경로 설정 - 사용자 환경에 맞게 수정
    road_data_path = "C:\\Users\\sec\\Desktop\\kim\\학회\\GTFS\\code\\road_data"
    gtfs_data_path = "C:\\Users\\sec\\Desktop\\kim\\학회\\GTFS\\code\\202303_GTFS_DataSet"
    
    try:
        # 로더 생성
        loader = KTDBGTFSLoader(road_data_path, gtfs_data_path)
        
        # 1단계: GTFS 데이터 로딩
        gtfs_results = loader.load_gtfs_data()
        
        # 2단계: 도로망 데이터 로딩  
        road_success = loader.load_road_network()
        
        # 3단계: GTFS 데이터 분석
        loader.analyze_gtfs_data()
        
        # 4단계: 데이터 통합
        integration_success = loader.integrate_transport_data()
        
        # 5단계: 강남구 지역 데이터 추출
        gangnam_data = loader.filter_by_region("강남구")
        
        # 6단계: 결과 요약 및 저장
        print("\n" + "=" * 70)
        print("📊 최종 결과 요약")
        print("=" * 70)
        
        summary = loader.get_comprehensive_summary()
        
        # GTFS 데이터 요약
        print(f"\n🚇 GTFS 데이터:")
        gtfs_summary = summary['gtfs_data']
        print(f"   기관(agency): {gtfs_summary['agency']:,}개")
        print(f"   정류장(stops): {gtfs_summary['stops']:,}개")
        print(f"   노선(routes): {gtfs_summary['routes']:,}개") 
        print(f"   운행(trips): {gtfs_summary['trips']:,}개")
        print(f"   정차시간(stop_times): {gtfs_summary['stop_times']:,}개")
        print(f"   달력(calendar): {gtfs_summary['calendar']:,}개")
        
        # 도로망 데이터 요약
        print(f"\n🛣️ 도로망 데이터:")
        road_summary = summary['road_network']
        print(f"   노드: {road_summary['nodes']:,}개")
        print(f"   링크: {road_summary['links']:,}개")
        print(f"   총 연장: {road_summary['total_length_km']:,.1f} km")
        
        # 통합 데이터 요약
        print(f"\n🔗 통합 데이터:")
        integration_summary = summary['integration']
        print(f"   지리공간 정류장: {integration_summary['integrated_stops']:,}개")
        print(f"   노선 분석: {'✅' if integration_summary['has_route_analysis'] else '❌'}")
        
        # 강남구 데이터 요약
        if gangnam_data:
            print(f"\n🎯 강남구 데이터:")
            if 'stops' in gangnam_data:
                print(f"   정류장: {len(gangnam_data['stops']):,}개")
            if 'routes' in gangnam_data:
                print(f"   노선: {len(gangnam_data['routes']):,}개")
            if 'roads' in gangnam_data:
                print(f"   도로: {len(gangnam_data['roads']):,}개")
        
        # 데이터 품질 평가
        print(f"\n🎯 데이터 품질 평가:")
        gtfs_quality = "우수" if gtfs_summary['stops'] > 1000 and gtfs_summary['routes'] > 10 else "보통"
        road_quality = "우수" if road_summary['links'] > 10000 else "보통"
        integration_quality = "성공" if integration_summary['integrated_stops'] > 100 else "제한적"
        
        print(f"   GTFS 데이터: {gtfs_quality}")
        print(f"   도로망 데이터: {road_quality}")
        print(f"   데이터 통합: {integration_quality}")
        
        # 데이터 저장
        output_dir = "output_integrated_transport_data"
        loader.save_data(output_dir)
        
        # Multi-modal RAPTOR 준비도 평가
        print(f"\n🚀 Multi-modal RAPTOR 준비도:")
        
        raptor_ready = False
        if gtfs_summary['stops'] > 0 and gtfs_summary['routes'] > 0 and gtfs_summary['stop_times'] > 0:
            print(f"   ✅ GTFS 기반 대중교통 라우팅 가능")
            raptor_ready = True
        else:
            print(f"   ❌ GTFS 데이터 부족")
        
        if road_summary['links'] > 0:
            print(f"   ✅ 도로망 기반 경로 탐색 가능")
        else:
            print(f"   ❌ 도로망 데이터 없음")
        
        if integration_summary['integrated_stops'] > 0:
            print(f"   ✅ 지리공간 분석 가능")
        else:
            print(f"   ❌ 좌표 정보 부족")
        
        if raptor_ready:
            print(f"\n🎉 Python Multi-modal RAPTOR 구현 준비 완료!")
            print(f"   📍 연구 지역: 강남구")
            print(f"   🚇 대중교통: GTFS 표준 데이터")
            print(f"   🛣️ 도로망: 전국 표준 데이터")
            print(f"   💾 통합 데이터: {output_dir}/ 저장됨")
            
            
        else:
            print(f"\n⚠️ 일부 데이터 부족으로 제한적 분석만 가능")
        
        print(f"\n" + "=" * 70)
        print("🎯 KTDB GTFS 표준 교통 데이터 로더 완료!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print(f"\n❌ 사용자에 의한 중단")
    except Exception as e:
        print(f"\n❌ 실행 오류: {e}")
        import traceback
        traceback.print_exc()