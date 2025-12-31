"""
CS:GO Oracle V13 - PyQt5 GUI Application
Profesyonel tahmin arayüzü
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import warnings
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QFrame, QLineEdit,
    QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None

# =============================================================================
# RENK PALETİ
# =============================================================================
COLORS = {
    'primary_yellow': '#F5A623',      # Tok sarı (turuncu)
    'dark_navy': '#1A2744',           # Koyu lacivert
    'light_navy': '#2D3E5C',          # Açık lacivert
    'white': '#FFFFFF',               # Beyaz
    'hover_yellow': '#FFB740',        # Hover sarı
    'success': '#27AE60',             # Başarı yeşili
    'warning': '#F39C12',             # Uyarı turuncu
    'danger': '#E74C3C',              # Tehlike kırmızı
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['dark_navy']};
}}

QWidget {{
    background-color: {COLORS['dark_navy']};
    color: {COLORS['white']};
    font-family: 'Segoe UI', Arial, sans-serif;
}}

QLabel {{
    color: {COLORS['white']};
}}

QLabel#title {{
    font-size: 24px;
    font-weight: bold;
    color: {COLORS['primary_yellow']};
    padding: 10px;
}}

QLabel#subtitle {{
    font-size: 12px;
    color: {COLORS['white']};
    opacity: 0.8;
}}

QLabel#vs_label {{
    font-size: 36px;
    font-weight: bold;
    color: {COLORS['primary_yellow']};
}}

QLabel#section_label {{
    font-size: 14px;
    font-weight: bold;
    color: {COLORS['white']};
    margin-bottom: 5px;
}}

QLabel#result_winner {{
    font-size: 32px;
    font-weight: bold;
    color: {COLORS['primary_yellow']};
    padding: 5px;
}}

QLabel#result_confidence {{
    font-size: 22px;
    font-weight: bold;
    color: {COLORS['white']};
    padding: 3px;
}}

QLabel#result_risk {{
    font-size: 20px;
    font-weight: bold;
    padding: 3px;
}}

QComboBox {{
    background-color: {COLORS['light_navy']};
    border: 2px solid {COLORS['primary_yellow']};
    border-radius: 8px;
    padding: 12px 15px;
    font-size: 14px;
    color: {COLORS['white']};
    min-width: 200px;
}}

QComboBox:hover {{
    border-color: {COLORS['hover_yellow']};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 30px;
    border-left: 2px solid {COLORS['primary_yellow']};
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    background-color: {COLORS['light_navy']};
}}

QComboBox::drop-down:hover {{
    background-color: {COLORS['primary_yellow']};
}}

QComboBox::down-arrow {{
    width: 12px;
    height: 12px;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid {COLORS['primary_yellow']};
}}

QComboBox::down-arrow:hover {{
    border-top: 8px solid {COLORS['dark_navy']};
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['light_navy']};
    border: 2px solid {COLORS['primary_yellow']};
    border-radius: 5px;
    selection-background-color: {COLORS['primary_yellow']};
    selection-color: {COLORS['dark_navy']};
    color: {COLORS['white']};
    padding: 5px;
}}

QLineEdit {{
    background-color: {COLORS['light_navy']};
    border: 2px solid {COLORS['primary_yellow']};
    border-radius: 8px;
    padding: 12px 15px;
    font-size: 14px;
    color: {COLORS['white']};
}}

QLineEdit:focus {{
    border-color: {COLORS['hover_yellow']};
}}

QPushButton#predict_btn {{
    background-color: {COLORS['primary_yellow']};
    color: {COLORS['dark_navy']};
    border: none;
    border-radius: 10px;
    padding: 15px 40px;
    font-size: 18px;
    font-weight: bold;
}}

QPushButton#predict_btn:hover {{
    background-color: {COLORS['hover_yellow']};
}}

QPushButton#predict_btn:pressed {{
    background-color: #E09515;
}}

QFrame#result_frame {{
    background-color: {COLORS['light_navy']};
    border: 2px solid {COLORS['primary_yellow']};
    border-radius: 15px;
    padding: 20px;
}}

QFrame#team_frame {{
    background-color: {COLORS['light_navy']};
    border-radius: 12px;
    padding: 15px;
}}
"""


class SearchableComboBox(QComboBox):
    """Arama destekli ComboBox - inline dropdown"""
    
    def __init__(self, placeholder="Ara...", parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.lineEdit().setPlaceholderText(placeholder)
        self.setMaxVisibleItems(15)
        
        # Completer'i devre disi birak
        self.setCompleter(None)
        
        # Wayland odak sorunu cozumu:
        # Dropdown listesine eris
        view = self.view()
        
        # 1. Focus Policy Ayari: Odaklanamaz yap
        view.setFocusPolicy(Qt.NoFocus)
        
        # 2. Pencere Bayraklari: WindowDoesNotAcceptFocus ekle
        # Not: QComboBox popup'i olusturdugunda bu flag'i ezmemesi icin showPopup override edilecek
        
        # Arama filtreleme icin text degisikligini dinle
        self.all_items = []
        self._filtering = False
        self.lineEdit().textChanged.connect(self.filter_items)
    
    def setItems(self, items):
        """Ogeleri ayarla"""
        self.all_items = sorted(list(items))
        self._filtering = True
        self.clear()
        self.addItems(self.all_items)
        self.setCurrentIndex(-1)
        self._filtering = False
    
    def showPopup(self):
        """Dropdown'i ac - Odagi lineEdit'te tut"""
        # Popup penceresine eris ve aktive etmeden goster ozelligini ekle
        container = self.view().window()
        container.setAttribute(Qt.WA_ShowWithoutActivating)
        
        super().showPopup()
        
        # Guvenlik onlemi: Popup acildiktan hemen sonra odagi geri al
        QTimer.singleShot(10, self.restoreFocus)
        
    def restoreFocus(self):
        """Odagi line edit'e geri ver"""
        if self.lineEdit():
            self.lineEdit().setFocus()
            # Cursor pozisyonunu koru
            if self.lineEdit().text():
                self.lineEdit().end(False)
    
    def filter_items(self, text):
        """Yazilan metne gore ogeleri filtrele"""
        if self._filtering:
            return
        
        self._filtering = True
        
        try:
            cursor_pos = self.lineEdit().cursorPosition()
            
            if not text:
                self.clear()
                self.addItems(self.all_items)
                self.setCurrentIndex(-1)  # Otomatik secimi engelle
                # Texti temizle ki 100 Thieves gelmesin
                self.lineEdit().setText("")
            else:
                # Filtreleme
                filtered = [item for item in self.all_items if text.lower() in item.lower()]
                
                self.clear()
                if filtered:
                    self.addItems(filtered)
                else:
                    self.addItems(self.all_items)
                
                self.setCurrentIndex(-1) # Otomatik secimi engelle
                
                self.lineEdit().setText(text)
                self.lineEdit().setCursorPosition(cursor_pos)
                
                # Yazarken dropdown ac ama focus lineEdit'te kalsin
                if len(text) >= 2:
                    self.showPopup()
        finally:
            self._filtering = False


class CSGOPredictorGUI(QMainWindow):
    """Ana uygulama penceresi"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CS:GO Oracle V13 - Match Predictor")
        self.setFixedSize(700, 750)  # Yukseklik artirildi
        self.setStyleSheet(STYLESHEET)
        
        # Veri ve model yükleme
        self.load_data()
        self.load_model()
        
        # UI oluşturma
        self.init_ui()
    
    def get_resource_path(self, relative_path):
        """PyInstaller ve dev ortami icin dosya yolunu bul"""
        try:
            # PyInstaller gecici klasoru (_MEIPASS)
            base_path = sys._MEIPASS
        except Exception:
            # Normal calisma ortami
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        return os.path.join(base_path, relative_path)

    def load_data(self):
        """CSV dosyalarından veri yükle"""
        try:
            self.df_results = pd.read_csv(self.get_resource_path('results.csv'))
            self.df_players = pd.read_csv(self.get_resource_path('players.csv'))
            self.df_eco = pd.read_csv(self.get_resource_path('economy.csv'), low_memory=False)
            
            # Sütun isimlerini temizle
            for df in [self.df_results, self.df_players, self.df_eco]:
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Map sütununu standartlaştır
            map_col = '_map' if '_map' in self.df_results.columns else 'map'
            self.df_results.rename(columns={map_col: '_map'}, inplace=True)
            
            # Tarih dönüşümü
            self.df_results['date'] = pd.to_datetime(self.df_results['date'])
            self.df_players['date'] = pd.to_datetime(self.df_players['date'])
            
            # Team name normalization
            def normalize_names(series):
                return series.str.lower().str.strip().str.replace(' ', '_')
            
            self.df_results['team_1'] = normalize_names(self.df_results['team_1'])
            self.df_results['team_2'] = normalize_names(self.df_results['team_2'])
            
            # Takım listesi oluştur
            all_teams = set(self.df_results['team_1'].unique()) | set(self.df_results['team_2'].unique())
            self.all_teams = sorted([t.replace('_', ' ').title() for t in all_teams])
            
            # Harita listesi oluştur
            all_maps = self.df_results['_map'].dropna().unique()
            self.all_maps = sorted([m.title() for m in all_maps if isinstance(m, str)])
            
            # Feature Engineering
            self.prepare_features()
            
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            self.all_teams = []
            self.all_maps = []
    
    def prepare_features(self):
        """Feature engineering - demo.py'den alındı"""
        # Rank ve ELO
        self.df_results['rank_1'] = pd.to_numeric(self.df_results['rank_1'], errors='coerce').fillna(300)
        self.df_results['rank_2'] = pd.to_numeric(self.df_results['rank_2'], errors='coerce').fillna(300)
        
        tr = {}
        self.df_results.sort_values(by='date', inplace=True)
        
        def elo(t1, t2, w, k=30):
            r1, r2 = tr.get(t1, 1500), tr.get(t2, 1500)
            e1 = 1/(1+10**((r2-r1)/400))
            e2 = 1/(1+10**((r1-r2)/400))
            tr[t1], tr[t2] = r1+k*((1 if w==1 else 0)-e1), r2+k*((1 if w==2 else 0)-e2)
            return r1, r2
        
        el = [elo(r['team_1'], r['team_2'], r['match_winner']) for i, r in self.df_results.iterrows()]
        self.df_results['t1_elo'] = [x[0] for x in el]
        self.df_results['t2_elo'] = [x[1] for x in el]
        
        # Team momentum
        t1r = self.df_results[['date','team_1','match_winner']].rename(columns={'team_1':'team'})
        t1r['w'] = (t1r['match_winner']==1).astype(int)
        t2r = self.df_results[['date','team_2','match_winner']].rename(columns={'team_2':'team'})
        t2r['w'] = (t2r['match_winner']==2).astype(int)
        th = pd.concat([t1r, t2r]).sort_values(by=['team','date'])
        th['wr'] = th.groupby('team')['w'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1)).fillna(0.5)
        self.t_wr = th.groupby(['date','team'])['wr'].mean().reset_index()
        
        # Player stats
        if 'team_name' not in self.df_players.columns and 'team' in self.df_players.columns:
            self.df_players.rename(columns={'team': 'team_name'}, inplace=True)
        
        self.df_players['team_name'] = self.df_players['team_name'].str.lower().str.strip().str.replace(' ', '_')
        self.df_players.fillna(0, inplace=True)
        self.df_players['imp'] = (self.df_players['kills']*1.2 + self.df_players['assists']*0.3 + 
                                   self.df_players['fkdiff']*0.7 + self.df_players['kast']*0.05 + 
                                   self.df_players['adr']*0.01 - self.df_players['deaths']*0.5)
        self.df_players.sort_values(by=['player_name','date'], inplace=True)
        self.df_players['avg'] = self.df_players.groupby('player_name')['imp'].transform(
            lambda x: x.rolling(10, min_periods=1).mean().shift(1)).fillna(0)
        self.t_stats = self.df_players.groupby(['match_id','team_name'])['avg'].mean().reset_index()
        
        # Merge işlemleri
        df = pd.merge(self.df_results, self.t_stats, left_on=['match_id','team_1'], 
                      right_on=['match_id','team_name'], how='left').rename(columns={'avg':'t1_p'}).drop(columns=['team_name'], errors='ignore')
        df = pd.merge(df, self.t_stats, left_on=['match_id','team_2'], 
                      right_on=['match_id','team_name'], how='left').rename(columns={'avg':'t2_p'}).drop(columns=['team_name'], errors='ignore')
        df = pd.merge(df, self.t_wr, left_on=['date','team_1'], 
                      right_on=['date','team'], how='left').rename(columns={'wr':'t1_wr'}).drop(columns=['team'], errors='ignore')
        df = pd.merge(df, self.t_wr, left_on=['date','team_2'], 
                      right_on=['date','team'], how='left').rename(columns={'wr':'t2_wr'}).drop(columns=['team'], errors='ignore')
        
        df['t1_pis'] = 0.5
        df['t2_pis'] = 0.5
        
        df.dropna(subset=['t1_p','t2_p'], inplace=True)
        df.fillna(0.5, inplace=True)
        
        # Differential features
        df['elo_d'] = df['t1_elo'] - df['t2_elo']
        df['rank_d'] = df['rank_2'] - df['rank_1']
        df['perf_d'] = df['t1_p'] - df['t2_p']
        df['wr_d'] = df['t1_wr'] - df['t2_wr']
        df['pis_d'] = df['t1_pis'] - df['t2_pis']
        df['elo_x_wr'] = df['elo_d'] * df['wr_d']
        df['rank_x_perf'] = df['rank_d'] * df['perf_d']
        
        # One hot encoding maps
        df = pd.get_dummies(df, columns=['_map'], prefix='map')
        df.columns = df.columns.str.lower()
        
        self.full_history = df.sort_values(by='date').copy()
        self.map_cols = [c for c in df.columns if c.startswith('map_') and 'win' not in c and 'result' not in c]
        
        # Feature columns
        self.cols = ['t1_elo', 't2_elo', 'elo_d', 'rank_1', 'rank_2', 'rank_d', 
                     't1_wr', 't2_wr', 'wr_d', 't1_p', 't2_p', 'perf_d', 
                     't1_pis', 't2_pis', 'pis_d', 'elo_x_wr', 'rank_x_perf']
        self.cols += self.map_cols
        
        # Normalized team list
        self.team_lookup = {t.replace('_', ' ').title(): t for t in 
                           set(self.full_history['team_1'].unique()) | set(self.full_history['team_2'].unique())}
    
    def load_model(self):
        """Model yükle"""
        try:
            model_path = self.get_resource_path('csgo_best_model.pkl')
            self.model = joblib.load(model_path)
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            self.model = None
    
    def init_ui(self):
        """UI bileşenlerini oluştur"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 20, 30, 30)
        
        # Başlık
        title_layout = QVBoxLayout()
        title = QLabel("[CS:GO] ORACLE V13")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Stacking Model - Match Predictor")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        main_layout.addLayout(title_layout)
        
        # Takım seçim alanı
        teams_layout = QHBoxLayout()
        teams_layout.setSpacing(30)
        
        # Team 1
        team1_frame = QFrame()
        team1_frame.setObjectName("team_frame")
        team1_layout = QVBoxLayout(team1_frame)
        
        team1_label = QLabel("TEAM 1")
        team1_label.setObjectName("section_label")
        team1_label.setAlignment(Qt.AlignCenter)
        
        self.team1_combo = SearchableComboBox("Takim ara...")
        self.team1_combo.setItems(self.all_teams)
        
        team1_layout.addWidget(team1_label)
        team1_layout.addWidget(self.team1_combo)
        
        # VS Label
        vs_label = QLabel("VS")
        vs_label.setObjectName("vs_label")
        vs_label.setAlignment(Qt.AlignCenter)
        
        # Team 2
        team2_frame = QFrame()
        team2_frame.setObjectName("team_frame")
        team2_layout = QVBoxLayout(team2_frame)
        
        team2_label = QLabel("TEAM 2")
        team2_label.setObjectName("section_label")
        team2_label.setAlignment(Qt.AlignCenter)
        
        self.team2_combo = SearchableComboBox("Takim ara...")
        self.team2_combo.setItems(self.all_teams)
        
        team2_layout.addWidget(team2_label)
        team2_layout.addWidget(self.team2_combo)
        
        teams_layout.addWidget(team1_frame)
        teams_layout.addWidget(vs_label)
        teams_layout.addWidget(team2_frame)
        main_layout.addLayout(teams_layout)
        
        # Harita seçimi
        map_layout = QVBoxLayout()
        map_label = QLabel("MAP")
        map_label.setObjectName("section_label")
        map_label.setAlignment(Qt.AlignCenter)
        
        self.map_combo = SearchableComboBox("Harita ara...")
        self.map_combo.setItems(self.all_maps)
        self.map_combo.setFixedWidth(250)
        
        map_container = QHBoxLayout()
        map_container.addStretch()
        map_container.addWidget(self.map_combo)
        map_container.addStretch()
        
        map_layout.addWidget(map_label)
        map_layout.addLayout(map_container)
        main_layout.addLayout(map_layout)
        
        # Tahmin butonu
        btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton(">> PREDICT MATCH <<")
        self.predict_btn.setObjectName("predict_btn")
        self.predict_btn.setCursor(Qt.PointingHandCursor)
        self.predict_btn.clicked.connect(self.predict_match)
        
        # Gölge efekti
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(COLORS['primary_yellow']))
        shadow.setOffset(0, 3)
        self.predict_btn.setGraphicsEffect(shadow)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        
        # Sonuç alanı
        self.result_frame = QFrame()
        self.result_frame.setObjectName("result_frame")
        self.result_frame.setMinimumHeight(150)
        result_layout = QVBoxLayout(self.result_frame)
        result_layout.setSpacing(10)
        
        result_title = QLabel("RESULT")
        result_title.setObjectName("section_label")
        result_title.setAlignment(Qt.AlignCenter)
        
        self.result_winner = QLabel("")
        self.result_winner.setObjectName("result_winner")
        self.result_winner.setAlignment(Qt.AlignCenter)
        
        self.result_confidence = QLabel("")
        self.result_confidence.setObjectName("result_confidence")
        self.result_confidence.setAlignment(Qt.AlignCenter)
        
        self.result_risk = QLabel("")
        self.result_risk.setObjectName("result_risk")
        self.result_risk.setAlignment(Qt.AlignCenter)
        
        self.result_real = QLabel("")
        self.result_real.setAlignment(Qt.AlignCenter)
        self.result_real.setStyleSheet(f"color: {COLORS['white']}; font-size: 14px; margin-top: 10px;")
        
        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_winner)
        result_layout.addWidget(self.result_confidence)
        result_layout.addWidget(self.result_risk)
        result_layout.addWidget(self.result_real)
        
        main_layout.addWidget(self.result_frame)
        main_layout.addStretch()
    
    def predict_match(self):
        """Maç tahmini yap"""
        t1_display = self.team1_combo.currentText().strip()
        t2_display = self.team2_combo.currentText().strip()
        map_display = self.map_combo.currentText().strip()
        
        if not t1_display or not t2_display:
            self.result_winner.setText("[!] Her iki takimi da secin!")
            self.result_winner.setStyleSheet(f"color: {COLORS['warning']};")
            self.result_confidence.setText("")
            self.result_risk.setText("")
            self.result_real.setText("")
            return
        
        # Normalize names
        t1 = t1_display.lower().replace(' ', '_')
        t2 = t2_display.lower().replace(' ', '_')
        sel_map = map_display.lower() if map_display else "mirage"
        
        if not hasattr(self, 'full_history') or self.full_history.empty:
            self.result_winner.setText("[!] Veri yuklenemedi!")
            self.result_confidence.setText("")
            self.result_risk.setText("")
            self.result_real.setText("")
            return
        
        if self.model is None:
            self.result_winner.setText("[!] Model yuklenemedi!")
            self.result_confidence.setText("")
            self.result_risk.setText("")
            self.result_real.setText("")
            return
        
        try:
            # Takım geçmişlerini bul
            all_teams_lower = set(self.full_history['team_1'].unique()) | set(self.full_history['team_2'].unique())
            
            if t1 not in all_teams_lower or t2 not in all_teams_lower:
                self.result_winner.setText("[!] Takim bulunamadi!")
                self.result_winner.setStyleSheet(f"color: {COLORS['warning']};")
                self.result_confidence.setText("")
                self.result_risk.setText("")
                self.result_real.setText("")
                return
            
            # Son maç verisini kullan
            mask = ((self.full_history['team_1'] == t1) & (self.full_history['team_2'] == t2)) | \
                   ((self.full_history['team_1'] == t2) & (self.full_history['team_2'] == t1))
            
            matches = self.full_history[mask]
            
            real_outcome = None  # Gercek mac sonucu
            
            if matches.empty:
                # Takımların son formlarını kullan
                t1_hist = self.full_history[self.full_history['team_1'] == t1]
                t2_hist = self.full_history[self.full_history['team_1'] == t2]
                
                if t1_hist.empty or t2_hist.empty:
                    self.result_winner.setText("[!] Yetersiz mac gecmisi!")
                    self.result_confidence.setText("")
                    self.result_risk.setText("")
                    self.result_real.setText("")
                    return
                
                row1 = t1_hist.iloc[-1]
                row2 = t2_hist.iloc[-1]
                real_outcome = "Bu takimlar daha once karsilasmamis."
                
                features = {}
                features['t1_elo'] = row1['t1_elo']
                features['t2_elo'] = row2['t1_elo']
                features['elo_d'] = row1['t1_elo'] - row2['t1_elo']
                features['rank_1'] = row1['rank_1']
                features['rank_2'] = row2['rank_1']
                features['rank_d'] = row2['rank_1'] - row1['rank_1']
                features['t1_wr'] = row1['t1_wr']
                features['t2_wr'] = row2['t1_wr']
                features['wr_d'] = row1['t1_wr'] - row2['t1_wr']
                features['t1_p'] = row1['t1_p']
                features['t2_p'] = row2['t1_p']
                features['perf_d'] = row1['t1_p'] - row2['t1_p']
                features['t1_pis'] = row1.get('t1_pis', 0.5)
                features['t2_pis'] = row2.get('t1_pis', 0.5)
                features['pis_d'] = features['t1_pis'] - features['t2_pis']
                features['elo_x_wr'] = features['elo_d'] * features['wr_d']
                features['rank_x_perf'] = features['rank_d'] * features['perf_d']
            else:
                last_match = matches.iloc[-1]
                
                # Gercek sonucu kaydet
                w_code = last_match['match_winner']
                real_winner = last_match['team_1'] if w_code == 1 else last_match['team_2']
                match_date = last_match['date'].strftime('%Y-%m-%d') if hasattr(last_match['date'], 'strftime') else str(last_match['date'])[:10]
                real_outcome = f"Gercek Sonuc ({match_date}): {real_winner.upper().replace('_', ' ')} kazandi!"
                
                if last_match['team_1'] == t1:
                    features = last_match.to_dict()
                else:
                    features = last_match.to_dict()
                    swap_map = {
                        't1_elo': 't2_elo', 't2_elo': 't1_elo',
                        'rank_1': 'rank_2', 'rank_2': 'rank_1',
                        't1_wr': 't2_wr', 't2_wr': 't1_wr',
                        't1_p': 't2_p', 't2_p': 't1_p',
                        't1_pis': 't2_pis', 't2_pis': 't1_pis'
                    }
                    new_features = {}
                    for k, v in features.items():
                        if k in swap_map:
                            new_features[swap_map[k]] = v
                        else:
                            new_features[k] = v
                    features = new_features
                    for c in ['elo_d', 'rank_d', 'perf_d', 'wr_d', 'pis_d', 'elo_x_wr', 'rank_x_perf']:
                        if c in features:
                            features[c] = -features[c]
            
            # Map features
            for c in self.map_cols:
                features[c] = 0
            
            map_key = f"map_{sel_map}"
            if map_key in self.map_cols:
                features[map_key] = 1
            elif 'map_mirage' in self.map_cols:
                features['map_mirage'] = 1
            
            # DataFrame oluştur ve tahmin yap
            input_df = pd.DataFrame([features])
            
            # Eksik sütunları ekle
            for col in self.cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[self.cols]
            
            # Tahmin
            prob = self.model.predict_proba(input_df)[0]
            winner = self.model.predict(input_df)[0]
            
            w_team = t1_display if winner == 0 else t2_display
            conf = prob[0] if winner == 0 else prob[1]
            
            # Sonuclari goster
            self.result_winner.setText(f">>> {w_team.upper()} WINS! <<<")
            self.result_winner.setStyleSheet(f"color: {COLORS['primary_yellow']}; font-size: 28px; font-weight: bold;")
            
            self.result_confidence.setText(f"Confidence: {conf*100:.1f}%")
            
            # Risk degerlendirmesi
            if conf >= 0.70:
                risk_msg = "[OK] RELIABLE"
                risk_color = COLORS['success']
            elif conf >= 0.60:
                risk_msg = "[!] MEDIUM RISK"
                risk_color = COLORS['warning']
            else:
                risk_msg = "[?] HIGH RISK (Coin Flip)"
                risk_color = COLORS['danger']
            
            self.result_risk.setText(f"Risk: {risk_msg}")
            self.result_risk.setStyleSheet(f"color: {risk_color}; font-size: 16px;")
            
            # Gercek mac sonucunu goster
            if real_outcome:
                self.result_real.setText(real_outcome)
            else:
                self.result_real.setText("")
            
        except Exception as e:
            self.result_winner.setText(f"[X] Hata: {str(e)[:50]}")
            self.result_winner.setStyleSheet(f"color: {COLORS['danger']};")
            self.result_confidence.setText("")
            self.result_risk.setText("")
            self.result_real.setText("")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS['dark_navy']))
    palette.setColor(QPalette.WindowText, QColor(COLORS['white']))
    palette.setColor(QPalette.Base, QColor(COLORS['light_navy']))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS['dark_navy']))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS['white']))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS['dark_navy']))
    palette.setColor(QPalette.Text, QColor(COLORS['white']))
    palette.setColor(QPalette.Button, QColor(COLORS['light_navy']))
    palette.setColor(QPalette.ButtonText, QColor(COLORS['white']))
    palette.setColor(QPalette.Highlight, QColor(COLORS['primary_yellow']))
    palette.setColor(QPalette.HighlightedText, QColor(COLORS['dark_navy']))
    app.setPalette(palette)
    
    window = CSGOPredictorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
