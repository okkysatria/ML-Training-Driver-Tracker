QSS = """
* { font-family: 'Plus Jakarta Sans','Segoe UI','Noto Sans',Arial,sans-serif; font-size: 13px; color: #0F172A; }
QMainWindow, QWidget#page, QScrollArea { background: #F8FAFC; border: none; }
QFrame#card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; }
QFrame#tile { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; }
QFrame#sidebar { background: #0F172A; }
QFrame#sidebar QPushButton { color: #CBD5E1; background: transparent; border: none; text-align: left; padding: 11px 18px; border-radius: 8px; }
QFrame#sidebar QPushButton:checked, QFrame#sidebar QPushButton:hover { background: #1E293B; color: #FFFFFF; }
QWidget#brandWidget { border-bottom: 1px solid #1E293B; padding-bottom: 8px; margin-bottom: 4px; }
QLabel#brandTitle { color: #FFFFFF; font-weight: 700; font-size: 14px; }
QLabel#brandSubtitle { color: #10B981; font-weight: 600; font-size: 11px; }
QLabel#brandLogo { border: none; background: transparent; }
QLabel#brand { color: #FFFFFF; font-weight: 700; font-size: 15px; padding: 14px 18px; }
QLabel#sidestat { color: #94A3B8; font-size: 12px; padding: 3px 18px; }
QLabel#h1 { font-size: 20px; font-weight: 700; }
QLabel#title { font-size: 15px; font-weight: 700; }
QLabel#muted, QLabel#hint { color: #64748B; }
QLabel#hint { font-size: 12px; }
QLabel#kpi { font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono',Consolas,'DejaVu Sans Mono',monospace; }
QLabel#mono, QLabel#metrics { font-family: 'JetBrains Mono',Consolas,'DejaVu Sans Mono',monospace; }
QLabel[chip="good"] { background: #D1FAE5; color: #065F46; border-radius: 10px; padding: 3px 12px; font-weight: 700; }
QLabel[chip="ok"]   { background: #FEF3C7; color: #92400E; border-radius: 10px; padding: 3px 12px; font-weight: 700; }
QLabel[chip="bad"]  { background: #FEE2E2; color: #991B1B; border-radius: 10px; padding: 3px 12px; font-weight: 700; }
QPushButton, QToolButton { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 6px 12px; }
QPushButton:hover, QToolButton:hover { background: #F1F5F9; }
QPushButton:checked, QToolButton:checked { background: #4F46E5; color: #FFFFFF; border-color: #4F46E5; }
QPushButton:disabled, QToolButton:disabled { color: #94A3B8; background: #F1F5F9; border-color: #E2E8F0; }
QToolButton::menu-indicator { image: none; width: 0px; }
QPushButton#primary { background: #059669; color: #FFFFFF; border: none; font-weight: 700; padding: 9px 18px; }
QPushButton#primary:hover { background: #047857; }
QPushButton#primary:disabled { background: #A7F3D0; color: #FFFFFF; }
QPushButton#dark { background: #0F172A; color: #FFFFFF; border: none; }
QPushButton#dark:disabled { background: #94A3B8; }
QLineEdit, QComboBox, QDateTimeEdit, QDateEdit, QDoubleSpinBox, QSpinBox, QPlainTextEdit { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 5px 8px; }
QLineEdit:focus, QComboBox:focus, QDateTimeEdit:focus, QSpinBox:focus { border: 1px solid #4F46E5; }
QListWidget, QTableView { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; alternate-background-color: #F8FAFC; }
QListWidget::item { padding: 8px; border-bottom: 1px solid #F1F5F9; }
QListWidget::item:selected { background: #ECFDF5; color: #065F46; }
QHeaderView::section { background: #F1F5F9; border: none; border-bottom: 1px solid #E2E8F0; padding: 6px; font-weight: 600; }
QTabWidget::pane { border: none; }
QTabBar::tab { padding: 6px 12px; background: #F1F5F9; border-radius: 6px; margin-right: 3px; }
QTabBar::tab:selected { background: #4F46E5; color: #FFFFFF; }
QProgressBar { border: 1px solid #E2E8F0; border-radius: 6px; text-align: center; background: #FFFFFF; }
QProgressBar::chunk { background: #059669; border-radius: 5px; }
QMenu { background: #FFFFFF; border: 1px solid #E2E8F0; padding: 4px; }
QMenu::item { padding: 7px 20px; border-radius: 6px; }
QMenu::item:selected { background: #EEF2FF; }
QStatusBar { background: #FFFFFF; border-top: 1px solid #E2E8F0; color: #475569; }
QToolTip { background: #0F172A; color: #FFFFFF; border: none; padding: 6px; }
QScrollBar:vertical { width: 10px; background: transparent; }
QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""
