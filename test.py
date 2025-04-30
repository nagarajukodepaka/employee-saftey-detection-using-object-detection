from ultralytics import YOLO
import cv2
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine
import base64
import threading
from collections import Counter
import mysql.connector

# Database and YOLO Setup
engine = create_engine("mysql+pymysql://root:root@localhost/safety_detection")
model = YOLO("best.pt")

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="safety_detection"
)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    class VARCHAR(255) NOT NULL,
    count INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 
              'Safety Vest', 'machinery', 'vehicle'] 

cap = cv2.VideoCapture("huuman.mp4")
frame_data = {'frame': None, 'counts': {}}

def process_video():
    global frame_data
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        results = model(img)
        detected_classes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                detected_classes.append(classNames[cls])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, classNames[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', img)
        frame_data['frame'] = base64.b64encode(buffer).decode('utf-8')
        frame_data['counts'] = dict(Counter(detected_classes))

        # Insert detected counts into the database
        for class_name, count in frame_data['counts'].items():
            cursor.execute("INSERT INTO detections (class, count) VALUES (%s, %s)", (class_name, count))
        conn.commit()

threading.Thread(target=process_video, daemon=True).start()

# Dash App Setup
app = dash.Dash(__name__)

dark_theme = {
    'background': '#121212',
    'text': '#FFFFFF',
    'chart_bg': '#1E1E1E',
    'chart_grid': '#333333'
}

app.layout = html.Div([
    html.Div([
        html.Img(src='/assets/logo.png', style={'height': '40px', 'marginRight': '15px'}),
        html.A("Home", href="#", style={'color': dark_theme['text'], 'marginRight': '15px'}),
        html.A("Statistics", href="#", style={'color': dark_theme['text'], 'marginRight': '15px'}),
    ], style={'display': 'flex', 'alignItems': 'center'}),
    html.Div([
        html.Img(id='video-feed', style={'width': '60%', 'border': '2px solid white'}),
        html.Div(id='data-table', style={
            'width': '35%', 
            'backgroundColor': dark_theme['chart_bg'], 
            'padding': '10px', 
            'borderRadius': '5px'
        })
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    html.Div([
        dcc.Graph(id='pie-chart', style={'width': '33%', 'backgroundColor': dark_theme['chart_bg']}), 
        dcc.Graph(id='bar-chart', style={'width': '33%', 'backgroundColor': dark_theme['chart_bg']}), 
        dcc.Graph(id='line-chart', style={'width': '33%', 'backgroundColor': dark_theme['chart_bg']})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(
    [Output('video-feed', 'src'),
     Output('pie-chart', 'figure'),
     Output('bar-chart', 'figure'),
     Output('line-chart', 'figure'),
     Output('data-table', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n_intervals):
    try:
        query = "SELECT class, COUNT(*) AS count FROM detections GROUP BY class"
        df = pd.read_sql(query, engine)

        if df.empty:
            bar_fig = px.bar(title="No Data Available", template="plotly_dark")
            pie_fig = px.pie(title="No Data Available", template="plotly_dark")
            line_fig = px.line(title="No Data Available", template="plotly_dark")
        else:
            bar_fig = px.bar(df, x="class", y="count", title="Object Count", template="plotly_dark")
            pie_fig = px.pie(df, values="count", names="class", title="Detection Distribution", template="plotly_dark")
            line_fig = px.line(df, x="class", y="count", title="Detection Over Time", template="plotly_dark")

        table_rows = [
            html.Tr([html.Th("Class"), html.Th("Count")], style={'color': dark_theme['text']})
        ] + [
            html.Tr([html.Td(row['class']), html.Td(row['count'])]) for _, row in df.iterrows()
        ]
        table = html.Table(table_rows, style={'width': '100%', 'color': dark_theme['text']})

        video_src = f"data:image/jpeg;base64,{frame_data['frame']}" if frame_data['frame'] else None

        return video_src, pie_fig, bar_fig, line_fig, table

    except Exception as e:
        print(f"Error: {e}")
        return None, {}, {}, {}, "Error fetching data"

if __name__ == '__main__':
    app.run(debug=True)
s