from flask import Flask, request, render_template, jsonify
from PIL import Image
import io, torch

app = Flask(__name__)
# load custom fruits model if exists, else fallback to default YOLOv5s
weights = 'runs/train/fruits/weights/best.pt'
try:
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        weights,
        source='local',
        trust_repo=True
    )
    print(f"Loaded custom model from {weights}")
except Exception:
    model = torch.hub.load(
        'ultralytics/yolov5',
        'yolov5s',
        pretrained=True,
        trust_repo=True,
        force_reload=True 
    )
    print("Loaded default YOLOv5s model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    f = request.files['file']
    img_bytes = f.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # inference
    results = model(img)  # includes auto-preprocess, NMS
    # parse into list of dicts
    df = results.pandas().xyxy[0]  # pandas DataFrame
    preds = df.to_dict(orient='records')
    return jsonify(predictions=preds)

if __name__ == '__main__':
    app.run(debug=True)