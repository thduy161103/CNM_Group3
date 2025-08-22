.\yolovenv\Scripts\Activate.ps1
cd ./yolov5
pip install -r requirements.txt (tầm 5-10ph)

Lệnh train: (lúc trc tôi dể 16 thì hết RAM nên giảm xuống 4) (tầm khoảng hơn 1 tiếng)
python train.py --img 640 --batch 16 --epochs 50 --data data\furniture.yaml --weights yolov5s.pt --project runs\train\furniture --name exp1
python train.py --img 640 --batch 4 --epochs 50 --data data\furniture.yaml --weights yolov5s.pt --project runs\train\furniture --name exp1


# Ở máy mới, sau khi cài đặt môi trường và dependencies:
git clone https://github.com/thduy161103/CNM_Group3.git
cd CNM_Group3
python -m venv yolovenv
.\yolovenv\Scripts\Activate.ps1
pip install -r requirements.txt

python detect.py --weights runs\train\furniture\exp1\weights\best.pt --img 640 --source your_images/
