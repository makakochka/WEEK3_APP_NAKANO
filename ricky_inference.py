import os
import shutil
from ultralytics import YOLO
from ricky_update_path import update_path

update_path()

# detectディレクトリからpredictトナのついたディレクトリを削除
detect_dir = "runs/detect"
if os.path.exists(detect_dir):
    for dir in os.listdir(detect_dir):
        if "predict" in dir:
            shutil.rmtree(f"{detect_dir}/{dir}")
# ricky_resultsディレクトリを削除して作り直す
if os.path.exists("ricky_results"):
    shutil.rmtree("ricky_results")
    os.makedirs("ricky_results")


# inference_dataの中のファイルを推論
file_names = os.listdir("inference_data")
for file_name in file_names:
    source = f"inference_data/{file_name}"
    print(source)
    model = YOLO("C:/Users/makak/camp/python/week3_app_nakano/runs/detect/train4/weights/best.pt")
    model.predict(
        source,
        save=True,
        imgsz=640,
        conf=0.5,
    )

# 出力結果をricky_resultsディレクトリにコピー
result_dirs = os.listdir(detect_dir)
print(result_dirs)
for dir in result_dirs:
    file = os.listdir(f"{detect_dir}/{dir}")[0]
    filepath = f"{detect_dir}/{dir}/{file}"
    shutil.copy(filepath, f"./ricky_results")

print("Done")
