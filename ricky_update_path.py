from ultralytics import settings


def update_path():
    # 出力先を指定
    settings.update({"runs_dir": "./"})
    settings.reset()
