import os
import shutil
from datetime import datetime


def delete_old_files(path_target, time_elapsed):
    """path_target:삭제할 파일이 있는 디렉토리, days_elapsed:경과일수"""
    for f in os.listdir(path_target):  # 디렉토리를 조회한다
        f = os.path.join(path_target, f)
        if os.path.isfile(f):  # 파일이면
            timestamp_now = datetime.now().timestamp()  # 타임스탬프(단위:초)
            # st_mtime(마지막으로 수정된 시간)기준 X일 경과 여부
            is_old = os.stat(f).st_mtime < timestamp_now - (time_elapsed)
            if is_old:  # X일 경과했다면
                try:
                    os.remove(f)  # 파일을 지운다
                    print(f, "is deleted: file")  # 삭제완료 로깅
                except OSError:  # Device or resource busy (다른 프로세스가 사용 중)등의 이유
                    print(f, "can not delete")  # 삭제불가 로깅
        else:
            timestamp_now = datetime.now().timestamp()  # 타임스탬프(단위:초)
            # st_mtime(마지막으로 수정된 시간)기준 X일 경과 여부
            is_old = os.stat(f).st_mtime < timestamp_now - (time_elapsed)
            if is_old:
                shutil.rmtree(f)
                print(f, "is deleted: folder")


try:
    while True:
        delete_old_files("/opt/ml/level3_cv_finalproject-cv-18/app/record", 14400)  # 4시간:14400
        delete_old_files("/opt/ml/level3_cv_finalproject-cv-18/app/temp", 14400)
except Exception as e:
    print(e)
