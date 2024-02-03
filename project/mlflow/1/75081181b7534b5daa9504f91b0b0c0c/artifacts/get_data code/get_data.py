import requests
import json
from pyyoutube import Api
import pandas as pd
import os

import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/andrey/project/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

key = "AIzaSyC-h2wJbvW66s70iMGMjD27l3GK7QpvxTM"
api = Api(api_key=key)

query = "'Taylor Swift'"
video = api.search_by_keywords(q=query, search_type=["video"], count=10, limit=30)

maxResults = 100
nextPageToken = ""

# Создаем списки
text_length = [] # Количество символов в комментарии
reply_count = [] # Количество откликов на комментарий
like_count = []  # Количество лайков

with mlflow.start_run():
    for id_ in [x.id.videoId for x in video.items]: # Заполняем списки
        uri = "https://www.googleapis.com/youtube/v3/commentThreads?" + \
                "key={}&textFormat=plainText&" + \
                "part=snippet&" + \
                "videoId={}&" + \
                "maxResults={}&" + \
                "pageToken={}"
        uri = uri.format(key, id_, maxResults, nextPageToken)
        content = requests.get(uri).text
        data = json.loads(content)
        for item in data['items']:
            str = item['snippet']['topLevelComment']['snippet']['textDisplay']
            text_length.append(len(str))
            reply_count.append(int(item['snippet']['totalReplyCount']))
            like_count.append(int(item['snippet']['topLevelComment']['snippet']['likeCount']))
    mlflow.log_artifact(local_path="/home/andrey/project/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()
        
# Из полученных списков создаем словарь, а затем датафрейм
dict = {'text_length': text_length, 'reply_count': reply_count, 'like_count': like_count} 
df = pd.DataFrame(dict)

# Записываем датафрейм
df.to_csv('/home/andrey/project/datasets/data.csv')
