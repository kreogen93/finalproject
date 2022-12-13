import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
from pydantic import BaseModel
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("C:\\Users\\OMEN\\PycharmProjects\\pythonProject14\\catboost_model")
    from_file = CatBoostClassifier(iterations=200,
                              learning_rate=1,
                              depth=2,
                              random_seed=12345)
    from_file.load_model(model_path)
    return from_file


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
    return (batch_load_sql('''SELECT * 
                            FROM public.user_info_features
                            ;'''), batch_load_sql('''SELECT * FROM public.posts_info_features'''),
            batch_load_sql('''SELECT distinct post_id, user_id 
                            FROM public.feed_data
                            where action='like'
                            ;'''))
user_feature, post_feature, liked_posts = load_features()
cat = load_models()

import os
from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime

app = FastAPI()

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True



@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
		id: int,
		time: datetime,
		limit: int = 10) -> List[PostGet]:
    global post_feature, user_feature, liked_posts, cat
    user = user_feature.loc[user_feature['user_id'] == id].drop(['user_id', 'index'], axis=1)
    posts = post_feature.drop(['index', 'text'], axis=1)
    content = post_feature[['post_id', 'topic', 'text']]
    add_user_features = dict(zip(user.columns, user.values[0]))
    user_post_features = posts.assign(**add_user_features).set_index('post_id')
    user_post_features['hour'] = time.hour
    user_post_features['month'] = time.month
    predicts = cat.predict_proba(user_post_features)[:, 1]
    user_post_features['predicts'] = predicts
    like = liked_posts[liked_posts['user_id'] == id]['post_id'].values
    filtered_ = user_post_features[~user_post_features.index.isin(like)]
    recommended_posts = filtered_.sort_values('predicts')[-limit:].index
    return [
        PostGet(**{'id': i,
                   'text': content[content['post_id'] == i]['text'].values[0],
                   'topic': content[content['post_id'] == i]['topic'].values[0]}) for i in recommended_posts]



if __name__ == '__main__':
    print(recommended_posts(6991, datetime(2022, 12, 12, 16), 5))
