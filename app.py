import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

st.title("""온도와 풍속에 따른 체감온도""")
# image = Image.open('tem_bg5.png')
# st.image(image, caption=None)

number1 = st.number_input('온도')
st.write('현재 온도는?', number1)

number2 = st.number_input('풍속')
st.write('현재 풍속은?', number2)
# import matplotlib.pyplot as plt
# import warnings

# warnings.filterwarnings('ignore')
# %matplotlib inline

df = pd.read_csv('./tem.xls')
# df.head()

# print(df.info())

# import seaborn as sns
# sns.pairplot(df)

dataset = df.values

X = dataset[:, 0:2]
Y = dataset[:, 2]
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(X,Y,random_state=42)

from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
# print(train_poly.shape)

test_poly = poly.transform(test_input)
# print(test_input.shape)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target))
# print(lr.score(test_poly, test_target))
sample = PolynomialFeatures(include_bias=False)
sample.fit([[number1, number2]])
sample_poly = sample.transform([[number1, number2]])
# print(train_poly.shape)
result = lr.predict(sample_poly)
print(result)
if result<5:
    st.subheader("현재 체감온도는 "+ str(result)+"입니다.")
    st.subheader("엄청 추워요. 따뜻하게 입고다니세요.")
    image = Image.open('chuwi3.png')
    st.image(image, caption=None)
elif result<18:
    st.subheader("현재 체감온도는 "+ str(result)+"입니다.")
    st.subheader("쌀쌀합니다. 가벼운 외투 챙기세요.")
    image = Image.open('chuwi1.png')
    st.image(image, caption=None)
elif result<26:
    st.subheader("현재 체감온도는 "+ str(result)+"입니다.")
    st.subheader("날씨가 좋습니다. 나들이하기 좋은 날씨네요")
    image = Image.open('flower1.png')
    st.image(image, caption=None)
else:
    st.subheader("현재 체감온도는 "+ str(result)+"입니다.")
    st.subheader("덥습니다. 무더위 조심하세요.")
    image = Image.open('duwi1.png')
    st.image(image, caption=None)
