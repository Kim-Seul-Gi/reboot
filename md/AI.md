# AI 실습1

## 1장. Numpy 다루기

```python
# 1-1

from elice_utils import EliceUtils
import numpy as np

elice_utils = EliceUtils()


def main():
	
    print("Array1: 파이썬 리스트로 만들어진 정수형 array")
    array1 = np.array([1,2,3], dtype=int)
    print("데이터:", array1)
    print("array의 자료형:", type(array1))
    print("dtype:", array1.dtype, "\n")

    print("Array2: 파이썬 리스트로 만들어진 실수형 array")
    array2 = np.array([1.1, 2.2, 3.3], dtype=float)
    print("데이터:", array2)
    print("dtype:", array2.dtype, "\n")

    print("Array3: 0으로 10개 채워진 정수형 array")
    array3 = np.zeros(10, dtype=int)
    print("데이터:", array3)
    print("dtype:", array3.dtype, "\n")

    print("Array4: 1으로 채워진 3x5형태 실수형 array")
    array4 = np.ones((3, 5), dtype=float)
    print("데이터:", array4)
    print("dtype:", array4.dtype, "\n")

    print("Array5: 0부터 9까지 담긴 정수형 array")
    array5 = np.linspace(0,9,10, dtype=int)
    print("데이터:", array5, "\n")

    print("Array6: 0부터 1사이에 균등하게 나눠진 5개의 값이 담긴 array")
    array6 = np.linspace(0, 1, 5)
    print("데이터:", array6, "\n")

    print("Array7: 0부터 10사이 랜덤한 값이 담긴 2x2 array")
    array7 = np.random.randint(0, 10, size=(2, 2))
    print("데이터:", array7, "\n")
    
    
if __name__ == "__main__":
    main()

```



```python
# 1-2

import numpy as np

array_1 = np.array([[4,2,5],[5,3,2],[9,1,2]])

#1. 배열 array_1에 대하여 2행 3열의 원소를 추출하세요. 
element_1 = array_1[1,2]
print("2행 3열의 원소는 ", element_1, " 입니다.")

#2. 배열 array_1에 대하여 3행을 추출하세요. 
row_1 = array_1[2]
print("3행은 배열 ", row_1, " 입니다.")

#3. 배열 array_1에 대하여 2열을 추출하세요. 
col_1 = array_1[:,1]
print("2열은 배열 ", col_1, " 입니다.")

#4. x의 1행과 3행을 바꾼 행렬을 만들어보세요. 
y = np.array([array_1[2], array_1[1], array_1[0]])
print(y)
```



```python
# 1-3

import numpy as np


def main():

    print(matrix_nom_var())
    print(matrix_uni_std())

def matrix_nom_var():
    
    # [[5,2,3,0], [3,4,5,1], [3,2,7,9]] 값을 갖는 A 메트릭스를 선언합니다.
    A = np.array([[5,2,3,0],[3,4,5,1],[3,2,7,9]])
    # print(A)

    # 주어진 A 메트릭스의 원소의 합이 1이 되도록 표준화 (Normalization) 합니다.
    sums = np.sum(A)
    A = A / sums
    # print(A)
    # print(np.sum(A))

    # 표준화 된 A 메트릭스의 분산을 구하여 리턴합니다.
    # print(np.var(A))
    return np.var(A)

def matrix_uni_std():
    
    # 모든 값이 1인 4 by 4 A 메트릭스를 생성합니다.
    A = np.ones((4,4))
    # print(A)

    # 표준화 된 A 메트릭스의 분산을 구하여 리턴합니다.
    return np.var(A)

main()
```



```python
# 1-4

import numpy as np

array1 = np.array([[1,2,3], [4,5,6], [7,8,9]])

#array1의 전치 행렬을 구해보자.
transposed = np.transpose(array1)
print(transposed, '는 array1을 전치한 행렬입니다.')    

#array1과 array1의 전치 행렬의 행렬곱을 구해보자.
power = np.dot(array1, transposed)
print(power,'는 array1과 array1의 전치 행렬을 행렬곱한 것입니다.')

#array1과 array1의 전치 행렬의 요소별 곱을 구해보자.
elementwise_prod = array1 * transposed
print(elementwise_prod, '는 array1과 array1의 전치행렬을 요소별로 곱한 행렬입니다.')


array2 = np.array([[2,3],[1,7]])

# array2의 역행렬을 만들어보자.
inverse_array2 = np.linalg.inv(array2)
print(inverse_array2,'는 array2의 역행렬입니다.')

# array2와 array2의 역행렬을 곱한 행렬을 만들어보자.
producted = np.dot(array2, inverse_array2)
print(producted,'는 array2와 array2의 역행렬을 곱한 행렬입니다.')
```



```python
# 1-5

import pandas as pd

def main():
    # Series()를 사용하여 1차원 데이터를 만들어보세요.
    # 5개의 age 데이터와 이름을 age로 선언해보세요.
    data = [19, 18, 27, 22, 33]
    age = pd.Series(data, name="age")

    # Python Dictionary 형태의 데이터가 있습니다.
    # class_name 데이터를 Series로 만들어보세요.
    class_name = {'국어' : 90,'영어' : 70,'수학' : 100,'과학' : 80}
    class_name = pd.Series(class_name)
    print(class_name,'\n')
    
    
    # DataFrame 만들기
    # DataFrame()을 사용하여 2차원 데이터를 생성해보세요.
    # index와 columns 값을 설정해보세요.
    data=[['name', 'age'],['철수',15],['영희',23],['민수',20],['다희', 18],['지수',20]]
    data = pd.DataFrame(data[1:], columns=data[0], index=[1,2,3,4,5])
    print(data,'\n')
    
    
if __name__ == "__main__":
    main()


```



```python
# 1-6

from elice_utils import EliceUtils
elice_utils = EliceUtils()
import pandas as pd

a = pd.Series([20, 15, 30, 25, 35], name='age')
b = pd.Series([68.5, 60.3, 53.4, 74.1, 80.7], name='weight')
c = pd.Series([180, 165, 155, 178, 185], name ='height')

data = {a.name : a.values, b.name : b.values,
c.name : c.values}

human = pd.DataFrame(data, index=[1,2,3,4,5]) # 병합했다고 생각하시길.
print(human)

def main():

    # loc(), iloc() 함수를 이용하여 특정 행, 열 추출 
    print(human.loc[1],'\n')
    print(human.iloc[1],'\n')
    
#     # loc(), iloc() 함수를 이용하여 데이터의 특정 범위 추출
    print(human.loc[1:3],'\n')
    print(human.iloc[1],'\n')
     
    sex = ['F','M','F','M','F']
#     # 새로운 데이터 추가하기
    human.loc[0] = [1, 2, 3]
    print(human,'\n')
    
# #     #원하는 행/열 데이터 삭제하기
    tmp = human.drop([3])
    print(tmp,'\n')


if __name__ == "__main__":
    main()
```



```python
# 1-6 다른 풀이입니다.

from elice_utils import EliceUtils
elice_utils = EliceUtils()
import pandas as pd

a = pd.Series([20, 15, 30, 25, 35], name='age')
b = pd.Series([68.5, 60.3, 53.4, 74.1, 80.7], name='weight')
c = pd.Series([180, 165, 155, 178, 185], name ='height')
human = pd.DataFrame([a, b, c])

def main():
    print(human)
    # loc(), iloc() 함수를 이용하여 특정 행, 열 추출 
    print(human.loc['age'],'\n')
    print(human.iloc[0],'\n')
    
    # loc(), iloc() 함수를 이용하여 데이터의 특정 범위 추출
    print(human.loc['weight'],'\n')
    print(human.iloc[2],'\n')
     
    sex = ['F','M','F','M','F']
    # 새로운 데이터 추가하기
    human.loc['sex'] = sex
    print(human,'\n')
    
    #원하는 행/열 데이터 삭제하기
    tmp = human.drop(['height'])
    print(tmp,'\n')


if __name__ == "__main__":
    main()
```





## 2장.  Linear Regression 이해하기

```python
# 2-1

# 실습에 필요한 패키지입니다. 수정하지 마세요.
import elice_utils
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
eu = elice_utils.EliceUtils()

# 실습에 필요한 데이터입니다. 수정하지마세요. 
X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

'''
beta_0과 beta_1 을 변경하면서 그래프에 표시되는 선을 확인해 봅니다.
기울기와 절편의 의미를 이해합니다.
'''

beta_0 = sum(Y) / sum(X)  # beta_0에 저장된 기울기 값을 조정해보세요. 
beta_1 = sum(Y)/len(Y) - beta_0*sum(X)/len(X) # beta_1에 저장된 절편 값을 조정해보세요.

plt.scatter(X, Y) # (x, y) 점을 그립니다.
plt.plot([0, 10], [beta_1, 10 * beta_0 + beta_1], c='r') # y = beta_0 * x + beta_1 에 해당하는 선을 그립니다.

plt.xlim(0, 10) # 그래프의 X축을 설정합니다.
plt.ylim(0, 10) # 그래프의 Y축을 설정합니다.

# 엘리스에 이미지를 표시합니다.
plt.savefig("test.png")
eu.send_image("test.png")
```



```python
# 2-2

import elice_utils
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
eu = elice_utils.EliceUtils()

def loss(x, y, beta_0, beta_1):
# x : X , y ; Y , beta_0 : 기울기, beta_1 : y절편
    N = len(x)
    answer=0
    result_array = []
    
    '''
    x, y, beta_0, beta_1 을 이용해 loss값을 계산한 뒤 리턴합니다.
    '''
    
    for i in range(N):
        real_y = y[i]
        predict_y = x[i] * beta_0 + beta_1        
        tmp = (real_y - predict_y)**2       
        result_array.append(tmp)
        answer += tmp
    print(result_array)    
    return answer/len(x)

X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

beta_0 = sum(Y) / sum(X) # 기울기
beta_1 = sum(Y)/len(Y) - beta_0*sum(X)/len(X) # 절편

print("Loss: %f" % loss(X, Y, beta_0, beta_1))

plt.scatter(X, Y) # (x, y) 점을 그립니다.
plt.plot([0, 10], [beta_1, 10 * beta_0 + beta_1], c='r') # y = beta_0 * x + beta_1 에 해당하는 선을 그립니다.

plt.xlim(0, 10) # 그래프의 X축을 설정합니다.
plt.ylim(0, 10) # 그래프의 Y축을 설정합니다.
plt.savefig("test.png") # 저장 후 엘리스에 이미지를 표시합니다.
eu.send_image("test.png")
```



```python
# 2-3 MSE 방식

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import elice_utils
eu = elice_utils.EliceUtils()

    
X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

train_X = np.array(X).reshape(-1, 1)
train_Y = np.array(Y).reshape(-1, 1)

'''
여기에서 모델을 트레이닝합니다.
'''
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

beta_0 = lrmodel.coef_[0]
beta_1 = lrmodel.intercept_
Loss = 0
for i in range(len(X)):
    Loss +=( (beta_0 * train_X[i] + beta_1) - train_Y[i] )**2
print(Loss/10) # 이게 mse 값을 의미합니다.
print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)

plt.scatter(X, Y) # (x, y) 점을 그립니다.
plt.plot([0, 10], [beta_1, 10 * beta_0 + beta_1], c='r') # y = beta_0 * x + beta_1 에 해당하는 선을 그립니다.

plt.xlim(0, 10) # 그래프의 X축을 설정합니다.
plt.ylim(0, 10) # 그래프의 Y축을 설정합니다.
plt.savefig("test.png") # 저장 후 엘리스에 이미지를 표시합니다.
eu.send_image("test.png")
```



```
# 2-3 개념 설명
단순 선형 회귀식을 y = ax + b 라고 가정하자
각 점의 좌표를 x[i], y[i] 라고 할 때 (x[i], y[i]는 상수입니다.)

(1) Loss = 시그마( (y - y[i])**2 ) (i는 각 좌표의 인덱스를 의미, )
	 = 시그마( (a*x[i] + b - y[i])**2 )  (x[i], y[i]는 변수가 아닌 상수입니다.)
	 의 형태를 띄고 있기 때문에, Loss 는 a, b의 차수가 2인 방정식이라고 정의할 수 있습니다.
	 
(2) Loss = c * a**2 + d * a * b + e * b**2 에서 c, e >= 0 이므로
(위에서 정의한 식을 보면 제곱이므로 무조건 0이상)

(3) Loss 를 a에 대해 편미분값=0 , (4) b에 대해 편미분값=0 
동시에 만족하는 지점이 Loss 값이 최소가 되는 지점입니다.

(3), (4) 두 개의 방정식이 있으므로 두 개의 변수(a, b) 값을 이론적으로 구할 수 있으며 이게 바로
Loss 값이 최소가 되는 선형 회귀식(y=ax+b) 을 의미합니다.

실제로 좌표를 3개정도 넣고 구한 값이랑, 코드로 구한 값이 같은 것을 확인할 수 있었습니다.
```



```python
# 2-4 gradient_descent 방식

import numpy as np
import elice_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
eu = elice_utils.EliceUtils()
#학습률(learning rate)를 설정한다.(권장: 0.0001~0.01)
learning_rate = 0.01 
#반복 횟수(iteration)를 설정한다.(자연수)
iteration = 100
def prediction(a,b,x):
    # 넘파이 배열 a,b,x를 받아서 'x*(transposed)a + b'를 계산하는 식을 만든다.
    # equation = x * a + b
    equation = x * np.transpose(a) + b
    return equation
    
def update_ab(a,b,x,error,lr):
    # a를 업데이트하는 규칙을 정의한다.
    delta_a = -(lr*(2/len(error))*(np.dot(x.T, error)))
    # b를 업데이트하는 규칙을 정의한다.
    delta_b = -(lr*(2/len(error))*np.sum(error))
    
    return delta_a, delta_b
    
def gradient_descent(x, y, iters):
    #초기값 a= 0, a=0
    a = np.zeros((1,1))
    b = np.zeros((1,1))
    
    for i in range(iters):
        #실제 값 y와 예측 값의 차이를 계산하여 error를 정의한다.
        
        error =  y - prediction(a,b,x) 
        
        a_delta, b_delta = update_ab(a,b,x,error,lr=learning_rate)
        
        a -= a_delta
        b -= b_delta
        
    return a, b

def plotting_graph(x,y,a,b):
    y_pred=a[0,0]*x+b
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.savefig("test.png")
    eu.send_image("test.png")

def main():

    x = 5*np.random.rand(100,1)
    y = 3*x + 5*np.random.rand(100,1)
    
    a, b = gradient_descent(x,y,iters=iteration)
    
    print("a:",a, "b:",b)
    plotting_graph(x,y,a,b)
    
main()

```



```python
# 2-5 gradient_descent 방식

import numpy as np
import elice_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
eu = elice_utils.EliceUtils()

#학습률(learning rate)를 설정한다.(권장: 0.0001~0.01)
learning_rate = 0.01
#반복 횟수(iteration)를 설정한다.(자연수)
iteration = 100
#릿지회귀에 사용되는 알파(alpha) 값을 설정한다.(권장: 0.0001~0.01)
alpha = 0.01

def prediction(a,b,x):  

    equation = x * np.transpose(a) + b
    return equation
    
def update_ab(a,b,x,error,lr, alpha):
    #alpha와 a의 곱으로 regularization을 설정한다.  
    regularization = a*alpha
    delta_a = -(lr*(2/len(error))*(np.dot(x.T, error)) + regularization)
    delta_b = -(lr*(2/len(error))*np.sum(error))
    return delta_a, delta_b
    
def gradient_descent(x, y, iters, alpha):
    #초기값 a= 0, a=0
    a = np.zeros((1,1))
    b = np.zeros((1,1))    
    
    for i in range(iters):
        error = y - prediction(a,b,x) 
        a_delta, b_delta = update_ab(a,b,x,error,lr=learning_rate, alpha=alpha)
        a -= a_delta
        b -= b_delta
    
    return a, b

def plotting_graph(x,y,a,b):
    y_pred=a[0,0]*x+b
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.savefig("test.png")
    eu.send_image("test.png")

def main():
    #x, y 데이터를 생성한다.
    x = 5*np.random.rand(100,1)
    y = 10*x**4 + 2*x + 1+ 5*np.random.rand(100,1)
    # a와 b의 값을 반복횟수만큼 업데이트하고 그 값을 출력한다. 
    a, b = gradient_descent(x,y,iters=iteration, alpha=alpha)
    print("a:",a, "b:",b)
    #회귀 직선과 x,y의 값을 matplotlib을 통해 나타낸다.
    plotting_graph(x,y,a,b)
    
main()
```



## 3. Naive Bayes 이해하기

```python
# 3-1

def main():
    sensitivity = float(input())
    prior_prob = float(input())
    false_alarm = float(input())

    print(mammogram_test(sensitivity, prior_prob, false_alarm))

def mammogram_test(sensitivity, prior_prob, false_alarm):

    """
    x=1 : 검진 결과 유방암 판정
    x=0 : 검진 결과 유방암 미판정
    y=1 : 유방암 발병됨
    y=0 : 유방암 미발병
    """

    # The likelyhood probability : 유방암을 가지고 있는 사람이 검진 결과 유방암 판정 받을 확률
    p_x1_y1 = sensitivity # p(x = 1|y = 1)

    # The prior probability : 유방암을 가지고 있을 확률로 매우 낮다.
    p_y1 = prior_prob # p(y = 1)

    # False alram : 유방암을 가지고 있지 않지만 검사 결과 유방암 판정을 받을 확률
    p_x1_y0 = false_alarm # p(x = 1|y = 0)
    # Bayes rule 
    p_y1_x1 = sensitivity * prior_prob / ((false_alarm*(1-prior_prob)) + (sensitivity*prior_prob)) # p(y = 1|x = 1)
    
    # 검사 결과 유방암 판정을 받은 환자가 정확한 검진을 받았단 확률
    return p_y1_x1

if __name__ == "__main__":
    main()

```



```python
# 3-3

import re
import math
import numpy as np

def main():
    M1 = {'r': 0.7, 'g': 0.2, 'b': 0.1} # M1 기계의 사탕 비율
    M2 = {'r': 0.3, 'g': 0.4, 'b': 0.3} # M2 기계의 사탕 비율
    
    test = {'r': 4, 'g': 3, 'b': 3}

    print(naive_bayes(M1, M2, test, 0.7, 0.3))

def naive_bayes(M1, M2, test, M1_prior, M2_prior):
    p_1 = 0.7*(0.7**4)*(0.2**3)*(0.1**3)
    p_2 = 0.3*(0.3**4)*(0.4**3)*(0.3**3)
    return [p_1/(p_1+p_2), p_2/(p_1+p_2)]

if __name__ == "__main__":
    main()

```

