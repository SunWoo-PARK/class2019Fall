#!/usr/bin/env python
# coding: utf-8

# # 중간고사(#11~20)

# In[1]:


a = [[1], [2,3], [4,5,6]]
n = 0
for b in a:
    for d in b:
        n += 1
print(n)


# In[2]:


n = [1, 2, 3, 4]
for i in range(len(n)):
    print(i)


# 12. 총 몇 개의 unique한 함수가 있는가?
# range, len, print 해서 총 3개의 함수

# 13. 함수가 총 몇 번 실행되는가?
# len 한 번 range 한 번 print 네 번 해서 총 여섯번

# In[3]:


a = [1, 2, [3, 4]]
print(a[-1][-1])


# In[4]:


b = [1, 'a', {'a':[3, 6], 'b':[9]}, 'b']
print(b[-2]['b'][-1])


# In[5]:


c = [1, 'a', {'a': 'abc', 'b':'def'}]
print(c[-1]['b'][-2])


# In[6]:


d = ['th', 's']
print('i'.join(d)[-3])


# In[7]:


e = [1, 1.2, 'k', [1, [2]], {'a', 'apple'}]
for x in e:
    if type(x) == 'dict':
        print(x)
    else:
        if x == 'k':
            print(x)


# In[8]:


t = 0
for p in range(3):
    if p <=1:
        for q in range(4):
            t += q
        for r in range(2):
            t += r
            print(t, r)
print(t*r)


# In[9]:


a = [1, 2, 3, 4]
if len(a) == 3:
    print(0)
else:
    print(1)


# In[10]:


a = (5.2, {1: 'apple', 'k': 4.9}, (9.1, [10, 12.1]))
print(int(a[2][-1][0]))


# # Numpy

# 배열 연산
# 

# In[11]:


a = [1, 3, 5]
b = [2, 4, 6]
c = a + b
c


# In[12]:


import numpy 


# In[13]:


A = numpy.array(a)
B = numpy.array(b) #array는 행렬을 벡터로 만드는 것 원래 리스트 상태에서는 계산이 불가능한데 array 통해서 계산이 가능해짐


# In[14]:


A + B


# In[15]:


type(A) #nd 는 n dimensional????


# In[16]:


import numpy as np #numpy 를 np로 줄여 쓸 수 있음


# In[17]:


A = np.array(a)
B = np.array(b)


# In[18]:


A + B


# In[19]:


X = np.array([[1,2,3], [4,5,6]]) #큰 리스트 안 작은 리스트 -> 행렬로 나타낼 때 


# In[20]:


X


# In[21]:


X.shape #차원 -> 2 by 3 줄이 행 2개에 열이 3열로 되는 행렬 


# In[22]:


import numpy as np
import matplotlib.pyplot as plt #matplot이라는 library에 있는 pyplot을 불러올건데 이걸 plt라고 부르겠다


# In[23]:


np.empty([2,3], dtype='int') #numpy의 empty 함수 -> 입력 리스트로 들어감 2by3행렬 int라고 데이터 타입을 선택해서 다 int로 나옴 나오는 값은 랜덤함


# In[24]:


np.zeros([2,3]) #2by3의 0으로 이뤄진 행렬을 만들라는 뜻


# In[25]:


np.ones([2,3]) #.찍혀있다는 건 float zeros, ones 의 default data type은 float -> dtype 'int'로 하면 점 사라질 것


# In[26]:


np.ones([2,3], dtype = 'int')


# In[27]:


np.ones([2,3], dtype = 'float64') #float 64는 소수점 몇번째 자리까지..? float 64가 제일 정교


# In[28]:


np.array(([0, 0, 0], [0, 0, 0])) #array 리스트에서 array로 convert


# In[29]:


np.arange(5) #range와 비슷하나 계산할 수 있는 array로 만들어줌 


# In[30]:


np.arange(0, 10)


# In[31]:


np.arange(0, 10, 2) #증가분을 2로 해서 -> 2씩 띄고 값 나옴


# In[32]:


np.arange(0, 10, 2, dtype='float64')


# In[33]:


np.linspace(0, 10, 6) #0부터 10까지 arange와 달리 10도 포함 이걸 총 6개로 똑같이(등간격) 나누어준다 처음과 끝을 포함하여!


# In[34]:


np.linspace(0, 10, 7) 


# In[35]:


x = np.array([[1, 2, 3], [4, 5, 6]])
x


# In[36]:


x = np.array([[1, 2], [4, 5], [8, 9]]) #대괄호 2개[[]] -> 2차원 3차원은 대괄호 3개 [[[]]]
x


# In[37]:


x = np.array([[[1, 2], [4, 5], [8, 9]],[[1, 2], [4, 5], [8, 9]]]) #3차원!
x


# In[38]:


x.ndim #x가 몇차원인지 알고 싶을 때


# In[39]:


x.shape #x는 2X3X2의 차원이다 삼차원 


# In[40]:


x.dtype


# In[41]:


x.astype(np.float64) #type바꾸고 싶을 때


# x라는 행렬 안 값 0으로 바꾸는 방법 (2)

# In[42]:


#1
np.zeros_like(x) #모두 변경시키고 싶을 때 


# In[43]:


#2
x * 0


# In[44]:


data = np.random.normal(0, 1, 100) #정규분포(normal distribution 종모양 데이터 만들어줌) 종의 shape을 만들기 위해 필요한 두가지 정보: (mean, 격차, 데이터의 개수)
print(data)
plt.hist(data, bins=10) #plt 속 hist라는 함수 histogram 바구니 안에 공 부을 때 공이 쌓여서 올라가는 것과 같이 그린 그래프
plt.show()


# 바구니 총 10개, 그 레인지 속 공을 던졌을 때 들어가는 값이 몇개있는지가 y축() -> y축은 0을 포함한 자연수 값) x에 해당하는 y축 값 다 더하면 총 데이터 개수인 100개

# In[45]:


data.ndim


# In[46]:


data.shape #총 데이터 개수 100개


# 한 줄(벡터) 1차원
# 직사각형 2차원
# 직사각형 여러장 쌓이면 3차원
# 그게 여러장 쌓이면 4차원...

# In[102]:


X = np.ones([2, 3, 4]) #이 행렬의 element는 총 2X3X4 = 24개(-> 1by 6by 4 처럼 곱했을 때 그대로 원소 개수 유지되는 행렬로 바꿀 수 있음)
X 


# 대괄호 3개 -> 3차원

# In[103]:


Y = X.reshape(-1, 3, 2) #앞선 2by 3by 4였던 X의 shape을 바꾸고 싶을 때 reshape 쓰면 됨 3by 2는 뒤에 할거고 그 나머지 하나가 뭔지 정확히 모르겠을 때 4대신 -1쓰면 알아서 들어감
Y


# In[104]:


Y = X.reshape(4, 3, 2)
Y


# In[105]:


np.allclose(X.reshape(-1, 3, 2), Y) #두개의 array 비교해서 똑같으면 True 값 출력


# In[106]:


a = np.random.randint(0, 10, [2, 3]) 
b = np.random.random([2, 3]) #random하게 만들되 2by3로
np.savez("test", a, b) #만든 데이터를 파일로 저장


# In[107]:


get_ipython().system('ls -al test* #내 컴퓨터에선 안돌아감 그냥 파일 만들어졌는지 확인하기 위한 코드')


# In[108]:


del a, b
get_ipython().run_line_magic('who', '#지금 available 한 variable이 뭐가 있는지 출력')


# In[109]:


npzfiles = np.load("test.npz") #위에서 만든 npz 파일 불러오고 싶을 때 
npzfiles.files


# In[110]:


npzfiles['arr_0'] #이렇게 하면 a라고 정의한 값이 출력됨


# In[111]:


npzfiles['arr_1'] #이렇게 하면 b라고 정의한 값 출력됨


# # 과제
# 
# CSV
# - CSV(영어: comma-separated values)는 몇 가지 필드를 쉼표(,)로 구분한 텍스트 데이터 및 텍스트 파일이다. 확장자는 .csv이며 MIME 형식은 text/csv이다.

# In[112]:


data = np.loadtxt("regression.csv", delimiter=",", skiprows=1, dtype={'names':("X", "Y"), 'formats':('f', 'f')})
data


# In[58]:


np.savetxt("regression_saved.csv", data, delimiter=",")
get_ipython().system('ls -al regression_saved.csv')


# In[59]:


arr = np.random.random([5, 2, 3])


# In[60]:


print(type(arr)) #numpy가 만들어낸 array라는 뜻
print(len(arr))
print(arr.shape)
print(arr.ndim)
print(arr.size) #총 element의 개수
print(arr.dtype) # 무엇을 default로 만들어졌는지


# In[65]:


a = np.arange(1, 5)     #[1, 2, 3, 4]
b = np.arange(9, 5, -1) #[9, 8, 7, 6]


# In[66]:


print(a - b)
print(a * b)


# In[67]:


a = np.arange(1, 10).reshape(3,3) #[1, 2, 3, ,,, 9]이걸 reshape 해서 3by 3로 만듦
b = np.arange(9, 0, -1).reshape(3,3) #[9, 8, 7, 6, ,,,1]이것도 reshape 해서 3 by 3 987/654/321
print(a)
print(b)


# In[68]:


a == b #각각 대응하는 원소 같은지 비교, 비교하려는 대상의 차원과 shape이 같아야만 비교가 가능함


# In[69]:


a > b


# In[70]:


a


# In[71]:


a.sum(), np.sum(a) #numpy 속 들어있는 sum이라는 함수를 쓰거나 a 자체가 numpy가 만들어낸 산물이기 때문에 a 자체가 numpy 따라서 a.sum()이라고 쓸 수 있는 것 다만 자기 자신을 sum 하는 것이므로 이 경우 괄호 안 input 넣지 않아도 됨


# In[72]:


a.sum(axis=0), np.sum(a, axis=0) #몇번째 차원에서 sum을 할 것인가 a는 3by 3로 2차원인데 첫번째 차원에 해당하는게 행  첫번째 차원의 관점에서(가로) sum을 하라는 것이므로 결과도 1 + 4+ 7 = 12 ...


# In[73]:


a.sum(axis=1), np.sum(a, axis=1) #두번째 차원에서 sum = 세로로 접는 것 1+ 2+ 3 = 6,,,


# In[74]:


a = np.arange(1, 25).reshape(4, 6)


# In[75]:


a #a는 4by 6 matrix


# In[76]:


a + 100


# In[77]:


b = np.arange(6); b


# In[78]:


a+ b #a의 차원은 4by 6이고 b의 차원은 6 인데 이 둘 더하기 하면 전체적으로 b가 더해짐 


# # preview
# sound 컴퓨터에 담을 땐 하나하나 값으로 담김 간격은 동일 1초 동안 얼마나 빽빽하게 값들을 담을 건가 듬성듬성하게 담을 건가가 sampling rate 만약 샘플링 rate를 10000으로 한다면 1초동안 총 10000개의 값을 담는 다는 것 1초에 숫자가 10000개의 데이터로 있다 1초에 몇개 있다 사운드에 대하여 -> Hz(1초에 몇번 떨리느냐) 
