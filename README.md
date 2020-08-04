# 利用三层神经网络与信号传递过程(삼층신경망과 신호전달과정)

### 主要原理(주요원리)
我用三层神经网络为对象，实现从输入到输出的处理。使用Numpy数组，可以完成神经网路的前向处理。(삼층신경망을 대상으로 입력에서 출력까지의 처리를 실현하고 Numpy 함수를 이용하여 신경망으로의 전방향 처리가 할수있다.)
![캡처](https://user-images.githubusercontent.com/60682087/89238750-178c4900-d632-11ea-9a24-fa92340eeae0.JPG)

三层神经网络：输入层（第零层）有两个神经元，第一个隐藏层（第一层）有三个神经元，第二个隐藏层（第二层）有两个神经元，输出层（第三层）有两个神经元。
### 导入新符号
![캡처1](https://user-images.githubusercontent.com/60682087/89239518-5f13d480-d634-11ea-8062-771853c37f04.JPG)

如图所示，权重和隐藏层的神经元的右上角有一个“(1)”，它表示权重和神经元的层号（即第1层的权重、第1层的神经元）。此外，权重的右下角有两个数字，它们是后一层的神经元和前一层的神经元的索引号。
### 各层间信号传递的实现
![캡처2](https://user-images.githubusercontent.com/60682087/89239746-0abd2480-d635-11ea-9f81-4f196be98622.JPG)

用数学公式表示为：

![캡처3](https://user-images.githubusercontent.com/60682087/89239832-5079ed00-d635-11ea-89cb-5a77498fd514.JPG)

用矩阵表示为：

![캡처4](https://user-images.githubusercontent.com/60682087/89239938-a6e72b80-d635-11ea-9836-c59636dfa022.JPG)

即：

![캡처5](https://user-images.githubusercontent.com/60682087/89240024-ddbd4180-d635-11ea-993f-3d3d124517d9.JPG)

也可以写转置后的表达式：

![캡처6](https://user-images.githubusercontent.com/60682087/89240139-3096f900-d636-11ea-9d81-58bafa2a5309.JPG)

即：

![캡처7](https://user-images.githubusercontent.com/60682087/89240140-312f8f80-d636-11ea-8bc7-2d2660521b09.JPG)

下面用Numpy计算 A(1), 这里将输入信号、权重、 偏置设置成任意值。

X = np.array([1.0, 0.5])

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])

B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) # (2, 3)

print(X.shape)  # (2,)

print(B1.shape) # (3,)

A1 = np.dot(X, W1) + B1

接下来，将A(1) 代入激活函数 sigmoid（）转换成信号 Z(1)

![캡처8](https://user-images.githubusercontent.com/60682087/89240613-9637b500-d637-11ea-8296-1fd5c1a0c6dd.JPG)

Z1 = sigmoid(A1)

print(A1) # [0.3, 0.7, 1.1] 

print(Z1) # [0.57444252, 0.66818777, 0.75026011

同理，可以描述第二第三层信号传递过程。

![캡처9](https://user-images.githubusercontent.com/60682087/89240943-88cefa80-d638-11ea-9aa4-0a1579a52f70.JPG)

![캡처10](https://user-images.githubusercontent.com/60682087/89240946-8a002780-d638-11ea-9b07-05edf08bc84c.JPG)

输出层所用的激活函数，要根据求解问题的性质决定。一般地，回归问题可以使用恒等函数（如上图的σ()σ() 函数），二元分类问题可以使用sigmoid函数， 多元分类问题可以使用softmax函数。

最后，总代码如下：

def sigmoid(x): #激活函数

return 1/(1+np.exp(-x)) 

def identity_function(x):  #恒等函数

return x

def init_network():  #初始化参数

network = {}

network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])

network['b1'] = np.array([0.1, 0.2, 0.3])

network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])

network['b2'] = np.array([0.1, 0.2])

network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])

network['b3'] = np.array([0.1, 0.2])

return network

def forward(network, x):  #信号传递过程计算

W1, W2, W3 = network['W1'], network['W2'], network['W3'] 

b1, b2, b3 = network['b1'], network['b2'], network['b3']

a1 = np.dot(x, W1) + b1

z1 = sigmoid(a1)

a2 = np.dot(z1, W2) + b2

z2 = sigmoid(a2)

a3 = np.dot(z2, W3) + b3

y = identity_function(a3)

return y

network = init_network()

x = np.array([1, 0.5])

y = forward(network, x)

print(y)  #[0.31682708 0.69627909]

- 这个项目是我为了重新学习人工智能而做的项目。（이 프로젝트는 내가 인공지능을 다시 공부하기위해서 만든 프로젝트입니다.）
