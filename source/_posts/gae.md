---
title: HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION (GAE)
date: 2018-07-07 11:20:45
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 양혁렬
subtitle: 피지여행 7번째 논문
---
<br>
<br>
# HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION (GAE)
<img src="https://www.dropbox.com/s/99w9yj4mclxjlsw/Screenshot%202018-07-08%2009.10.43.png?dl=1">

- [논문 링크]("https://arxiv.org/abs/1506.02438")
- Policy gradient 에서 효과적으로 Variance를 줄일 수 있는 방법을 제안. 
- GAE를 사용하면, 파라미터 $\lambda$ 와 $\gamma$를 통해서 bias-variance trade-off 를 조절할 수 있음.

<br>
## 5.Value Fuction Estimation

Value function 의 Estimation 을 설명하는 파트입니다.</br>
- Tust Region Opimization Scheme 에 따라 Value function 을 Estimation 합니다. 
- 
### 5.1 Simplest approach
<center>$ minimize_{\phi} \sum_{n=1}^{N} \vert\vert V_{\phi}(s_n) - \hat{V_n} \vert\vert^{2} $</center>

- 위는 가장 간단하게 non-linear approximation 으로 푸는 방법입니다. 
- $ \hat{V_t} = \sum_{l=0}^{\infty}\gamma^l r_{t+l}$ 은 reward 에 대한 discounted sum 을 의미합니다. 

### 5.2 Trust region method to optimize the value function

- Value function 을 최적화 하기 위해 trust region method 를 사용합니다. 
- Trust region 은 최근 데이터에 대해 overfitting 되는 것을 막아줍니다. 

Trust region 문제를 풀기 위해서는 다음 스텝을 따릅니다.

- $ \sigma^2 = \frac{1}{N} \sum_{n=1}^{N} \vert\vert V_{\phi old}(s_n) - \hat{V_n} \vert\vert^{2} $ 을 계산합니다. 
- 그 후에 다음과 같은 constrained opimization 문제를 풉니다. 



<center> $ minimize_{\phi} \sum_{n=1}^{N} \vert\vert V_{\phi}(s_n) - \hat{V_n} \vert\vert^{2} $ </center>

<center> subject to $ \frac{1}{N} \sum_{n=1}^{N}  \frac{\vert\vert V_{\phi}(s_n) - V_{\phi old}(s_n) \vert\vert^{2}}{2\sigma^2} \le \epsilon $ </center>

- 위의 수식은 사실 old Value function과 새로운 Value function KL distance 가 $\epsilon$ 보다 작아야한다는 수식과 같습니다. Value function이 평균은 $V_{\phi}(s)$ 이고 분산이 $\sigma^2$인 conditional Gaussian distribution 으로 parameterize되었을 뿐입니다. 
    - <img width ="400px" src="https://www.dropbox.com/s/uw0v05feu8chqmc/Screenshot%202018-07-08%2010.04.38.png?dl=1">


이 trust region 문제의 답을 conjudate gradient algorithm 을 이용하여 근사값을 구할 수 있습니다. 특히, 다음과 같은  quadratic program 을 풀게됩니다. 

<center> $ minimize_{\phi}$  $ g^{T}(\phi - \phi_{old}) $ </center>

<center> subject to $  \frac{1}{N} \sum_{n=1}^{N} (\phi - \phi_{old})^{T} H (\phi - \phi_{old}) \le \epsilon $ </center>


- 여기서 g 는 objective 의 gradient 입니다. 
- $j_n = \nabla_{\phi} V_{\phi}(s_n) $ 일때, $H = \frac{1}{N} \sum_{n} j_n j^T_n $ 이며, H 는 objective의 hessian에 대해서 gaussian newton method 로 근사한 값입니다. 따라서, value function 을 conditional probability로 해석한다면 Fisher information matrix 가 됩니다. 
- 구현할때의 방법은 TRPO 에서 사용한 방법과 모두 같습니다. 


## 6.Experiments

실험은 다음 두 가지 물음에 대해서 디자인 되었습니다.

- GAE 에 따라서 episodic total reward 를 최적화 할때, $\lambda $와 $\gamma$ 가 변함에 따라서 어떤 경험적인 효과를 볼 수 있는지 ?
- GAE 와 trust region alogorithm 을 policy 와 value function 모두에 함께 사용했을 때 어려운 문제에 적용되는 큰 뉴럴넷을 최적화 할 수 있을까 ?


### 6.1 Policy Optimization Algorithm

Policy update는 TRPO 를 사용합니다. TRPO 에 대한 설명은 여기서는 생략하겠습니다. TRPO 포스트를 보고 돌아와주세요 !

- 이전 TRPO에서 이미 TRPO 와 다른 많은 알고리즘들을 비교하였기 때문에, 여기서는 똑같은 짓을 반복하지 않고 $\lambda $, $\gamma$ 가 변함에 따라 어떤 영향이 있는 지에 대한 실험에 집중하겠다고 합니다. (귀찮았던거죠)

- TRPO 를 적용한 GAE의 최종 알고리즘은 다음과 같습니다.
<center> <img width = "600px" src="https://www.dropbox.com/s/b1klz11f2frrvg4/Screenshot%202018-07-08%2010.33.54.png?dl=1"> </center>
<center> <img width = "300px" src="https://www.dropbox.com/s/35fxte05pqtiel7/Screenshot%202018-07-08%2010.35.22.png?dl=1"> </center>
<center> <img width = "300px" src="https://www.dropbox.com/s/u35pjow3w50bvz1/Screenshot%202018-07-08%2010.36.03.png?dl=1"> </center>


    - 여기서 주의할 점은 Policy update ( $\theta_i \rightarrow \theta_{i+1}$ )에서 $ V_{\phi_i} $ 를 사용했다는 점입니다. 
    - 만약 Value function을 먼저 update하게 된다면 추가적인 bias가 발생합니다.
    - 극단적으로 생각해보아서, 우리가 Value function을 완벽하게 overfit 해낸다면 Bellman residual ($ r_t + \gamma V(s_{t+1}) - V(S_t) $)은 0 이 됩니다. 그럼 Policy gradient 의 estimation 도 거의 0이 될 것입니다. 

    
### 6.2 Expermint details

#### 6.2.1 Environment 
실험에서 사용된 환경은 다음 네 가지 입니다. 

1. classic cart-pole (x 3D)
2. bipedal locomotion 
3. quadrupedal locomotion 
4. dynamically standing up for the biped

#### 6.2.2 Architecture

- 3D robot task에 대해서는 같은 모델을 사용하였습니다. 
    - layers  = [100, 50, 25] 각각 tanh 사용.(Policy와 Value 네트워크 모두)
    - Final output layer은 linear
- Cartpole 에 대해서는 1개의 layer 안에 20개의 hidden unit 만 있는 linear policy를 사용했다고 합니다. 

#### 6.2.3 Task

- Cartpole
    - 한 배치당 20 개의 trajectory 를 모았고, maximum length 는 1000 입니다. 
- 3D biped locomotion
    - 33 dim state , 10 dim action
    - 50000 time step per batch
- 3D quadruped locomotion
    - 29 dim state, 8 dim action
    - 200000 time step per batch 
- 3D biped locomotion Standing
    - 33 dim state , 10 dim action
    - 200000 time step per batch 


#### 6.2.3 results
cost 의 관점에서 결과를 나타내었다고 합니다. Cost 는 negative reward와 이것이 최소화 되었는가로 정의되었다고 하는데, 정확히는 안나와있습니다. 
##### 6.2.3.1 Cartpole

<center> <img width = "500px" src="https://www.dropbox.com/s/x9pbms1wvg38lda/Screenshot%202018-07-08%2011.08.22.png?dl=1"> </center>

- 왼쪽 그림은 $\gamma$ 를 0.99로 고정시켜놓은 상태에서 $\lambda$ 를 변화시킴에 따라서 cost 를 측정한 것입니다. 
- 오른쪽은 $\gamma$ 와 $\lambda$ 를 둘 다 변화 시키면서 performance 를 그림으로 나타낸 표입니다. 흰색에 가까울 수록 좋은 퍼포먼스입니다. 

##### 6.2.3.2 3D BIPEDAL LOCOMOTINO
<center> <img width = "500px" src="https://www.dropbox.com/s/i9wj4p6ijojsy82/Screenshot%202018-07-08%2011.25.08.png?dl=1"> </center>


- 다른 random seed 로 부터 9번 씩 시도한 결과를 mean 을 취해서 사용함
- Best performance 는   $\gamma \in [0.99, 0.995]$ 그리고  $\lambda \in [0.96, 0.99]$ 일때. 
- 1000 iteration 후에 빠르고 부드럽고 안정적인 걸음거이가 나옴
- 실제로 걸린 시간은 0.01(타입스텝당 시간) * 50000(배치당 타임스텝) * 1000(배치) * 3600(초->시간) * 24 = 5.8일 정도 

##### 6.2.3.3 다른 ROBOT TASKS
<center> <img width = "500px" src="https://www.dropbox.com/s/fuuat65we52quht/Screenshot%202018-07-08%2011.33.11.png?dl=1"> </center>

- 다른 로봇 TASK에 대해서는 아주 제한적인 실험만 진행함.(시간이 부족했던듯)
- Quadruped 에 대해서는 $\gamma$ =0.995  로 fix, $\lambda \in ${0, 0.96}
- Standingup 에 대해서는 $\gamma$ =0.99  로 fix, $\lambda \in ${0, 0.96}


## 7. Discussion

### 7.1 future work

- Value function estimation error 와 Policy gradient estimation error 사이의 관계를 알아낸다면, 우리는 Value function fitting 에 더 잘 맞는 error metric 을 사용할 수 있다. (policy gradient estimation 의 정확성과 더 잘 맞는 value function)
- Policy와 Value function의 파라미터를 공유하는 모델을 만드는 것은 아주 흥미롭고 이점이 많다. 하지만 수렴을 보장하도록 적절한 numerical optimization을 제시하여야 한다. 
- DDPG 는 별로다. TD(0) 는 bias 가 너무 크고, poor performance 로 이끈다. 특히나 이 paper 에서는 low-dimention 의 쉬운 문제들만 해결했다. 

<center> <img width = "500px" src="https://www.dropbox.com/s/nhc7t9psul5lr3x/Screenshot%202018-07-08%2011.45.15.png?dl=1"> </center>

### 7.2 FAQ 

- Compatible features 와는 무슨 관계 ?
     - Compatible features 는 value function을 이용하는 policy gradient 알고리즘들과 함께 자주 언급된다 .
     - Actor Critic 의 저자는 policy의 제한된 representation power 때문에, policy gradient 는 단지 advantage function space 의 subspace에만 의존하게된다. 
     - 이 subspace는 compatible features 에 의해 span 된다. 
     - 이 이론은 현재 문제 구조를 어떻게 이용해야 advantage function에 대해 더 나은 estimation 을 할 수 있는 지에 대한 지침을 주지 않는다. GAE paper 의 idea 와 orthogonal 하다 . 
- 왜 Q function을 사용하지 않는가 ?
     - 먼저 state-value function이 더 낮은 차원의 input 을 가진다. 그래서 Q function보다 더 배우기 쉽다. 
     - 두 번째로 이 페이퍼에서 제안하는 방법으로는 high bias estimator 에서 low bias estimator 로 $\lambda$를 통해서 부드럽게 interpolate 할수 있다. 
     - 반면에 Q 를 사용하면 단지 high-bias estimator 밖에 사용할 수 없다. 
     - 특히나 return에 대한 one-step estimation 은 엄두를 못낼 정도로 bias 가 크다.
