---
title: Deep Determinstic Policy Gradient (DDPG)
date: 2018-06-26 11:20:45
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 양혁렬
subtitle: 피지여행 3번째 논문
---

<center > <img src="https://www.dropbox.com/s/zv8rk0uf87ipiaj/Screenshot%202018-06-23%2010.01.48.png?dl=1" width="600"> </center>

논문 저자 : Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra
논문 링크 : https://arxiv.org/pdf/1509.02971.pdf
Proceeding : ??
정리 : 양혁렬

---
## 1.Introduction
DDPG 알고리즘에 대한 개요입니다.
### 1.1 Success & Limination of DQN  
-  Success
    - sensor로 부터 나오는 전처리를 거친 input 대신에 raw pixel input 을 사용 <br>
     : High dimensional observation space 문제를 풀어냄.
- Limitation
    - discrete & low dimensional action space 만 다룰 수 있음 <br>
     : Continuous action space 를 다루기 위해서는 매 스텝 이를 위한 iterative optimization process 를 거쳐야 함.

### 1.2 Problems of discritization
<p align="center">
<img src="https://www.dropbox.com/s/nulhzxs8bak2fn6/Screenshot%202018-06-23%2012.22.20.png?dl=1" width="300px">
</p>


- 만약 7개의 관절을 가진 로봇 팔이 있다면, 가장 간단한 discretization 은 각 관절을 다음과 같이 $a_{i}\in \\{ -k, 0, k \\}$ 3 개의 값을 가지도록 하는 것이다.
- 그렇다면 $3 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3 = 3^{7} = 2187$ 가지의 dimension을 가진 action space가 만들어진다.
    - Discretization 을 하면 action space 가 exponential 하게 늘어남.
- 충분히 큰 action space 임에도 discretization으로 인한 정보의 손실도 있다
    - 섬세한 Control 을 할 수 없다.

### 1.3 New approach for continuous control

- Model-free, Off-policy, Actor-critic algorithm 을 제안.
- Deep Deterministic Policy(이하 DPG) 를 기반으로 함.
- Actor-Critic approach 와 DQN의 성공적이었던 부분을 합침
    - Replay buffer : 샘플들 사이의 상관관계를 줄여줌
    - target Q Network : Update 동안 target 을 안정적으로 만들어줌.

## 2.Background

### 2.1 Notation
- Observation : $x_{t}$
- Action : $a_t \in {\rm IR}^N $
- Reward : $r_t$
- Discount factor : $\gamma$
- Environment : $E$
- Policy : $\pi : S \rightarrow P(A)  $
- Transition dynamics : $p(s_{t+1} \vert s_t, a_t) $
- Reward function : $r(s_t, a_t)$
- Return : $ \sum_{ i=t }^{ T } \gamma^{(i-t)}r(s_i, a_i) $
- Discounted state visitation distribution for a policy : $\rho^\pi $


### 2.2 Bellman Equation
- 상태 $s_t$ 에서 행동 $a_t$를 취했을 때 Expected return
  $ Q^{\pi}(s_t, a_t)={\rm E}_{r_{i \geqq t},s_{i \geqq t} \backsim E, a_{i \geqq t} \backsim \pi } [R_{t} \vert s_t, a_t  ] $
  <br>
- 벨만 방정식을 사용하여 위의 식을 변형
  $ Q^{\pi}(s_t, a_t)={\rm E}_{r_{t},s_{t} \backsim E } \[r(s_t,a_t)+\gamma {\rm E}_{a_{t+1} \backsim \pi } \[ Q^{\pi}(s_{t+1}, a_{t+1}) \] \] $
  <br>
- Determinsitc policy 를 가정
  $ Q^{\mu}(s_t, a_t)={\rm E}_{r_{t},s_{t} \backsim E } \[r(s_t,a_t)+\gamma Q^{\mu}(s_{t+1}, \mu (s_{t+1})) \] $
  - 위의 식에서 아래 식으로 내려오면서 policy 가 determinstic 하기 때문에 policy 에 dependent 한 Expectation 이 빠진 것을 알 수 있습니다. 
  - Deterministic policy를 가정하기 전의 수식에서는 $a_{t+1}$ 을 골랐던 순간의 policy로 Q에 대한 Expection을 원래 구해야하기 때문에 off-policy가 아니지만, Determinsitic policy를 가정한다면 update 할 당시의 policy로 $a_{t+1}$ 를 구할 수 있기 때문에 off-policy 입니다. 
  <br>
- Q learning 
$L(\theta^{Q}) = {\rm E}_{s_t \backsim \rho^\beta , a_t \backsim \beta , r_t \backsim E} [(Q(s_t, a_t \vert \theta^Q)-y_t)^2] $ 참고)  $\beta$ 는 behavior policy를 의미합니다. 
    - $ y_t = r(s_t, a_t) + \gamma Q^{\mu}(s_{t+1},\mu(s_{t+1})) $  
    - $\mu(s) = argmax_{a}Q(s,a)$
        - Q learning 은 위와 같이 $argmax$ 라는 deterministic policy를 사용하기 때문에 off policy 로 사용할 수 있습니다. 

### 2.3 DPG
<center> $ \nabla_{\theta^\mu} J \approx  {\rm E}_{s_t \backsim \rho^\beta} [ \nabla_{\theta^\mu} Q(s, a \vert \theta ^ Q) \vert_{s=s_t, a=\mu(s_t)}] $ <br> $ = {\rm E}_{s_t \backsim \rho^\beta} [ \nabla_{a} Q(s, a \vert \theta ^ Q) \vert_{s=s_t, a=\mu(s_t)} \nabla_{\theta^\mu} \mu(s \vert Q^{\mu})\vert_{s=s_t}] $ </center>

- 위의 수식은 피지여행 DPG 글 4-2.Q-learning 을 이용한 off-policy actor-critic 에서 이미 정리 한 바 있습니다.



## 3.Algorithm
Continous control 을 위한 새로운 알고리즘을 제안합니다. 제안하는 알고리즘의 특징은 다음과 같습니다. 

- Replay buffer 를 사용하였다.
- "soft" target update 를 사용하였다. 
- 각 차원의 scale이 다른 low dimension vector 로 부터 학습할 때 Batch Normalization을 사용하였다. 
- 탐험을 위해 action에 Noise 를 추가하였다. 

### 3.1 Replay buffer
<center>
<img src="https://www.dropbox.com/s/lc61b8nas1clqme/Screenshot%202018-06-23%2016.32.53.png?dl=1" width="600px"></center>

- 큰 State space를 학습하고 일반화 하기위해서는 Neural Network와 같은 non-linear approximator 가 필수적이지만 수렴한다는 보장이 없음.
- NFQCA 에서는 수렴의 안정성을 위해서 batch learning을 도입함. 하지만 NFQCA에서는 업데이트시에 policy를 reset 하지 않음.
- DDPG 는 DQN에서 사용된 Replay buffer를 사용하여 online batch update를 가능하게 했음.

### 3.2 Soft target update
<center> 
$ \theta^{Q^{'}} \leftarrow \tau \theta^{Q} + (1-\tau) \theta^{Q^{'}}   $<br>
$ \theta^{\mu^{'}} \leftarrow \tau \theta^{\mu} + (1-\tau) \theta^{\mu^{'}}   $
</center>

- DQN에서는 일정 주기마다 origin network의 weight 를 target network로 직접 복사해서 사용했음.
- DDPG 에서는 exponential moving average(지수이동평균) 식으로 대체
- soft update 가 DQN에서 사용했던 방식에 비해 어떤 장점이 있는지는 명확하게 설명되어있지 않음.

### 3.3 Batch Normalization
<center> 
<img src="https://www.dropbox.com/s/1erxzrgk69x04j8/Screenshot%202018-06-23%2016.51.56.png?dl=1" width="300px">
</center>

- 서로 scale 이 다른 feature 를 state 로 사용할 때에 Neural Net이 일반화에서 어려움을 겪는다. 
    - 이걸 해결하기 위해서는 원래 직접 스케일을 조정해주었음.
- 하지만 각 layer 의 Input 을 Unit Gaussian 이 되도록 강제하는 BatchNormalization을 사용하여 이 문제를 해결하였다.

### 3.4 Noise Process
DDPG 에서는 Exploration을 위해서 output 으로 나온 행동에 노이즈를 추가해줍니다.
<center>
ORNSTEIN UHLENBECK PROCESS(이하 OU) : $dx_t = \theta (\mu - x_t) dt + \sigma dW_t$
</center>

- OU Process는 평균으로 회귀하는 random process 입니다.
- $\theta$ 는 얼마나 빨리 평균으로 회귀할 지를 나타내는 파라미터이며  $\mu$ 는 평균을 의미합니다.
- $\sigma$ 는 process 의 변동성을 의미하며 $W_t$ 는 Wiener process 를 의미합니다. <br>
- 따라서 이전의 noise들과 temporally correlated 입니다.
- 위와 같은 temporally correlated noise process를 사용하는 이유는 physical control 과 같은 관성이 있는 환경에서 학습 시킬 때 보다 효과적이기 때문입니다.

### 3.5 Diagram & Pseudocode 

<img src="https://www.dropbox.com/s/0ffb2c9irctjx2n/Screenshot%202018-06-23%2017.58.57.png?dl=1"> 

- DDPG 의 학습 과정을 간단히 도식화 해본 다이어 그램입니다. 

<center>
<img src="https://www.dropbox.com/s/fd0nj7goixfnd6z/Screenshot%202018-06-23%2018.02.13.png?dl=1" width="500px"></center>

- DDPG 의 알고리즘 수도코드입니다. 모든 항목을 위에서 설명했으니 순서대로 보시면 이해에 도움이 될것입니다. 


## 4. Results

### 4.1 Variants of DPG
<center>
<img src="https://www.dropbox.com/s/k8q6tih85lgow5l/Screenshot%202018-06-23%2018.08.05.png?dl=1" width="500px"></center>

- original DPG 에 batchnorm 만 추가 (연한 회색), target network만 추가 (진한 회색), 둘 다 추가 (초록), pixel로만 학습 (파랑) . Target network가 성능을 가장 좌지우지한다.

### 4.2 Q estimation of DDPG
<center>
<img src="https://www.dropbox.com/s/vvcfoni0lqrisst/Screenshot%202018-06-23%2018.10.23.png?dl=1" width="500px"></center>

- DQN 은 Q value 를 Over-estimate 하는 경향이 있었지만, DDPG 는 simple task에 대해서는 잘한다. 복잡한 문제에 대해서는 estimation을 잘 못했지만, 여전히 좋은 Policy를 찾아내었다.

### 4.3 Performance Comparison
<center>
<img src="https://www.dropbox.com/s/u8ibmz9q4kfxh6l/Screenshot%202018-06-23%2018.11.40.png?dl=1" width="500px"></center>

- Score 는 naive policy 를 0, ILQG (planning algorithm) 의 mean score를 1점으로 놓았을 때의 점수 Torcs 환경에 대해서만 raw reward 를 score로 사용.

## 5. Implementation Details

### 5.1 Hyper parameters

- Optimizer : Adam
    - actor lr : 0.0001, critic lr : 0.001
- Weight decay(L2) for critic(Q) = 0.001 
- discount factor, $\gamma = 0.99 $
- soft target updates. $\tau = 0.001 $
- Size of replay buffer = 1,000,000
- Orstein Uhlenbeck Process : $\theta = 0.15$, $\sigma = 0.2$


### 5.2 Etc.

- Final output layer of actor : tanh (행동의 최소 최대를 맞춰주기 위해서)
- low-dimentional 문제에서 네트워크는 2개의 hidden layer (1st layer 400 units, 2nd layer 300 units) 를 가진다. 
- 이미지를 통해서 학습시킬 때 : 3 convolutional layers (no pooling) with 32 filters at each layer.
- actor 와 critic 각각의 final layer(weight, bias 모두 ) 는 다음 범위의 uniform distribution 에서 샘플링한다. [ - 0.003, 0.003], [ - 0.0003 , 0.0003] . 이렇게 하는 이유는 가장 처음의 policy와 value 의 output 이 0에 가깝게 나오도록 하기 위함. 


<br>
## 6.Conclusion
<br>

- 이 연구는 최근 딥러닝의 발전과 강화학습을 엮은 것으로 Continuous action space 를 가지는 문제를 robust 하게 풀어냄.
- non-linear function approximators 을 쓰는 것은 수렴을 보장하지 않지만, 여러 환경에 대해서 특별한 조작 없이 안정적으로 수렴하는 것을 실험으로 보여냄.
- Atari 도메인에서 DQN 보다 상당히 적은 step 만에 수렴하는 것을 실험을 통해서 알아냄. 
- model-free 알고리즘은 좋은 solution 을 찾기 위해서는 많은 sample을 필요로 한다는 한계가 있지만 더 큰 시스템에서는 이러한 한계를 물리칠 정도로 중요한 역할을 하게 될 것임.