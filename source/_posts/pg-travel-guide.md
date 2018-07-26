---
title: PG Travel Guide
date: 2018-06-29 01:11:26
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 공민서, 김동민, 양혁렬, 이동민, 이웅원, 장수영, 차금강
subtitle: 피지여행에 관한 개략적 기록
---

---

# 1. Policy Gradient의 세계로

반갑습니다! 저희는 PG여행을 위해 모인 PG탐험대입니다. 강화학습하면 보통 Q-learning을 떠올립니다. 그렇지만 오래전부터 Policy Gradient라는 Q-learning 못지 않은 훌륭한 방법론이 연구되어 왔고, 최근에는 강화학습의 최정점의 기술로 자리매김하고 있습니다. 강화학습의 아버지인 Sutton의 논문을 필두로 하여 기존의 DQN보다 뛰어난 성능을 내는 DPG와 DDPG, 그리고 현재 가장 주목받는 강화학습 연구자인 John Schulmann의 TRPO, GAE, PPO와 이를 이해하기 위해 필요한 Natural Policy Gardient까지 더불어 살펴보고자 합니다.

<center> <img src="https://www.dropbox.com/s/tbcyhvilaqy4ra0/Policy%20Optimization%20in%20the%20RL%20Algorithm%20Landscape.png?dl=1" width="800"> </center>

위의 그림은 강화학습 알고리즘 landscape에서 Policy Optimization의 관점을 중점적으로 하여 나타낸 그림입니다. 위의 그림에서 빨간색 작은 숫자로 나타낸 것이 저희 PG여행에서 다룰 논문들입니다. 순서는 다음과 같습니다.

1. Sutton_PG
2. DPG
3. DDPG
4. NPG
5. TRPO
6. GAE
7. PPO

위와 같이 총 7가지 논문들을 리뷰하여 블로그로 정리하였습니다. 이 블로그에는 각각의 기술을 제안한 논문을 PG탐험대분들이 자세하게 리뷰한 포스트들이 있습니다. 우리나라에서 PG에 대해서 이렇게 자세하게 리뷰한 포스트들은 없었다고 감히 말씀드리고 싶습니다. 본 글에서는 이 포스트들을 읽기 전에 전체 내용을 개략적으로 소개하고 각각의 포스트들로 안내하고자 합니다. 자, 저희와 함께 PG여행을 즐겨보시겠습니까?

<br><br>

# 2. \[Sutton PG\] Policy gradient methods for reinforcement learning with function approximation

[Sutton PG 여행하기](../../../06/15/sutton-pg/)
[Sutton PG Code](https://github.com/reinforcement-learning-kr/pg_travel)

[Sutton PG 여행하기](../../../06/15/sutton-pg/)
[Sutton PG Code](https://github.com/reinforcement-learning-kr/pg_travel)

<br><br>

# 3. \[DPG\] Deterministic policy gradient algorithms

[DPG 여행하기](../../../06/16/dpg/)
[DPG Code](https://github.com/reinforcement-learning-kr/pg_travel)

[DPG 여행하기](../../../06/16/dpg/)
[DPG Code](https://github.com/reinforcement-learning-kr/pg_travel)

<br><br>

# 4. \[DDPG\] Continuous control with deep reinforcement learning

[DDPG 여행하기](../../../06/23/ddpg/)
[DDPG Code](https://github.com/reinforcement-learning-kr/pg_travel)

[DDPG 여행하기](../../../06/23/ddpg/)
[DDPG Code](https://github.com/reinforcement-learning-kr/pg_travel)

<br><br>

# 5. \[NPG\] A natural policy gradient

[NPG 여행하기](../../../06/14/2018-06-15-npg/)
[NPG Code](https://github.com/reinforcement-learning-kr/pg_travel)

일반적으로 Policy Gradient 기법들은 앞에서 확일할 수 있듯이 Objective funtction, $\eta(\pi_\theta)$을 최대화하는 쪽으로 파라미터를 업데이트하면서 복잡한 비선형 함수을 최적화 합니다. 일반적으로 파라미터를 업데이트하는 방법은 Objective function를 각 파라미터들로 편미분(Gradient를 구하여)하여 그 변수들의 변화에 따른 Objective function의 변화를 따라 파라미터를 업데이트 합니다. 하지만 컴퓨터는 해석학적으로 목적함수를 최적화하는 방법을 따를 수 없어 수치학적으로 접근을 합니다. 수치학적으로 Objective function을 최대화하려면 대부분의 경우 반복적인 방법, $\theta_{k+1} = \theta_k + \bigtriangledown_\theta \eta(\theta)$,을 사용합니다.

해석학적으로는 $\bigtriangledown_\theta\eta(\theta)$를 편미분하여 직접 Objective function의 Gradient를 구할 수 있겠지만 수치학적으로는 매우 작은 크기를 가지는 $d\theta$에 대해 $\eta(\theta+d\theta)-\eta(\theta)$를 구함으로써 Gradient를 간접적으로 얻을 수 있습니다. 그리고 이것은 간단하게 파라미터들의 집합인 $\theta$의 Positive-Definite Matrix인 $G(\theta)$에 의해 $G^{-1}\eta(\theta)$로 구해질 수 있습니다.

하지만 여기서 Objective function은 매우 많은 파라미터로 구성되어 있으며 파라미터들은 매 스텝마다 업데이트되기 때문에 $G(\theta)$를 통해 얻은 Gradient는 매번 달라지며 파라미터들의 성능을 추정하는 정확한 Metric이 될 수 없습니다. 본 논문에서는 Fisher Information Matrix라는 것을 도입하여 파라미터의 업데이트 등에 따라 영향을 받지 않는 Metric을 구해내며 이를 통해 파라미터를 업데이트하는 것을 소개하고 있습니다.

[NPG 여행하기](../../../06/14/2018-06-15-npg/)
[NPG Code](https://github.com/reinforcement-learning-kr/pg_travel)

<br><br>

# 6. \[TRPO\] Trust region policy optimization

[TRPO 여행하기](blog link)
[TRPO Code](https://github.com/reinforcement-learning-kr/pg_travel)

PG기법이 각광을 받게 된 계기는 아마도 TRPO 때문이 아닌가 싶습니다. DQN이라는 인공지능이 아타리게임을 사람보다 잘 플레이했던 영상을 보고 충격을 받았던 사람들이 많았던 것처럼 TRPO가 난제로 여겨졌던 서기, 걷기, 뛰기 같은 로봇에 적용할 수 있는 continuous action을 잘 수행하는 것을 보고 많은 사람들이 놀라워했습니다. (솔직히 처음 아타리게임을 봤을 때 만큼은 아니지만...) 이것들은 DQN으로는 달성할 수 없었던 난제였습니다. 그렇지만 거인의 어깨 위에서 세상을 바라보았던 많은 과학자들이 그랬던 것처럼 Schulmann도 TRPO를 갑자기 생각낸 것은 아닙니다. policy를 업데이트할 때 기존의 policy와 너무 많이 달라지면 문제가 생깁니다. 일단, state distribution이 변합니다. state distribution은 간단히 얘기하면 각각의 state를 방문하는 빈도를 의미한다고 할 수 있습니다. policy가 많이 바뀌면 기존의 state distribution을 그대로 이용할 수 없고 이렇게 되면 policy gradient를 구하기 어려워집니다. 그래서 착안한 것이 policy의 변화 정도를 제한하는 것입니다. 여기서 우리는 stochastic policy, 즉, 확률적 policy를 이용한다고 전제하고 있기 때문에, policy의 변화는 policy 확률 분포의 변화를 의미합니다. 현재 policy와 새로운 policy 사이의 확률 분포의 변화를 어떻게 측정할 수 있을까요? Schulmann은 [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)라는 척도를 이용하였습니다. 이 변화량을 특정값 이하로 제한하고 이것을 만족하는 영역을 trust region이라고 불렀습니다. 그렇지만 문제는 풀기에는 복잡했습니다. 이것을 풀기 위해서 여러 번의 근사화를 적용시켰습니다. objective function에는 linear approximation을 적용하고 trust region constraint에는 second-order approximation을 적용하면 문제가 좀 더 풀기 쉬운 형태로 바뀐다는 것을 알아냈습니다. 이것은 사실 natural gradient를 이용하는 것이었습니다. 이러한 이유로 TRPO를 이해하기 위해서 우리는 natural gradient도 살펴보았던 것입니다.

[TRPO 여행하기](blog link)
[TRPO Code](https://github.com/reinforcement-learning-kr/pg_travel)

<br><br>

# 7. \[GAE\] High-Dimensional Continuous Control Using Generalized Advantage Estimation

[GAE 여행하기](blog link)
[GAE Code](https://github.com/reinforcement-learning-kr/pg_travel)

TRPO가 나오고 난 뒤로도 복잡하고 어려운 control problem에서 Reinforcement Learning(RL)은 high sample complexity 때문에 제한이 되어왔습니다. 따라서 이 논문에서 그 제한을 풀고자 advantage function의 good estimate를 얻는 "variance reduction"에 대해 연구하였습니다.

"Generalized Advantage Estimator(GAE)"라는 것을 제안했고, 이것은 bias-variance tradeoff를 조절하는 두 개의 parameter $\gamma,\lambda$를 가집니다.
또한 어떻게 Trust Region Policy Optimization과 value function을 optimize하는 Trust Region Algorithm의 idea를 합치는 지를 보였습니다.

이렇게 함으로써 보다 더 복잡하고 어려운 control task들을 해결할 수 있었습니다.

GAE의 실험적인 입증으로는 robotic locomotion을 simulation하는 domain입니다. 실험에서도 보여준 것처럼 [0.9, 0.99]의 범위에서 $\lambda$의 적절한 중간의 값을 통해 best performance를 얻습니다. 좀 더 나아가 연구되어야할 점은 adaptive or automatic하도록 estimator parameter $\gamma,\lambda$를 조절하는 방법입니다.

추가적으로 앞으로 연구되어야할 부분은 만약 Value function estimation error와 Policy gradient estimation error 사이의 관계를 알아낸다면, Value function fitting에 더 잘 맞는 error metric(policy gradient estimation의 정확성과 더 잘 맞는 value function)을 사용할 수 있을 것입니다. 여기서 Policy와 Value function의 파라미터를 공유하는 모델을 만드는 것은 아주 흥미롭고 이점이 많습니다. 하지만 수렴을 보장하도록 적절한 numerical optimization을 제시해야 할 것입니다.

[GAE 여행하기](blog link)
[GAE Code](https://github.com/reinforcement-learning-kr/pg_travel)

<br><br>

# 8. \[PPO\] Proximal policy optimization algorithms

[PPO 여행하기](blog link)
[PPO Code](https://github.com/reinforcement-learning-kr/pg_travel)

PPO는 TRPO의 연장선상에 있는 알고리즘이라고 할 수 있습니다. 사실 Schulmann은 TRPO 논문을 쓸 당시 이미 PPO를 구상하고 있었던 것 같습니다. TRPO 논문에도 PPO와 관련있는 내용이 좀 나옵니다. 아이디어 자체는 간단합니다. 그래서인지 PPO는 arxiv에만 발표되었고 논문도 비교적 짧습니다. TRPO에서는 문제를 단순하게 만들기 위해서 최적화 문제를 여러 번 변형시켰습니다. PPO는 단순하게 clip이라는 개념을 사용합니다. TRPO에서 이용했던 surrogate objective function을 reward가 특정값 이상이거나 이하가 될 때 더 이상 변화시키지 않는 것입니다.

[PPO 여행하기](blog link)
[PPO Code](https://github.com/reinforcement-learning-kr/pg_travel)