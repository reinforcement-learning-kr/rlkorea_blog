---
title: PG Travel Guide
date: 2018-06-29 01:11:26
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 김동민, 이동민, 차금강
subtitle: 피지여행에 관한 개략적 기록
---

---

# 0. Policy Gradient의 세계로

반갑습니다! 저희는 PG여행을 위해 모인 PG탐험대입니다. 강화학습하면 보통 Q-learning을 떠올립니다. 그렇지만 오래전부터 Policy Gradient라는 Q-learning 못지 않은 훌륭한 방법론이 연구되어 왔고, 최근에는 강화학습의 최정점의 기술로 자리매김하고 있습니다. 강화학습의 아버지인 Sutton의 논문을 필두로 하여 기존의 DQN보다 뛰어난 성능을 내는 DPG와 DDPG, 그리고 현재 가장 주목받는 강화학습 연구자인 John Schulmann의 TRPO, GAE, PPO와 이를 이해하기 위해 필요한 Natural Policy Gardient까지 더불어 살펴보고자 합니다.

<center> <img src="https://www.dropbox.com/s/tbcyhvilaqy4ra0/Policy%20Optimization%20in%20the%20RL%20Algorithm%20Landscape.png?dl=1" width="800"> </center>

위의 그림은 강화학습 알고리즘 landscape에서 Policy Optimization의 관점을 중점적으로 하여 나타낸 그림입니다. 위의 그림에서 빨간색 작은 숫자로 나타낸 것이 저희 PG여행에서 다룰 논문들입니다. 순서는 다음과 같습니다.

1. [Sutton_PG](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
2. [DPG](http://proceedings.mlr.press/v32/silver14.pdf)
3. [DDPG](https://arxiv.org/pdf/1509.02971.pdf)
4. [NPG](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)
5. [TRPO](https://arxiv.org/pdf/1502.05477.pdf)
6. [GAE](https://arxiv.org/pdf/1506.02438.pdf)
7. [PPO](https://arxiv.org/pdf/1707.06347.pdf)

위와 같이 총 7가지 논문들을 리뷰하여 블로그로 정리하였습니다. 각 순서에 맞춰 보시는 것을 권장해드립니다.

이 블로그에는 각각의 기술을 제안한 논문을 PG탐험대분들이 자세하게 리뷰한 포스트들이 있습니다. 우리나라에서 PG에 대해서 이렇게 자세하게 리뷰한 포스트들은 없었다고 감히 말씀드리고 싶습니다. 본 글에서는 이 포스트들을 읽기 전에 전체 내용을 개략적으로 소개하고 각각의 포스트들로 안내하고자 합니다. 자, 저희와 함께 PG여행을 즐겨보시겠습니까?

<br><br>

# 1. \[Sutton PG\] Policy gradient methods for reinforcement learning with function approximation

[Sutton PG 여행하기](https://reinforcement-learning-kr.github.io/2018/06/28/1_sutton-pg/)

policy gradient (PG)는 expected reward를 policy의 파라미터에 대한 함수로 모델링하고 이 reward를 최대화하는 policy를 gradient ascent 기법을 이용해서 찾는 기법입니다. 강화학습의 대표격이라고 할 수 있는 Q-learning이라는 훌륭한 방법론이 이미 존재하고 있었지만 Q값의 작은 변화에도 policy가 크게 변할 수도 있다는 단점이 있기 때문에 policy의 점진적인 변화를 통해 더 나은 policy를 찾아가는 PG기법이 개발되었습니다.

이 PG기법은 먼저 개발되었던 [REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)라는 기법과 관련이 아주 많습니다. 서튼의 PG기법은 REINFORCE 기법을 [actor-critic algorithm](http://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)을 사용하여 개선시킨 것이라고 볼 수도 있습니다. 저희 PG여행 팀도 처음에는 REINFORCE를 출발지로 삼으려고 했었지만 예전 논문이다보니 논문의 가독성이 너무 떨어져서 강화학습의 아버지라고 할 수 있는 서튼의 논문을 출발지로 삼았습니다. 하지만 이 논문도 만만치 않았습니다. 이 논문을 읽으시려는 분들께 저희의 여행기가 도움이 될 것입니다. PG기법에 대해서 먼저 감을 잡고 시작하시려면 [Andre Karpathy의 PG에 대한 블로그](http://karpathy.github.io/2016/05/31/rl/)를 먼저 한 번 읽어보세요.  한글번역도 있습니다! 1) http://keunwoochoi.blogspot.com/2016/06/andrej-karpathy.html 2) https://tensorflow.blog/2016/07/13/reinforce-pong-at-gym/

[Sutton PG 여행하기](https://reinforcement-learning-kr.github.io/2018/06/28/1_sutton-pg/)

<br><br>

# 2. \[DPG\] Deterministic policy gradient algorithms

[DPG 여행하기](https://reinforcement-learning-kr.github.io/2018/06/27/2_dpg/)

deterministic policy gradeint (DPG)는 어찌보면 상당히 도전적인 아이디어였던 것 같습니다. Sutton PG 논문에서는 DPG 스타일의 기법이 가진 단점에 대해서 언급하면서 stochastic policy gradient (SPG)를 써야 optimal을 찾을 수 있다고 기술하고 있었기 때문입니다.

그런데 이 논문에서 높은 차원의 action space를 가지는 문제들에(예를 들면 문어발 제어) 대해서는 DPG가 상당히 좋은 성능을 내는 것을 저자들이 보였습니다. 그리고 DPG는 SPG와 대척점에 있는 기술이 아니고 SPG의 special case 중 하나임을 증명하면서 SPG를 가정하고 만들어진 기술들을 DPG에서도 그대로 이용할 수 있음을 보였습니다. David Silver의 [동영상 강의](http://techtalks.tv/talks/deterministic-policy-gradient-algorithms/61098/)를 한 번 보시길 추천드립니다. 짧은 강의지만 랩을 하듯이 쉴새없이 설명하는 Silver의 모습에서 천재성이 엿보이는 것을 확인하실 수 있습니다. 

[DPG 여행하기](https://reinforcement-learning-kr.github.io/2018/06/27/2_dpg/)

<br><br>

# 3. \[DDPG\] Continuous control with deep reinforcement learning

[DDPG 여행하기](https://reinforcement-learning-kr.github.io/2018/06/26/3_ddpg/)

DPG의 후속 연구로 DPG보다 더 큰 주목을 받은 논문입니다. 소위 말하는 deep deterministic policy gradient (DDPG)로 불리는 기술을 제안한 논문입니다. 이 논문의 저자 중 일부는 그 유명한 DQN 논문의 저자이기도 합니다. Q-learning과 deep neural network를 접목시켰던 DQN처럼 이 논문도 DPG와 deep neural network를 접목시킨 논문입니다.

이 논문은 DQN으로는 좋은 성능을 내지 못했던 continuous action을 가지는 상황들에 대해서 상당히 훌륭한 결과를 보이면서 큰 주목을 받았습니다. 소위 말하는 deep reinforcement learning (DRL)에서 Q-learning 계열의 DQN, PG 계열의 DDPG로 양대산맥을 이루는 논문이라고 할 수 있습니다. 두 논문 모두 Deepmind에서 나왔다는 것은 Deepmind 기술력이 DRL 분야에서 최정점에 있음을 보여주는 상징이 아닌가 싶습니다. 논문 자체는 그리 어렵지 않습니다. 새로운 아이디어를 제시했다기보다는 딥러닝을 활용한 강화학습의 가능성을 보여주는 논문이라는 점에서 큰 의의를 가지는 것 같습니다. 여러분도 한번 코딩에 도전해보시는게 어떨까요?

[DDPG 여행하기](https://reinforcement-learning-kr.github.io/2018/06/26/3_ddpg/)

<br><br>

# 4. \[NPG\] A natural policy gradient

[NPG 여행하기](https://reinforcement-learning-kr.github.io/2018/06/25/4_npg/)
[NPG Code](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/tnpg.py)

이 논문은 뒤이어 나오는 TRPO를 더 잘 이해하기 위해서 보는 논문입니다. 이번 논문부터 내용이 상당히 어려워집니다. 다소 생소한 수학 개념들이 많이 나오기 때문입니다. 하지만 이 블로그를 보시면 많은 부분들이 채워질 것이라고 믿습니다.

2002년 당시에도 많은 연구자들이 objective function의 gradient 값을 따라서 좋은 policy $\pi$를 찾고자 하였습니다. 하지만 기존의 우리가 알던 gradient descent method는 steepest descent direction이 아닐 수 있기 때문에(쉽게 말해 가장 가파른 방향을 따라서 내려가야 하는데 그러지 못할 수도 있다는 것입니다.) 이 논문에서 steepest descent direction를 나타내는 natural gradient method를 policy gradient에 적용하여 좋은 policy $\pi$를 찾습니다. gradient descent는 parameter를 한 번에 많이 update 할 수 없는 반면, natural gradient는 가장 좋은 action을 고르도록 학습이 됩니다.

또한 natural gradient method는 Line search 기법과 함께 사용하면 더 Policy Iteration과 비슷해집니다. Greedy Policy Iteration에서와 비교하면 Performance Improvement가 보장됩니다. 하지만 Fisher Information Matrix(FIM)이 asymtotically Hessian으로 수렴하지 않습니다. 그렇기 때문에 Conjugate Gradient Method로 구하는 방법이 더 좋을수 있습니다.

위의 요약한 문장들만 봤을 때는 생소한 용어들이 많이 나와서 무슨 말인지 감이 안잡히실 수 있습니다. 저희가 포스팅한 블로그 글에는 다음과 같은 추가적인 내용이 나옵니다. 반드시 알고 가야 TRPO를 이해하는 것은 아닙니다. 다만 NPG를 이해하면 할 수록 TRPO를 접하기가 더 쉬울 수 있습니다.

- Euclidean space와 Riemannian space의 차이
- Natural Gradient 증명
- Fisher Information Matrix(FIM)
- Line Search
- FIM과 Hessian 방법의 차이
- Conjugate Gradient Method

아래의 NPG Code는 Hessian 방법이 아닌 Conjugate Gradient Method를 사용한 "Truncated Natural Policy Gradient(TNPG)"라고 하는 방법의 코드입니다.

마지막으로 프로젝트 내에 있는 한 팀원의 문장을 인용하겠습니다. "머리가 아프고 힘들수록 잘하고 있는겁니다." NPG 논문을 보시는 분들 화이팅입니다!

[NPG 여행하기](https://reinforcement-learning-kr.github.io/2018/06/25/4_npg/)
[NPG Code](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/tnpg.py)

<br><br>

# 5. \[TRPO\] Trust region policy optimization

[TRPO 여행하기](https://reinforcement-learning-kr.github.io/2018/06/24/5_trpo/)
[TRPO Code](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/trpo_gae.py)

PG기법이 각광을 받게 된 계기는 아마도 TRPO 때문이 아닌가 싶습니다. DQN이라는 인공지능이 아타리게임을 사람보다 잘 플레이했던 영상을 보고 충격을 받았던 사람들이 많았던 것처럼 TRPO가 난제로 여겨졌던 서기, 걷기, 뛰기 같은 로봇에 적용할 수 있는 continuous action을 잘 수행하는 것을 보고 많은 사람들이 놀라워했습니다. (솔직히 처음 아타리게임을 봤을 때 만큼은 아니지만...) 이것들은 DQN으로는 달성할 수 없었던 난제였습니다. 

그렇지만 거인의 어깨 위에서 세상을 바라보았던 많은 과학자들이 그랬던 것처럼 Schulmann도 TRPO를 갑자기 생각낸 것은 아닙니다. policy를 업데이트할 때 기존의 policy와 너무 많이 달라지면 문제가 생깁니다. 일단, state distribution이 변합니다. state distribution은 간단히 얘기하면 각각의 state를 방문하는 빈도를 의미한다고 할 수 있습니다. policy가 많이 바뀌면 기존의 state distribution을 그대로 이용할 수 없고 이렇게 되면 policy gradient를 구하기 어려워집니다. 

그래서 착안한 것이 policy의 변화 정도를 제한하는 것입니다. 여기서 우리는 stochastic policy, 즉, 확률적 policy를 이용한다고 전제하고 있기 때문에, policy의 변화는 policy 확률 분포의 변화를 의미합니다. 현재 policy와 새로운 policy 사이의 확률 분포의 변화를 어떻게 측정할 수 있을까요? Schulmann은 [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)라는 척도를 이용하였습니다. 이 변화량을 특정값 이하로 제한하고 이것을 만족하는 영역을 trust region이라고 불렀습니다. 

그렇지만 문제는 풀기에는 복잡했습니다. 이것을 풀기 위해서 여러 번의 근사화를 적용시켰습니다. objective function에는 linear approximation을 적용하고 trust region constraint에는 second-order approximation을 적용하면 문제가 좀 더 풀기 쉬운 형태로 바뀐다는 것을 알아냈습니다. 이것은 사실 natural gradient를 이용하는 것이었습니다. 이러한 이유로 TRPO를 이해하기 위해서 natural gradient도 살펴보았던 것입니다. 

Schulmann이 너무 똑똑해서 일까요? 훌륭한 아이디어로 policy gradient 기법의 르네상스를 열은 Schulmann의 역작인 TRPO 논문은 이해하기 쉽게 쓰여지지 않은 것 같습니다. (더 잘 쓸 수 있었잖아 Schulmann...) 저희의 포스트와 함께 보다 편하게 여행하시길 바랍니다. 이 [유투브 영상](https://youtu.be/CKaN5PgkSBc)도 무조건 보세요~

[TRPO 여행하기](https://reinforcement-learning-kr.github.io/2018/06/24/5_trpo/)
[TRPO Code](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/trpo_gae.py)

<br><br>

# 6. \[GAE\] High-Dimensional Continuous Control Using Generalized Advantage Estimation

[GAE 여행하기](https://reinforcement-learning-kr.github.io/2018/06/23/6_gae/)

TRPO가 나오고 난 뒤로도 복잡하고 어려운 control problem에서 Reinforcement Learning (RL)은 high sample complexity 때문에 제한이 되어왔습니다. 따라서 이 논문에서 그 제한을 풀고자 advantage function의 good estimate를 얻는 "variance reduction"에 대해 연구하였습니다.

"Generalized Advantage Estimator (GAE)"라는 것을 제안했고, 이것은 bias-variance tradeoff를 조절하는 두 개의 parameter $\gamma,\lambda$를 가집니다.
또한 어떻게 Trust Region Policy Optimization과 value function을 optimize하는 Trust Region Algorithm의 idea를 합치는지를 보였습니다.

이렇게 함으로써 보다 더 복잡하고 어려운 control task들을 해결할 수 있었습니다.

GAE의 실험적인 입증으로는 robotic locomotion을 simulation하는 domain입니다. 실험에서도 보여준 것처럼 [0.9, 0.99]의 범위에서 $\lambda$의 적절한 중간의 값을 통해 best performance를 얻습니다. 좀 더 나아가 연구되어야할 점은 adaptive or automatic하도록 estimator parameter $\gamma,\lambda$를 조절하는 방법입니다.

추가적으로 앞으로 연구되어야할 부분은 만약 value function estimation error와 policy gradient estimation error 사이의 관계를 알아낸다면, value function fitting에 더 잘 맞는 error metric(policy gradient estimation의 정확성과 더 잘 맞는 value function)을 사용할 수 있을 것입니다. 여기서 policy와 value function의 파라미터를 공유하는 모델을 만드는 것은 아주 흥미롭고 이점이 많습니다. 하지만 수렴을 보장하도록 적절한 numerical optimization을 제시해야 할 것입니다.

[GAE 여행하기](https://reinforcement-learning-kr.github.io/2018/06/23/6_gae/)

<br><br>

# 7. \[PPO\] Proximal policy optimization algorithms

[PPO 여행하기](https://reinforcement-learning-kr.github.io/2018/06/22/7_ppo/)
[PPO Code](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/ppo_gae.py)

이 논문에서는 Reinforcement Learning에서 Policy Gradient Method의 새로운 방법인 PPO를 제안합니다. 이 방법은 agent가 환경과의 상호작용을 통해 data를 sampling하는 것과 stochastic gradient ascent를 사용하여 "surrogate" objective function을 optimizing하는 것을 번갈아가면서 하는 방법입니다. data sample마다 one gradient update를 수행하는 기존의 방법과는 달리, minibatch update의 multiple epochs를 가능하게 하는 새로운 objective function을 말합니다.

또한 PPO는 TRPO의 연장선상에 있는 알고리즘이라고 할 수 있습니다. TRPO에서는 문제를 단순하게 만들기 위해서 최적화 문제를 여러 번 변형시켰지만, PPO는 단순하게 clip이라는 개념을 사용합니다. TRPO에서 이용했던 surrogate objective function을 reward가 특정값 이상이거나 이하가 될 때 더 이상 변화시키지 않는 것입니다.

이 알고리즘의 장점으로는

- TRPO의 장점만을 가집니다. 다시 말해 알고리즘으로 학습하기에 훨씬 더 간단하고, 더 일반적이고, 더 좋은 sample complexity (empirically)를 가집니다.
- 또한 다른 online policy gradient method들을 능가했고, 전반적으로 sample complexity(특히 computational complexity), simplicity, wall-time가 좋다고 합니다.

[PPO 여행하기](https://reinforcement-learning-kr.github.io/2018/06/22/7_ppo/)
[PPO Code](https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/ppo_gae.py)
