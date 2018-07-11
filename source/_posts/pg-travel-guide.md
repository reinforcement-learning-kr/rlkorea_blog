---
title: pg-travel-guide
date: 2018-07-11 05:34:26
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 공민서, 김동민, 양혁렬, 이동민, 이웅원, 장수영, 차금강
subtitle: 피지여행에 관한 개략적 기록
---

---
# 1. Policy Gradient의 세계로

반갑습니다! 저희는 PG여행을 위해 모인 PG탐험대입니다. 강화학습하면 보통 Q-learning을 떠올립니다. 그렇지만 오래전부터 Policy Gradient라는 Q-learning 못지 않은 훌륭한 방법론이 연구되어 왔고, 최근에는 강화학습의 최정점의 기술로 자리매김하고 있습니다. 강화학습의 아버지인 Sutton의 논문을 필두로 하여 DQN 연구자들이 제안한 DQN보다 뛰어난 성능을 내는 DPG와 DDPG, 그리고 현재 가장 주목받는 강화학습 연구자인 John Schulmann의 TRPO, GAE, PPO와 이를 이해하기 위해 필요한 Natural Policy Gardient까지 더불어 살펴보고자 합니다.

이 블로그에는 각각의 기술을 제안한 논문을 PG탐험대분들이 자세하게 리뷰한 포스트들이 있습니다. 우리나라에서 PG에 대해서 이렇게 자세하게 리뷰한 포스트들은 없었다고 감히 말씀드리고 싶습니다. 본 글에서는 이 포스트들을 읽기 전에 전체 내용을 개략적으로 소개하고 각각의 포스트들로 안내하고자 합니다. 자, 저희와 함께 PG여행을 즐겨보시겠습니까?

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

[GAE 여행하기](blog link)
[GAE Code](https://github.com/reinforcement-learning-kr/pg_travel)

<br><br>
# 8. \[PPO\] Proximal policy optimization algorithms
[PPO 여행하기](blog link)
[PPO Code](https://github.com/reinforcement-learning-kr/pg_travel)

PPO는 TRPO의 연장선상에 있는 기술이라고 할 수 있습니다. 사실 Schulmann은 TRPO 논문을 쓸 당시 이미 PPO를 구상하고 있었던 것 같습니다. TRPO 논문에도 PPO와 관련있는 내용이 좀 나옵니다. 아이디어 자체는 간단합니다. 그래서그런지 PPO는 arxiv에만 발표되었고 논문도 비교적 짧습니다. TRPO에서는 문제를 단순하게 만들기 위해서 최적화 문제를 여러 번 변형시켰습니다. PPO는 clip이라는 개념을 사용합니다. TRPO에서 이용했던 surrogate objective function을 reward가 특정값 이상이거나 이하가 될 때 더 이상 변화시키지 않는 것입니다.

[PPO 여행하기](blog link)
[PPO Code](https://github.com/reinforcement-learning-kr/pg_travel)

