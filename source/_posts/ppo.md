---
title: Proximal Policy Optimization
date: 2018-07-12 16:53:12
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 장수영, 차금강
subtitle: 피지여행 6번째 논문
---

<center> <img src="https://www.dropbox.com/s/145van5kldfvvd5/Screen%20Shot%202018-07-18%20at%201.19.30%20AM.png?dl=1" width="700"> </center>

논문 저자 : John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
논문 링크 : https://arxiv.org/pdf/1707.06347.pdf
Proceeding : ??
정리 : 장수영, 차금강

---
# 1. 왜 Proximal Policy Optimization(이하 PPO)인가?

이전의 Trust Region Policy Optimization(이하 TRPO)의 핵심적인 부분을 살펴보자면 다음과 같습니다.
$$
maximize_\theta \hat{E}_t [\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)}\hat{A}_t]-\hat{E}_t[D_{KL}(\pi_{\theta old}(*|s)||\pi_\theta(*|s))]
$$
위의 식을 살펴보면 KL-Divergence를 구하는 과정에서 굉장한 연산량이 필요합니다.
$$
D_{KL}(\pi_{\theta old}(*|s)||\pi_\theta(*|s)) = -\int{\pi_{\theta old}(*|s)ln\pi_\theta(*|s)}+\int{\pi_\theta(*|s)ln\pi_{\theta old}(*|s)}
$$
특히 위의 식에서 볼 수 있듯이 $ln()$을 계산하는데 있어 많은 계산량이 필요로 하며 매번 파라미터들을 업데이트할 때마다 이 복잡한 연산을 반복적으로 수행해야 합니다. TRPO에서 KL-Divergence의 역할을 쉽게 풀이하자면 복잡한 비선형 최적화에서 조금의 파라미터 변화에도 성능이 크게 변화할 수 있는데 이를 방지하며 기존의 성능을 잘 보존할 수 있게 하는 장치입니다. 꼭 TRPO만이 아니라 일반적인 Policy Gradient 방법에서도 Objective Function은 $ln()$으로 되어 있으며 이를 어떤 연산량이 적은 방법으로 쉽게 적용할 수 있을까에 대해 고민하는 것이 PPO의 핵심입니다.

PPO 논문에서는 TRPO에서 제시한 surrogate function을 두 가지로 나누어서 접근하며 이를 서로, 그리고 TRPO와 비교 분석하고 있습니다. 본 논문에서 제시하고 있는 방법 두 가지 중 첫번째는 surrogate function을 매우 작은 값으로 clipping함으로써 gradient 자체를 강제적으로 적게 만들어주는 방법이며 두번째는 TRPO의 KL-Divergence의 크기에 따라 Adaptive한 파라미터를 적용하는 방법입니다.

2\. Clipped Surrogate Objective

2장에서는 TRPO의 surrogate obejctive function을 강제적으로 clipping해버리는 방법을 제시하고 있습니다. 기존의 TRPO의 surrogate function을 다음과 같이 표현합니다. $r_t(\theta)=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)}, r_t(\theta old) = 1$

위의 notation을 이용하여 TRPO의 surrogate object를 최대화합니다.
$$
L^{CLI}(\theta) =\hat{E}_t [\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)}\hat{A}_t]=\hat{E}_t[r_t(\theta)\hat{A}_t]
$$
위의 식을 그대로 사용한다면 매우 큰 policy update를 야기할 수 있기 때문에 이를 penalty를 이용하여 필요 이상의 policy update를 방지합니다. TRPO에서는 KL-Divergence를 이용하여 penalty를 구현하지만 PPO에서는 연산량적으로 효율적인 penalty를 적용하기 위해 아래와 같은 clipping 기법을 통해 구현합니다.
$$
L^{CLIP}(\theta) = \hat{E}_t [min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]
$$
이를 논문에서는 두 가지의 그래프($\hat{A}_t$의 부호에 따라)로 설명하고 있습니다.  $\hat{A}_t$가 양수일 때는 Advantage가 현재보다 높다라는 뜻이며 파라미터를 +의 방향으로 업데이트 하여야 합니다. $\hat{A}_t$가 음수일 때는 Advantage가 현재보다 좋지 않다라는 뜻이며 그의 반대 방향으로 업데이트 하여야 합니다. $r_t(\theta)$은 확률을 뜻하는 두개의 함수를 분자 분모로 가지고 있으며 분수로 구성되어 있기 때문에 무조건 양수로 이루어져 있으며 Advantage function인 $\hat{A}_t$와 곱해져 Objective function인 $L^{CLIP}$은 Advantage function과 방향이 같아집니다.

3\. Adaptive KL Penalty Coefficient

2절에서 설명한 Clipped Surrogate Objective 방법과 달리 기존의 TRPO의 방법에 Adaptive한 파라미터를 적용한 방법 또한 제시하고 설명하고 있습니다.

기존의 TRPO의 Objective function에 $\beta$를 적용하여 다음과 같이 $\beta$를 Adaptive하게 조절하고 있습니다.
$$
L^{KLPEN}(\theta) = \hat{E}_t [\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)}\hat{A}_t-\beta D_{KL}(\pi_{\theta old}(*|s)||\pi_\theta(*|s))]
$$
$d = \hat{E}_t[ D_{KL}(\pi_{\theta old}(*|s)||\pi_\theta(*|s))]$, 

$d<d_{targ}/1.5, \beta \xleftarrow{}\beta/2$ 

$ d>d_{targ}, \beta \xleftarrow{}\beta\times2$

$\beta$를 조절하는 방법을 해석하자면 다음과 같습니다. 만약 $\theta old$와 $\theta$간의 파라미터 차이가 크다면 penalty를 강하게 부여해 적은 변화를 야기하며, 파라미터 차이가 적다면 penalty를 완화시켜 주어 더 큰 변화를 야기하는 것을 위의 수식으로 표현할 수 있습니다.

4\. Algorithm

2절, 3절의 내용을 보면 policy만을 어떻게 업데이트 하는지에 대해 설명하고 있습니다. 4절에서는 다른 알고리즘과 같이 value, entropy-exploration과 합쳐 어떻게 통합적으로 업데이트 하는지에 대해 설명하고 있습니다.

본 논문에서 설명하기를 state-value function $V(s)$는 [generalized advantage estimation](https://arxiv.org/abs/1506.02438)를 선택하여 사용하고 있으며 일반적인 방식인 Mean-square error를 통해 업데이트하고 있습니다. 이에 대한 Objective function은 본 논문 내에서 $L^{VF}$로 표현하고 있습니다.

또한 일반적인 policy gradient 방법에서 [exploration]("Asynchronous methods for deep reinforcement learning" )의 기능을 추가하기 위해 다음과 같은 항, $S[\pi_\theta(s_t)]=\pi_\theta(a|s_t)log\pi_\theta(a|s_t)$, 을 추가합니다. 

2절, 3절에서 제시한 Objective function을 $L^{PPO}$라고 표현한다면($L^{PPO}$는 $L^{KLPEN}$, $L^{CLIP}$일 수 있습니다.) PPO에서 제시하는 통합적으로 최대화해야하는 함수는 다음과 같이 표현할 수 있습니다.
$$
L^{PPO+VF+S}(\theta) = \hat{E}_t[L^{PPO}(\theta)-c_1L^{VF}(\theta)+c_2S[\pi_\theta(s_t)]
$$
위의 Objective function을 최대화하면 Reinforcement Learning을 통한 policy 학습, state-value function의 학습, exploration을 할 수 있습니다.

5\. Experimnet

5.1\. Surrogate Objectives의 비교

5.1절에서는 3가지 Surrogate Objectives를 비교분석 하고 있습니다.

* No clipping or penalty: $L_t(\theta) = r_t(\theta)\hat{A}_t$
* Clipping: $L_t(\theta) = min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon,1+\epsilon)\hat{A}_t)$
* KL penalty(fixed or adaptive): $L_t(\theta)=r_t(\theta)\hat{A}_t-\beta KL[\pi_{\theta old}, \pi_\theta]$

그리고 환경은 MuJoCo 엔진을 사용하며 OpenAI Gym 환경에서 실험을 진행하였습니다. 그리고 네트워크들은 2개의 hidden layer를 가지며 각각 64개의 unit들을 가지고 있습니다. 그리고 tanh를 활성화 함수로써 사용하고 있으며 네트워크는 Gaussian distribution을 출력하는 것으로 구성되어 있습니다. 그리고 Policy Network와 Value Network는 파라미터들을 공유하고 있지 않으며 위에서 설명한 Exploration을 위한 entropy는 사용하지 않습니다.

7개의 환경에서 실험하였으며 각 환경들은 HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPerdulum, Reacher, Swimmer, Walker2d를 사용하였으며 모든 환경들은 v1 버전을 사용하였습니다.

| Algorithm                      | avg. normalized score |
| ------------------------------ | --------------------- |
| No clipping or penalty         | -0.39                 |
| Clipping, $\epsilon=0.1$       | 0.76                  |
| Clipping, $\epsilon=0.2$       | 0.82                  |
| Clipping, $\epsilon=0.3$       | 0.70                  |
| Adaptive KL $d_{targ}=0.003$   | 0.68                  |
| Adaptive KL $d_{targ} = 0.001$ | 0.74                  |
| Adaptive KL  $d_{targ} = 0.03$ | 0.71                  |
| Fixed KL, $\beta=0.3$          | 0.62                  |
| Fixed KL, $\beta=1$            | 0.71                  |
| Fixed KL, $\beta=3$            | 0.72                  |
| Fixed KL, $\beta=10$           | 0.69                  |

위의 표는 실험 결과입니다. 세 가지 Surrogate Objective에 대해 각 파라미터들을 테이블에 표기된 방식대로 변화하며 실험하였습니다. 

5.2\. Comparison to other Algorithms in the Continuous Domain

5.2절에서는 Clipping 버전의 PPO와 다른 알고리즘들 간의 비교를 연속적인 Action을 가지는 환경에서 실험을 한 결과를 보여주고 있습니다. 다른 알고리즘은 TRPO, cross-entropy method, vanilla policy gradient with adaptive stepsize, A2C, A2C with trust region를 사용하였습니다. 환경과 사용한 네트워크의 구성은 5.1절에서 사용한 것과 동일합니다.

여타 다른 알고리즘보다 좋은 성능을 보이는 것을 알 수 있습니다.

5.3\. Showcase in the Continuous Domain: Humanoid Running and Steering

5.3절에서는 Humanoid-v0, HumanoidFlagrun-v0, HumanoidFlagrunHarder-v0의 환경에서 Roboschool엔진을 사용하여 실험하였습니다. Humanoid-v0는 단순히 앞으로 걸어나가는 환경, HumanoidFlagrun-v0은 200스텝마다 혹은 목적지에 도착할 때 마다 위치가 바뀌는 목적지에 걸어서 도달하는 환경입니다. HumanoidFlagrunHarder-v0 환경은 환경이 초기화될 때 마다 특정 경계안에 Humanoid가 위치하게 되며 특정 영역 밖으로 걸어나가는 것을 수행하는 환경입니다.

5.4\. Comparison to Other Algorithms on the Atari Domain

5.4절에서는 Atari Domain에서 PPO, A2C, ACER(Actor Critic with Experience Replay)의 3가지 알고리즘을 비교합니다.

|                                                | A2C  | ACER | PPO  | Tie  |
| ---------------------------------------------- | ---- | ---- | ---- | ---- |
| (1) avg. episode reward over all of training   | 1    | 18   | 30   | 0    |
| (2) avg. episode reward over last 100 episodes | 1    | 28   | 19   | 1    |

모든 학습하는 도중에는 PPO가 ACER보다 좋은 성능을 나타내지만 마지막 100 에피소드만을 비교했을 때는 PPO보다 ACER이 더 좋은 성능을 나타냅니다. 이는 PPO가 더 빨리 최종 능력에 도달하지만 ACER이 가진 잠재력이 더 높다는 것을 나타냅니다.

6\. Conclusion

본 논문에서는 TRPO를 컴퓨터 친화적?으로 알고리즘을 수정하고 다른 알고리즘들과의 성능비교를 통해 PPO가 좋다라는 것을 보여주고 있습니다.