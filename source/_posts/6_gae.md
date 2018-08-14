---
title: High-Dimensional Continuous Control using Generalized Advantage Estimation
date: 2018-06-23 19:18:45
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 양혁렬, 이동민
subtitle: 피지여행 6번째 논문
---

<center> <img src="https://www.dropbox.com/s/p8gfpyo6xf9wm5w/Screen%20Shot%202018-07-18%20at%201.25.53%20AM.png?dl=1" width="700"> </center>

논문 저자 : John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan and Pieter Abbeel
논문 링크 : https://arxiv.org/pdf/1506.02438.pdf
Proceeding : International Conference of Learning Representations (ICLR) 2016
정리 : 양혁렬, 이동민

---

# 1. 들어가며...

현존하는 Policy Gradient Method들의 목적은 누적되는 reward들을 optimization하는 것입니다. 하지만 학습할 때에 많은 양의 sample이 필요로 하고, 들어오는 data가 nonstationarity임에도 불구하고 stable and steady improvement가 어렵습니다.

그래서 이 논문에서는 다음과 같은 방법을 제시합니다.

- TD($\lambda$)와 유사한 advantage function의 exponentially-weighted estimator를 사용하여 policy gradient estimate의 variance를 줄이는 것 
- policy와 value function에 대한 Trust Region Optimization 사용하는 것

3D locomotion tasks에 대한 empirical results는 다음과 같습니다.

- bipedal and quadrupedal simulated robots의 달리는 자세를 학습
- bipedal 사람이 땅에 누워있다가 일어서는 것을 학습

※ TD($\lambda$)를 혹시 모르실 경우도 있을 것 같아 간략하게 정리하였습니다. http://dongminlee.tistory.com/10 를 먼저 보시고 아래의 내용을 봐주시기 바랍니다!

<br><br>

# 2. Introduction

기본적으로 "parameterized stochastic policy"를 가정합니다. 이 때 expected total returns의 gradient에 대한 unbiased estimate를 얻을 수 있는데 이것을 REINFORCE라고 부릅니다. 하지만 하나의 action의 결과가 과거와 미래의 action의 결과로 혼동되기 때문에 gradient estimator의 high variance는 시간에 따라 scaling됩니다.

또 다른 방법은 Actor-Critic이 있습니다. 이 방법은 empirical returns보다 하나의 value function을 사용합니다. 또한 bias하고 lower variance를 가진 estimator입니다. 구체적으로 말하자면, high variance하다면 더 sampling을 하면 되는 반면에 bias는 매우 치명적입니다. 다시 말해 bias는 algorithm이 converge하는 데에 실패하거나 또는 local optimum이 아닌 poor solution에 converge하도록 만듭니다.

따라서 이 논문에서는 $\gamma\in [0,1]$ and $\lambda\in [0,1]$에 의해 parameterized estimation scheme인 Generalized Advantage Estimator(GAE)를 다룹니다.

이 논문을 대략적으로 요약하자면 다음과 같습니다.

- 효과적인 variance reduction scheme인 GAE를 다룹니다. 또한 실험할 때 batch trust region algorithm에 더하여 다양한 algorithm들에 적용됩니다.
- value function에 대해 trust region optimization method를 사용합니다. 이렇게 함으로서 더 robust하고 efficient한 방법이 됩니다.
- A와 B를 합쳐서, 실험적으로 control task에 neural network policies를 learning하는 데에 있어서 효과적인 algorithm을 얻습니다. 이러한 결과는 high-demensional continuous control에 RL을 사용함으로서 state of the art로 확장되었습니다.

<br><br>

# 3. Preliminaries

먼저 policy optimization의 "undiscounted formulation"을 가정합니다. (undiscounted formulation에 주목합시다.)

- initial state $s_0$는 distribution $\rho_0$으로부터 sampling된 것입니다.
- 하나의 trajectory ($s_0, a_0, s_1, a_1, ...$)는 terminal state에 도달될 때 까지 policy $a_t$ ~ $\pi(a_t | s_t)$에 따라서 action을 sampling하고, dynamics $s_{t+1}$ ~ $P(s_{t+1} | s_t, a_t)$에 따라서 state를 sampling함으로써 생성됩니다.
- reward $r_t = r(s_t, a_t, s_{t+1})$은 매 time step마다 받아집니다.
- 목표는 모든 policies에 대해 finite하다고 가정됨으로서 expected total reward $\sum_{t=0}^{\infty} r_t$를 maximize하는 것입니다.

여기서 중요한 점은 $\gamma$를 discount의 parameter로 사용하는 것이 아니라 "bias-variance tradeoff를 조절하는 parameter로 사용한다" 는 것입니다.

policy gradient method는 gradient $g := \nabla_\theta \mathbb{E} [\sum_{t=0}^\infty r_t]$를 반복적으로 estimate함으로써 expected total reward를 maximize하는 것인데, policy gradient에는 여러 다른 표현들이 있습니다.
$$g = \mathbb{E} [\sum_{t=0}^\infty \Phi_t \nabla_\theta \log \pi_\theta (a_t | s_t)]$$

여기서 $\Phi_t$는 아래 중의 하나일 수 있습니다.

1. $\sum_{t=0}^\infty r_t$: trajectory의 total reward
2. $\sum_{t'=t}^\infty r_t'$: action $a_t$ 후의 reward
3. $\sum_{t'=t}^\infty r_t' - b(s_t)$: 2의 baselined version
4. $Q^\pi (s_t, a_t)$: state-action value function
5. $A^\pi (s_t, a_t)$: advantage function
6. $r_t + V^\pi (s_{t+1}) - V^\pi (s_t)$: TD residual

위의 번호 중 4, 5, 6의 수식들은 다음의 정의를 사용합니다.

- $V^\pi (s_t) := \mathbb{E}_{s_{t+1}:\infty, a_t:\infty} [\sum_{l=0}^\infty r_{t+1}]$
- $Q^\pi (s_t, a_t) := \mathbb{E}_{s_{t+1}:\infty, a_{t+1}:\infty} [\sum_{l=0}^\infty r_{t+l}]$
- $A^\pi (s_t, a_t) := Q^\pi (s_t, a_t) - V^\pi (s_t)$, (Advantage function)

추가적으로 colon notation $a : b$는 포괄적인 범위 $(a, a+1, ... , b)$입니다. (잘 기억해둡시다. 뒤에 계속해서 colon notation이 나옵니다.)

여기서부터 parameter $\gamma$에 대해 좀 더 자세히 알아봅시다. parameter $\gamma$는 bias하면서 동시에 reward를 downweighting함으로써 variance를 줄입니다. 다시 말해 MDP의 discounted formulation에서 사용된 discounted factor와 일치하지만, 이 논문에서는 "$\gamma$를 undiscounted MDP에서 variance reduction parameter"로 다룹니다. -> 결과는 같지만, 의미와 의도가 다릅니다.

discounted value function들은 다음과 같습니다.

- $V^{\pi, \gamma} (s_t) := \mathbb{E}_{s_{t+1}:\infty, a_t:\infty} [\sum_{l=0}^\infty \gamma^l r_{t+l}]$ 
- $Q^\pi (s_t, a_t) := \mathbb{E}_{s_{t+1}:\infty, a_{t+1}:\infty} [\sum_{l=0}^\infty \gamma^l r_{t+l}]$
- $A^{\pi, \gamma} (s_t, a_t) := Q^{\pi, \gamma} (s_t, a_t) - V^{\pi, \gamma} (s_t)$

따라서 policy gradient에서의 discounted approximation은 다음과 같이 나타낼 수 있습니다.
$$g^\gamma := \mathbb{E}_{s_{0:\infty} a_{0:\infty}} [\sum_{t=0}^\infty A^{\pi, \gamma} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t | s_t)]$$
뒤이어 나오는 section에서는 위의 수식에서 $A^{\pi, \gamma}$에 대해 biased (but not too biased) estimator를 얻는 방법에 대해서 나옵니다.

다음으로 advantage function에서 새롭게 접하는 "$\gamma$-just estimator"의 notation에 대해 알아봅시다.

- 먼저 $\gamma$-just estimator는 $g^\gamma$ ${}^1$를 estimate하기 위해 위의 수식에서 $A^{\pi, \gamma}$를 $\gamma$-just estimator로 사용할 때, bias하지 않은 estimator라고 합시다. 그리고 이 $\gamma$-just advantage estimator를 $\hat{A}_t (s_{0:\infty} , a_{0:\infty})$라고 하고, 전체의 trajectory에 대한 하나의 function이라고 합시다.
- ${}^1$에서 이 논문의 저자가 하고 싶은 말이 있는 것 같습니다. 개인적으로 $\gamma$-just estimator를 이해하는 데에 있어서 중요한 주석이라 정확히 해석하고자 합니다.
    - $A^\pi$를 $A^{\pi, \gamma}$로 사용함으로써 이미 bias하다라고 말했지만, "이 논문에서 하고자 하는 것은 $g^\gamma$에 대해 unbiased estimate를 얻고 싶은 것입니다." 하지만 undiscounted MDP의 policy gradient에 대해서는 당연히 $\gamma$를 사용하기 때문에 biased estimate를 얻습니다. 개인적으로 이것은 일단 무시하고 $\gamma$를 사용할 때 어떻게 unbiased estimate를 얻을 지에 대해 좀 더 포커스를 맞추고 있는 것 같습니다.
    - 그러니까 저 $A^{\pi, \gamma}$를 $\gamma$-just estimator로 바꿔줌으로써 unbiased estimate를 하고 싶다는 것이 뒤이어 나오는 정의와 명제의 핵심이라고 할 수 있습니다.

Definition 1.
먼저 가정을 합니다. (가정을 바탕으로 이루어지는 정의라는 것을 주목합시다.) 만약
$$\mathbb{E}_{s_{0:\infty} a_{0:\infty}} [\hat{A}_t (s_{0:\infty}, a_{0:\infty}) \nabla_\theta \log \pi_\theta (a_t | s_t)] = \mathbb{E}_{s_{0:\infty} a_{0:\infty}} [A^{\pi, \gamma} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t | s_t)]$$
두 수식이 같다면, estimator $\hat{A}_t$는 $\gamma$-just입니다.

그리고 만약 모든 t에 대해서 $\hat{A}_t$이 $\gamma$-just이라면, 다음과 같이 표현할 수 있습니다.
$$\mathbb{E}_{s_{0:\infty} a_{0:\infty}} [\hat{A}_t (s_{0:\infty}, a_{0:\infty}) \nabla_\theta \log \pi_\theta (a_t | s_t)] = g^\gamma$$
위의 수식이 바로 unbiased estimate입니다.

$\gamma$-just인 $\hat{A}_t$에 대한 한가지 조건은 $\hat{A}_t$이 두 가지 function $Q_t$ and $b_t$로 나뉠 수 있다는 것입니다.

- $Q_t$는 $\gamma$-discounted Q-function의 unbiased estimator입니다.
- $b_t$는 action $a_t$전에 sampling된 states and actions의 arbitrary function이다. 

Proposition 1.
모든 $(s_t, a_t)$에 대해,
$$\mathbb{E}_{s_{t+1}:\infty, a_{t+1}:\infty | s_t, a_t} [Q_t (s_{t:\infty}, a_{t:\infty})] = Q^{\pi, \gamma} (s_t, a_t)$$
로 인하여 $\hat{A}_t$이 
$$\hat{A}_{s_{0:\infty}, a_{0:\infty}} = Q_t (s_{0:\infty}, a_{0:\infty}) - b_t (s_{0:t}, a_{0:t-1})$$
형태라고 가정합시다. (가정을 바탕으로 이루어지는 명제라는 점을 주목합시다.)

그 때, $\hat{A}_t$은 $\gamma$-just입니다.

이 명제에 대한 증명은 Appendix B에 있습니. 그리고 $\hat{A}_t$에 대한 $\gamma$-just advantage estimator들은 다음과 같습니다.

- $\sum_{l=0}^\infty \gamma^l r_{t+1}$
- $A^{\pi, \gamma} (s_t, a_t)$
- $Q^{\pi, \gamma} (s_t, a_t)$
- $r_t + \gamma V^{\pi, \gamma} (s_{t+1}) - V^{\pi, \gamma} (s_t)$

증명을 보기 전에 먼저 Advantage function에 대해서 먼저 살펴봅시다. 아래의 내용은 Sutton이 쓴 논문인 Policy Gradient Methods for Reinforcement Learning with Function Approximation(2000)에서 나온 내용을 리뷰한 것입니다. 

<center> <img src="https://www.dropbox.com/s/x737yq97ut6gp1a/Screen%20Shot%202018-07-15%20at%201.16.09%20PM.png?dl=1" width="600"> </center>

여기서는 "이 수식은 $\sum_a \frac{\partial \pi (s,a)}{\partial \theta} = 0$이기 때문에 가능해진다.", "즉, 이 수식은 $\pi(s,a)$의 gradient에만 dependent하기 때문에 advantage 역할을 하는 함수들을 넣어도 아무런 상관이 없다"라는 두 문장을 기억해둡시다.

이제 증명을 보면 됩니다.

<center> <img src="https://www.dropbox.com/s/e7sj2fwcm1f7hof/figure2.jpg?dl=1" width="600"> </center>
<center> <img src="https://www.dropbox.com/s/kyk4jes1202az4m/figure3.jpg?dl=1" width="600"> </center>

위의 증명 수식에서 Q와 b의 각각 세번째 수식과 마지막 수식을 자세히 봅시다.

1. $Q$에 대한 증명
    - 빨간색 부분
        - $\mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty}} [Q_t (s_{0:\infty}, a_{0:\infty})]$를 쉽게 말하자면 "모든(과거, 현재, 미래) state와 action에 대해서 expectation을 $t+1$부터 $\infty$까지 하겠다."라는 말입니다. 이렇게 함으로써 원래는 $Q^\pi (s_t, a_t)$가 나와야 맞는 건데, 앞에서 살펴봤다시피 4번째 수식 밑줄 친 자리에는 advantage 역할을 하는 함수들을 넣어도 결과값에 아무런 영향을 끼치지 않기 때문에 의도적으로 $A^\pi (s_t, a_t)$로 바꾼 것입니다. 또한 이 증명을 바탕으로 $\hat{A}$이 $\gamma$-just라는 것을 표현하고 싶기 때문이라고 봐도 됩니다.

    - 파란색 부분
        - 또한 $a_{0:t}$에서 $a_{0:t-1}$로 변한 것은 $A^\pi (s_t, a_t)$에 baseline이 들어가기 때문에 바꿔준 것입니다.

2. $b$에 대한 증명
    - 빨간색 부분
        - $\mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty}} [\nabla_\theta \log \pi_\theta (a_t | s_t)]$가 0으로 바뀌는 것은 위에서 설명했듯이 $\nabla_\theta$자체가 $\log \pi_\theta$만 gradient하기 때문에 이것을 expectation을 취하면 0이 됩니다.

<br><br>

# 4. Advantage Function Estimation

이번 section에는 discounted advantage function $A^{\pi, \gamma} (s_t, a_t)$의 accurate estimate $\hat{A}_t$에 대해서 살펴봅시다. 이에 따른 수식은 다음과 같습니다.

$$\hat{g} = \frac{1}{N} \sum_{n=1}^N \sum_{t=0}^\infty \hat{A}_t^n \nabla_{\theta} \log \pi_{\theta}(a_t^n | s_t^n)$$

여기서 n은 a batch of episodes에 대하여 index한 것입니다.

$V$를 approximate value function이라고 합시다. 그리고 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$이라고 합시다.

만약 (이전과 마찬가지로 가정부터 합니다.) correct value function $V = V^{\pi, \gamma}$가 있다고 한다면, 이것은 $\gamma$-just advantage estimator입니다. 실제로, $A^{\pi, \gamma}$의 unbiased estimator는 다음과 같습니다.
$$\mathbb{E}_{s_{t+1}} [\delta_t^{V^{\pi, \gamma}}] = \mathbb{E}_{s_{t+1}} [r_t + \gamma V^{\pi, \gamma} (s_{t+1}) - V^{\pi, \gamma} (s_t)]$$
$$= \mathbb{E}_{s_{t+1}} [Q^{\pi, \gamma} (s_t, a_t) - V^{\pi, \gamma} (s_t)] = A^{\pi, \gamma} (s_t, a_t)$$
그러나, "이 estimator는 유일하게 $V = V^{\pi, \gamma}$에 대한 $\gamma$-just입니다." 다른 경우라면 이것은 biased policy gradient estimate일 것입니다. (우리가 하고 싶은 것은 $V$에 대해서만 unbiased estimator가 아니라 advantage function에 대해서 일반화된 unbiased estimator를 얻고 싶은 것입니다. 그래서 아래에서도 나오겠지만, $\gamma$와 함께 $\lambda$를 이용한 estimator가 나옵니다.ㅏ)

그래서 $\delta$에 대해 $k$의 sum으로 생각해봅시다. 이것을 $\hat{A}_t^{(k)}$라고 하자. 그러면 아래와 같이 표현할 수 있습니다.

<center> <img src="https://www.dropbox.com/s/ra7hxksveg2hz45/figure4.jpg?dl=1" width="600"> </center>

$\hat{A}_t^{(k)}$은 returns의 $k$-step estimate와 연관지을 수 있고, $\delta_t^V = \hat{A}_t^{(1)}$의 case와 유사하게도 $\hat{A}_t^{(k)}$를 advantage function의 estimator로 생각할 수 있습니다.

여기서 $k \rightarrow \infty$로 생각해보면 bias가 일반적으로 점점 더 작아집니다. 왜냐하면 $\gamma^k V(s_{t+k})$가 점점 많이 discounted되서 $-V(s_t)$가 bias에 영향을 미치지 못하기 때문입니다. $k \rightarrow \infty$를 취하면 다음과 같은 수식이 나옵니다.

$$\hat{A}_t^{(\infty)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = -V(s_t) + \sum_{l=0}^\infty \gamma^l r_{t+l}$$
우변의 수식과 같이 empirical returns에서 value function baseline을 뺀 것으로 나타낼 수 있습니다.

Generalized Advantage Estimator GAE($\gamma, \lambda$)는 $k$-step estimators의 exponentially-weighted average로 나타낼 수 있습니다.

<center> <img src="https://www.dropbox.com/s/yg1ybmfkep3towu/figure5.jpg?dl=1" width="600"> </center>

(TD($\lambda$)를 떠올려주세요..!)

위의 마지막 수식에서도 알 수 있듯이, advantage estimator는 Bellman residual terms의 discounted sum과 관련있는 간단한 수식입니다. 다음 section에서 modified reward function을 가진 MDP에 returns로서의 관점에서 위의 마지막 수식을 더 자세히 살펴봅시다. 위의 수식은 TD($\lambda$)와 많이 유사합니다. 그러나 TD($\lambda$)는 value function를 estimator하고, 여기서는 advantage function을 estimator합니다.

위의 수식에서 $\lambda = 0$ and $\lambda = 1$에 대해서는 특별한 case가 존재한다. 수식으로 표현하면 다음과 같습니다.

<center> <img src="https://www.dropbox.com/s/ufglxzanhcnbi36/figure6.jpg?dl=1" width="600"> </center>

- $GAE(\gamma, 1)$은 $V$의 정확도와 관계없이 $\gamma$-just입니다. 그러나 returns의 sum때문에 high variance합니다.
- $GAE(\gamma, 0)$은 $V = V^{\pi, \gamma}$에 대해 $\gamma$-just입니다. 그리고 bias하지만 일반적으로 훨씬 lower variance를 가집니다. 
- $0 < \lambda < 1$에 대해 GAE는 parameter $\lambda$를 control하여 bias와 variance사이에 compromise를 만듭니다.

두 가지 별개의 parameter $\gamma$ and $\lambda$를 가진 advantage estimator는 bias-variance tradeoff에 도움을 줍니다. 그러나 이 두 가지 parameter는 각각 다른 목적을 가지고 작동합니다.

- $\gamma$는 가장 중요하게 value function $V^{\pi, \gamma}$ 의 scale을 결정합니다. 또한 $\gamma$는 $\lambda$에 의존하지 않습니다. $\gamma < 1$로 정하는 것은 policy gradient estimate에서 bias합니다.
- 반면에, $\lambda < 1$는 유일하게 value function이 부정확할 때 bias합니다. 그리고 경험상, $\lambda$의 best value는 $\gamma$의 best value보다 훨씬 더 낮습니다. 왜냐하면 $\lambda$가 정확한 value function에 대해 $\gamma$보다 훨씬 덜 bias하기 때문입니다.

GAE를 사용할 때, $g^\gamma$의 biased estimator를 구성할 수 있습니다. 수식은 다음과 같습니다.
$$g^\gamma \approx \mathbb{E} [\sum_{t=0}^\infty] \nabla_\theta \log \pi_\theta (a_t | s_t) \hat{A}_t^{GAE(\gamma, \lambda)}] = \mathbb{E} [\sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta (a_t | s_t) \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+1}^V]$$
여기서 $\lambda = 1$일 때 동일해집니다.

<br><br>

# 5. Interpretation as Reward Shaping

이번 section에서는 앞서 다뤘던 수식 $\sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$를 modified reward function의 MDP의 관점으로 생각해봅시다. 조금 더 구체적으로 말하자면, MDP에서 reward shaping transformation을 실행한 후에 적용된 extra discounted factor로서 $\lambda$를 어떻게 볼 것인지에 대해서 다룹니다.

(개인적인 comment) 한 가지 먼저 언급하자면, 본래의 목적은 reward shaping이 아니라 variance reduction입니다. 이번 section은 그저 이전에 이러한 개념이 있었고, gae를 다른 관점에서 생각해보자라는 뜻에서 나온 section인 것 같습니다. '아~ 이러한 개념이 있구나~!' 정도로만 알면 될 것 같습니다. 'gae가 reward shaping의 효과까지 있어서 이러한 section을 넣은 것일까?' 라는 생각도 해봤지만 아직 잘 모르겠습니다. 실험부분에도 딱히 하지 않은 걸로 봐서는.. 아닌 것 같기도 하고.. 아직까지는 그저 'reward shaping의 관점에서 봤을 때에도 gae로 만들어줄 수 있다.' 정도만 생각하려고 합니다.

Reward Shaping이란 개념 자체는 일반적인 알고리즘 분야(최단경로 문제 등)에도 있습니다. 하지만 이 개념이 머신러닝에 적용된 건 Andrew Y. Ng이 쓴 Policy Invariance under Reward Transformations Theory and Application to Reward Shaping(1999) 논문입니다. 논문에서 Reward Shaping을 설명하는 부분은 간략하게 다음과 같습니다.

<center> <img src="https://www.dropbox.com/s/0w9cuxjkhz2mbcy/figure7.jpg?dl=1" width="400"> </center>
<center> <img src="https://www.dropbox.com/s/xcp9eqvcbv3268g/figure8.jpg?dl=1" width="400"> </center>

위의 글만 보고 이해가 잘 안됩니다. 그래서 다른 자료들을 찾던 중에 다음의 그림을 찾게 되었습니다. (아래의 그림은 Youtube에 있는 [Udacity 영상](https://www.youtube.com/watch?v=xManAGjbx2k&t=95s)에서 있는 그림입니다.)

<center> <img src="https://www.dropbox.com/s/87kacngez9cl2e6/figure9.jpg?dl=1" width="400"> </center>

이 그림을 통해서 Reward Shaping을 이해해봅시다.

Reward Shaping(보상 형성)을 통해서 뭘 하고 싶은지 목적부터 봅시다. reward space가 sparse 한 경우에 reward가 너무 드문드문 나옵니다. 따라서 이것을 꾸준히 reward를 받을 수 있도록 바꿉니다. potential-based shaping function인 $\Phi$를 만들어서 더하고 빼줍니다. 저 $\Phi$ 자리에는 대표적으로 state value function이 많이 들어간다고 합니다. (아직 목적이 이해가 안될 수 있습니다. Reward Shaping에 대해서 다 보고 나서 다시 한 번 읽어봅시다.)

위의 그림은 돌고래가 점프하여 불구멍을 통과해야하는 환경입니다. 하지만 저 불구멍을 통과하기 위해서는 (1) 점프도 해야하고, (2) 불도 피해야하고, (3) 알맞게 착지까지 완료해야합니다. 이렇게 해야 reward를 +1 얻습니다. 

생각해봅시다. 어느 세월에 점프도 해야하고.. 불도 피해야하고.. 알맞게 착지까지해서 reward를 +1 얻겠습니까? 따라서 그 전에도 잘하고 있는 지, 못하고 있는 지를 판단하기 위해, 다시 말해 reward를 꾸준히 받도록 다음과 같이 transformed reward function $\tilde{r}$을 정의합니다.

$$\tilde{r} (s, a, s') = r(s, a, s') + \gamma \Phi (s') - \Phi (s)$$

- 여기서 $\Phi : S \rightarrow \mathbb{R}$를 state space에서의 arbitrary scalar-valued function을 나타냅니다. 그리고 $\Phi$자리에는 대표적으로 state value function이 들어간다고 생각합시다.
- 형태가 TD residual term의 결과와 비슷하지만 의미와 의도가 다릅니다. reward shaping은 sparse reward 때문이고, 이전 section에서 봤던 gae는 variance reduction때문에 나온 것입니다.

<center> <img src="https://www.dropbox.com/s/tkfpqniaazwfbld/figure10.jpg?dl=1" width="600"> </center>

이 transformation은 discounted advantage function $A^{\pi, \gamma}$으로도 둘 수 있습니다. state $s_t$를 시작하여 하나의 trajectory의 rewards의 discounted sum을 표현하면 다음과 같습니다.
$$\sum_{l=0}^\infty \gamma^l \tilde{r} (s_{t+l}, a_t, s_{t+l+1}) = \sum_{l=0}^\infty \gamma^l r(s_{t+l}, a_{t+l}, s_{t+l+1}) - \Phi(s_t)$$

$\tilde{Q}^{\pi, \gamma}, \tilde{V}^{\pi, \gamma}, \tilde{A}^{\pi, \gamma}$를  transformed MDP의 value function과 advantage function이라고 하면, 다음과 같은 수식이 나옵니다.
$$\tilde{Q}^{\pi, \gamma} (s, a) = Q^{\pi, \gamma} (s, a) - \Phi (s)$$
$$\tilde{V}^{\pi, \gamma} (s, a) = V^{\pi, \gamma} (s, a) - \Phi (s)$$
$$\tilde{A}^{\pi, \gamma} (s, a) = (Q^{\pi, \gamma} (s, a) - \Phi (s)) - (V^\pi (s) - \Phi (s)) = A^{\pi, \gamma} (s, a)$$

이제 reward shaping의 idea를 가지고 어떻게 policy gradient estimate를 얻을 수 있는 지에 대해서 알아봅시다.

- $0 \le \lambda \le 1$의 범위에 있는 steeper discount $\gamma \lambda$를 사용합니다.
- shaped reward $\tilde{r}$는 Bellman residual term $\delta^V$와 동일합니다.
- 그리고 $\Phi = V$와 같다고 보면 다음과 같은 수식이 나옵니다.

$$\sum_{l=0}^\infty (\gamma \lambda)^l \tilde{r} (s_{t+l}, a_t, s_{t+l+1}) = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V = \hat{A}_t^{GAE(\gamma, \lambda)}$$
이렇게 shaped rewards의 $\gamma \lambda$-discounted sum을 고려함으로써 GAE를 얻을 수 있습니다. 더 정확하게 $\lambda = 1$은 $g^\gamma$의 unbiased estimate이고, 반면에 $\lambda < 1$은 biased estimate입니다.

좀 더 나아가서, shaping transformation과 parameters $\gamma$ and $\lambda$의 결과를 보기 위해, response function $\chi$를 이용하면 다음과 같은 수식이 나옵니다.
$$\chi (l; s_t, a_t) = \mathbb{E} [r_{t+l} | s_t, a_t] - \mathbb{E} [r_{t+l} | s_t]$$
추가적으로 $A^{\pi, \gamma} (s, a) = \sum_{l=0}^\infty \gamma^l \chi (l; s, a)$입니다.
그래서 discounted policy gradient estimator는 다음과 같이 쓸 수 있다.
$$\nabla_\theta \log \pi_\theta (a_t | s_t) A^{\pi, \gamma} (s_t, a_t) = \nabla_\theta \log \pi_\theta (a_t | s_t) \sum_{l=0}^\infty \gamma^l \chi (l; s, a)$$

<br><br>

# 6. Value Fuction Estimation

이번 section에서는 Trust Region Optimization Scheme에 따라 Value function을 Estimation합니다.

<br>
## 6.1 Simplest approach

$$minimize_{\phi} \sum_{n=1}^{N} \vert\vert V_{\phi}(s_n) - \hat{V_n} \vert\vert^{2}$$

- 위는 가장 간단하게 non-linear approximation으로 푸는 방법입니다.
- $\hat{V_t} = \sum_{l=0}^{\infty}\gamma^l r_{t+l}$은 reward 에 대한 discounted sum을 의미합니다.

<br>
## 6.2 Trust region method to optimize the value function

- Value function을 최적화 하기 위해 trust region method를 사용합니다.
- Trust region은 최근 데이터에 대해 overfitting되는 것을 막아줍니다.

Trust region문제를 풀기 위해서는 다음 스텝을 따릅니다.

- $\sigma^2 = \frac{1}{N} \sum_{n=1}^{N} \vert\vert V_{\phi old}(s_n) - \hat{V_n} \vert\vert^{2}$을 계산합니다.
- 그 후에 다음과 같은 constrained opimization문제를 풉니다.

$$minimize_{\phi} \, \sum_{n=1}^N \parallel V_\phi (s_n) - \hat{V}_n \parallel^2$$
$$subject \, \, to \, \frac{1}{N} \sum_{n=1}^N \frac{\parallel V_\phi (s_n) - V_{\phi old} (s_n) \parallel^2}{2 \sigma^2} \le \epsilon$$

- 위의 수식은 사실 old Value function과 새로운 Value function KL distance가 $\epsilon$ 다 작아야한다는 수식과 같습니다. Value function이 평균은 $V_{\phi}(s)$이고 분산이 $\sigma^2$인 conditional Gaussian distribution으로 parameterize되었을 뿐입니다.

<img width ="400px" src="https://www.dropbox.com/s/uw0v05feu8chqmc/Screenshot%202018-07-08%2010.04.38.png?dl=1"> 

이 trust region문제의 답을 conjudate gradient algorithm을 이용하여 근사값을 구할 수 있습니다. 특히, 다음과 같은  quadratic program을 풀게됩니다.

$$minimize_{\phi} \, g^T (\phi - \phi_{old})$$
$$subject \, \, to \, \frac{1}{N} \sum_{n=1}^N (\phi - \phi_{old})^T H(\phi - \phi_{old}) \le \epsilon$$

- 여기서 $g$는 objective 의 gradient입니다.
- $j_n = \nabla_{\phi} V_{\phi}(s_n)$일때, $H = \frac{1}{N} \sum_{n} j_n j^T_n$이며, $H$는 objective의 hessian에 대해서 gaussian newton method로 근사한 값입니다. 따라서, value function을 conditional probability로 해석한다면 Fisher information matrix가 됩니다.
- 구현할때의 방법은 TRPO에서 사용한 방법과 모두 같습니다.

<br><br>

# 7.Experiments

실험은 다음 두 가지 물음에 대해서 디자인 되었습니다.

- GAE에 따라서 episodic total reward를 최적화 할때, $\lambda$와 $\gamma$가 변함에 따라서 어떤 경험적인 효과를 볼 수 있는지?
- GAE와 trust region alogorithm을 policy와 value function 모두에 함께 사용했을 때 어려운 문제에 적용되는 큰 뉴럴넷을 최적화 할 수 있을지?

<br>
## 7.1 Policy Optimization Algorithm

Policy update는 TRPO를 사용합니다. TRPO에 대한 설명은 여기서는 생략하겠습니다. TRPO 포스트를 보고 돌아와주세요!

- 이전 TRPO에서 이미 TRPO와 다른 많은 알고리즘들을 비교하였기 때문에, 여기서는 똑같은 짓을 반복하지 않고 $\lambda$, $\gamma$가 변함에 따라 어떤 영향이 있는 지에 대한 실험에 집중하겠다고 합니다. (귀찮았던거죠..)

- TRPO를 적용한 GAE의 최종 알고리즘은 다음과 같습니다.

<center> <img width = "600px" src="https://www.dropbox.com/s/b1klz11f2frrvg4/Screenshot%202018-07-08%2010.33.54.png?dl=1"> </center>
<center> <img width = "300px" src="https://www.dropbox.com/s/35fxte05pqtiel7/Screenshot%202018-07-08%2010.35.22.png?dl=1"> </center>
<center> <img width = "300px" src="https://www.dropbox.com/s/u35pjow3w50bvz1/Screenshot%202018-07-08%2010.36.03.png?dl=1"> </center>

- 여기서 주의할 점은 Policy update($\theta_i \rightarrow \theta_{i+1}$)에서 $V_{\phi_i}$를 사용했다는 점입니다.
- 만약 Value function을 먼저 update하게 된다면 추가적인 bias가 발생합니다.
- 극단적으로 생각해보아서, 우리가 Value function을 완벽하게 overfit을 해낸다면 Bellman residual($r_t + \gamma V(s_{t+1}) - V(S_t)$)은 0이 됩니다. 그럼 Policy gradient의 estimation도 거의 0이 될 것입니다.

<br>
## 7.2 Expermint details

### 7.2.1 Environment 
실험에서 사용된 환경은 다음 네 가지 입니다.

1. classic cart-pole (x 3D)
2. bipedal locomotion
3. quadrupedal locomotion
4. dynamically standing up for the biped

### 7.2.2 Architecture

- 3D robot task에 대해서는 같은 모델을 사용하였습니다.
    - layers  = [100, 50, 25] 각각 tanh 사용. (Policy와 Value 네트워크 모두)
    - Final output layer은 linear
- Cartpole에 대해서는 1개의 layer 안에 20개의 hidden unit만 있는 linear policy를 사용했다고 합니다.

### 7.2.3 Task

- Cartpole
    - 한 배치당 20 개의 trajectory를 모았고, maximum length는 1000입니다.
- 3D biped locomotion
    - 33 dim state , 10 dim action
    - 50000 time step per batch
- 3D quadruped locomotion
    - 29 dim state, 8 dim action
    - 200000 time step per batch
- 3D biped locomotion Standing
    - 33 dim state , 10 dim action
    - 200000 time step per batch


### 7.2.3 results
cost의 관점에서 결과를 나타내었다고 합니다. Cost는 negative reward와 이것이 최소화 되었는가로 정의되었습니다.

#### 7.2.3.1 Cartpole

<center> <img width = "500px" src="https://www.dropbox.com/s/x9pbms1wvg38lda/Screenshot%202018-07-08%2011.08.22.png?dl=1"> </center>

- 왼쪽 그림은 $\gamma$를 0.99로 고정시켜놓은 상태에서 $\lambda$를 변화시킴에 따라서 cost를 측정한 것입니다. 
- 오른쪽은 $\gamma$와 $\lambda$를 둘 다 변화 시키면서 성능을 그림으로 나타낸 표입니다. 흰색에 가까울 수록 좋은 성능입니다. 

#### 7.2.3.2 3D BIPEDAL LOCOMOTINO

<center> <img width = "500px" src="https://www.dropbox.com/s/i9wj4p6ijojsy82/Screenshot%202018-07-08%2011.25.08.png?dl=1"> </center>


- 다른 random seed로 부터 9번 씩 시도한 결과를 mean을 취해서 사용합니다.
- Best performance는   $\gamma \in [0.99, 0.995]$ 그리고  $\lambda \in [0.96, 0.99]$일때. 
- 1000 iteration 후에 빠르고 부드럽고 안정적인 걸음거이가 나옵니다.
- 실제로 걸린 시간은 0.01(타입스텝당 시간) * 50000(배치당 타임스텝) * 1000(배치) * 3600(초->시간) * 24 = 5.8일 정도가 걸렸습니다.

#### 7.2.3.3 다른 ROBOT TASKS

<center> <img width = "500px" src="https://www.dropbox.com/s/fuuat65we52quht/Screenshot%202018-07-08%2011.33.11.png?dl=1"> </center>

- 다른 로봇 TASK에 대해서는 아주 제한적인 실험만 진행합니다.(시간이 부족했던듯 하네요..)
- Quadruped에 대해서는 $\gamma = 0.995$로 fix, $\lambda \in {0, 0.96}$
- Standingup에 대해서는 $\gamma = 0.99$로 fix, $\lambda \in {0, 0.96}$

<br><br>

# 8. Discussion

<br>
## 8.1 Main discussion

지금까지 복잡하고 어려운 control problem에서 Reinforcement Learning(RL)은 high sample complexity 때문에 제한이 되어왔습니다. 따라서 이 논문에서 그 제한을 풀고자 advantage function의 good estimate를 얻는 "variance reduction"에 대해 연구하였습니다.

"Generalized Advantage Estimator(GAE)"라는 것을 제안했고, 이것은 bias-variance tradeoff를 조절하는 두 개의 parameter $\gamma,\lambda$를 가집니다.
또한 어떻게 Trust Region Policy Optimization과 value function을 optimize하는 Trust Region Algorithm의 idea를 합치는 지를 보였습니다.

이렇게 함으로써 보다 더 복잡하고 어려운 control task들을 해결할 수 있었습니다.

GAE의 실험적인 입증으로는 robotic locomotion을 simulation하는 domain입니다. 실험에서도 보여준 것처럼 [0.9, 0.99]의 범위에서 $\lambda$의 적절한 중간의 값을 통해 best performance를 얻습니다. 좀 더 나아가 연구되어야할 점은 adaptive or automatic하도록 estimator parameter $\gamma,\lambda$를 조절하는 방법입니다.

<br>
## 8.2 Future work

Value function estimation error와 Policy gradient estimation error사이의 관계를 알아낸다면, 우리는 Value function fitting에 더 잘 맞는 error metric(policy gradient estimation 의 정확성과 더 잘 맞는 value function)을 사용할 수 있습니다. 여기서 Policy와 Value function의 파라미터를 공유하는 모델을 만드는 것은 아주 흥미롭고 이점이 많습니다. 하지만 수렴을 보장하도록 적절한 numerical optimization을 제시해야 할 것입니다.

추가적으로 DDPG는 별로라고 합니다. 그리고 TD(0)는 bias가 너무 크고, poor performance로 이끈다고 합니다. 특히나 이 논문에서는 low-dimention의 쉬운 문제들만 해결했습니다.

<center> <img width = "500px" src="https://www.dropbox.com/s/nhc7t9psul5lr3x/Screenshot%202018-07-08%2011.45.15.png?dl=1"> </center>

<br>
## 8.3 FAQ

- Compatible features와는 무슨 관계?
     - Compatible features는 value function을 이용하는 policy gradient 알고리즘들과 함께 자주 언급됩니다.
     - Actor Critic의 저자는 policy의 제한된 representation power때문에, policy gradient는 단지 advantage function space의 subspace에만 의존하게 됩니다.
     - 이 subspace는 compatible features에 의해 span됩니다.
     - 이 이론은 현재 문제 구조를 어떻게 이용해야 advantage function에 대해 더 나은 estimation을 할 수 있는 지에 대한 지침을 주지 않습니다. GAE 논문의 idea와 orthogonal합니다.
- 왜 Q function을 사용하지 않는가?
     - 먼저 state-value function이 더 낮은 차원의 input을 가진다. 그래서 Q function보다 더 학습하기가 쉽습니다.
     - 두 번째로 이 논문에서 제안하는 방법으로는 high bias estimator에서 low bias estimator로 $\lambda$를 통해서 부드럽게 interpolate를 할 수 있습니다.
     - 반면에 Q를 사용하면 단지 high-bias estimator 밖에 사용할 수 없습니다.
     - 특히나 return에 대한 one-step estimation은 엄두를 못낼 정도로 bias가 큽니다.

<br><br>

# 다음으로

# [PPO 여행하기](https://reinforcement-learning-kr.github.io/2018/06/22/7_ppo/)
