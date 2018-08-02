---
title: Proximal Policy Optimization
date: 2018-06-22 16:53:12
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 이동민, 장수영, 차금강
subtitle: 피지여행 7번째 논문
---

<center> <img src="https://www.dropbox.com/s/145van5kldfvvd5/Screen%20Shot%202018-07-18%20at%201.19.30%20AM.png?dl=1" width="700"> </center>

논문 저자 : John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
논문 링크 : https://arxiv.org/pdf/1707.06347.pdf
Proceeding : ??
정리 : 이동민, 장수영, 차금강

---

# 1. 들어가며...

Sutton_PG부터 시작하여 TRPO, GAE를 거쳐 PPO까지 대단히 고생많으셨습니다. 먼저 이 논문은 TRPO보다는 쉽습니다. cliping이라는 새로운 개념이 나오지만 크게 어렵진 않습니다.

이 논문에서는 Reinforcement Learning에서 Policy Gradient Method의 새로운 방법인 PPO를 제안합니다. 이 방법은 agent가 환경과의 상호작용을 통해 data를 sampling하는 것과 stochastic gradient ascent를 사용하여 "surrogate" objective function을 optimizing하는 것을 번갈아가면서 하는 방법입니다. data sample마다 one gradient update를 수행하는 기존의 방법과는 달리, minibatch update의 multiple epochs를 가능하게 하는 새로운 objective function을 말합니다.

이 알고리즘의 장점으로는

- Trust Region Policy Optimization(TRPO)의 장점만을 가집니다. 다시 말해 알고리즘으로 학습하기에 훨씬 더 간단하고, 더 일반적이고, 더 좋은 sample complexity (empirically)를 가집니다. 
- 또한 다른 online policy gradient method들을 능가했고, 전반적으로 sample complexity(특히 computational complexity), simplicity, wall-time가 좋다고 합니다.

<br><br>

# 2. Introduction

<br>
## 2.1 대표적인 방법들

- DQN
    - Discrete action space를 가지는 문제들에는 효과적으로 적용 가능하지만, continuous control에도 잘 작동하는지는 검증되지 않았습니다.
- A3C - "Vanilla" policy gradient methods
    - Data efficiency와 robustness 측면이 좋지 않습니다.
- TRPO
    - 간단히 말해 복잡합니다.
    - 또한 noise(ex. dropout)나 parameter sharing(policy와 value function 간 혹은 auxiliary tasks와의)를 포함하는 architecture와의 호환성 없습니다.

<br>
## 2.2 대표적인 방법들 대비 개선사항

- Scalability
    - large models and parallel implementations
- Data Efficiency
- Robustness
    - hyperparameter tuning없이 다양한 문제들에 적용되어 해결

<br>
## 2.3 제한하는 알고리즘

이 논문에는 Clipped probability ratios를 포함하는 objective function 제안하였습니다.

- TRPO의 data efficiency와 robustness를 유지하면서도 first-order approximation만 사용합니다.
- Policy 성능에 대한 lower bound를 제공합니다.

따라서 Policy로부터의 data sampling과 sampled data를 이용한 최적화를 번갈아가면서 수행합니다.

<br>
## 2.4 실험 결과

다양한 버전의 surrogate objectives 중에는 clipped probability ratio가 가장 성능이 좋았습니다.

- Continuous control tasks에서 기존 알고리즘 대비 성능이 좋습니다.
- Atari에서는 A2C 대비 sampling efficiency 측면에서는 성능이 월등히 좋으며, ACER 대비 훨씬 간단하지만 성능은 비슷합니다.

<br><br>

# 3. Backgroud: Policy Optimization

<br>
## 3.1 Policy Gradient Methods

일반적인 PG Method들은 실험적으로 destructive large policy updates가 발생합니다.

- 더 자세히 말하자면, PG Method가 수행하는 parameter space에서의 gradual update가 policy space에서는 큰 변화를 유발할 수 있다는 의미입니다.

<br>
## 3.2 Trust Region Methods

Policy update 크기에 대한 contraint하에 objective function("surrogate" function)을 최대화하는 것이 목표입니다. 수식은 아래와 같습니다.

<center> <img src="https://www.dropbox.com/s/gx6udoz5upswyf9/Screen%20Shot%202018-07-31%20at%2011.11.49%20PM.png?dl=1" width="400"> </center>

위의 수식은 contraint로 인해 excessive large policy update가 방지됩니다.

TRPO에서는 constrained optimization problem을 풀기 위해서는 다음과 같은 방법들이 필요합니다.

1. Fisher Information Matrix인 second-order derivative of KL divergence를 사용하거나,
    - 여기서 second-order matrixes를 구하기 위해서는 많은 계산량 필요합니다.
2. Conjugate Gradient를 사용합니다.
    - Conjugate Gradient는 구현하기가 어렵습니다.

<center> <img src="https://www.dropbox.com/s/6xpw9igndl3dmb9/Screen%20Shot%202018-07-31%20at%2011.15.13%20PM.png?dl=1" width="500"> </center>

원래 이론적으로 위의 수식과 같이 "contraint"가 아니라 objective에 "penalty"를 부여하는 형태입니다. 하지만 다양한 문제들(혹은 학습 중에 특성이 변하는 문제)에서 모두 잘 동작하는 single value $\beta$를 찾는 것(robustness)이 어렵기 때문에, TRPO에서는 penalty대신 contraint를 취하는 방식을 택한 것입니다.

<br><br>

# 4. Clipped Surrogate Objective

이번 section에서는 TRPO의 surrogate obejctive function을 강제적으로 clipping하는 방법에 대하여 말합니다.

먼저 기존의 TRPO의 surrogate function을 다음과 같이 표현합니다. 
$$r_t(\theta)=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)}, \, \, r_t(\theta old) = 1$$

위의 수식을 이용하여 TRPO의 surrogate object를 최대화합니다.
<center> <img src="https://www.dropbox.com/s/e524vwsyolp6h9n/Screen%20Shot%202018-07-26%20at%2010.12.22%20AM.png?dl=1" width="350"> </center>

위의 수식을 그대로 사용한다면 excessively large policy update가 됩니다. 따라서 penalty를 이용하여 필요 이상의 policy update를 방지합니다. TRPO에서는 KL-Divergence를 이용하여 penalty를 적용하지만 PPO에서는 computation적으로 효율적인 penalty를 적용하고 excessively large policy update를 방지하기 위해 아래와 같은 clipping 기법을 사용합니다.

$$L^{CLIP}(\theta) = \hat{E}_t [min(r_t(\theta) \, \hat{A}_t,  clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \, \hat{A}_t)]$$

- 위의 수식에 대한 추가적인 설명
    - $\epsilon$은 hyperparameter로서 continuous control에서는 0.2일 때 성능이 가장 좋았으며, Atari Game에서는 0.1 x $\alpha$ 값을 사용합니다. (여기서 $\alpha$ 는 학습률로 1 에서 시작하여 학습이 진행됨에 따라 0 으로 감소합니다.)
    - Clipped 와 unclipped objective 중 min 값을 택함으로써 $L^{CLIP} (\theta)$는 unclipped objective에 대한 lowerbound가 됩니다.

이어서 아래의 그림을 보겠습니다.

<center> <img src="https://www.dropbox.com/s/3wcuf2tq7q24fy6/Screen%20Shot%202018-07-26%20at%2010.25.08%20AM.png?dl=1" width="600"> </center>

위의 그림을 보면 두 가지의 그래프(Advantage Function $\hat{A}_t$의 부호에 따라)로 clip에 대해서 설명하고 있습니다.

- Advantage Function $\hat{A}_t$가 양수일 때
    - Advantage가 현재보다 높다라는 뜻이며 파라미터를 +의 방향으로 업데이트 하여야 합니다. 다시 말해 어떠한 상태 $s$에서 행동 $a$가 평균보다 좋다는 의미입니다. 따라서 이를 취할 확률이 증가하게 되고, $r_t (\theta)$를 clip하여 $\epsilon$보다 커지지 않도록 유도하는 것입니다.
    - 추가적으로, TRPO에서 다뤘던 constraint가 아니라 단순히 clip하는 것이기 때문에 $\pi_\theta (a_t|s_t)$의 증가량은 $\epsilon$보다 더 커질 수도 있지만, 더 커져봐야 objective function에 도움이 되지 않기에 대부분 $\epsilon$ 이하로 유지합니다.
    - 또한 만약 $r_t (\theta)$가 objective function의 값을 감소시키는 방향으로 움직이는 경우더라도 $1-\epsilon$보다 작아져도 됩니다. 여기서의 목적은 최대한 lowerbound를 구하는 것이 목적이기 때문입니다. 그러니까 쉽게 말해서 왼쪽 그림에서도 볼 수 있듯이 $1+\epsilon$만 구하는 데에 포커스를 맞추고 있고, $1-\epsilon$은 신경쓰지 않고 있는 것입니다. (이 부분은 Advantage Function $\hat{A}_t$가 음수일 때도 동일합니다.)
- Advantage Function $\hat{A}_t$가 음수일 때 
    - Advantage가 현재보다 좋지 않다라는 뜻이며 그의 반대 방향으로 업데이트 하여야 합니다. 다시 말해 어떠한 상태 $s$에서 행동 $a$가 평균보다 좋지 않다는 의미입니다. 따라서 이를 취할 확률이 감소하게 되고, $r_t (\theta)$를 clip하여 $\epsilon$보다 작아지지 않도록 유도하는 것입니다.  
    - 또한 $r_t(\theta)$은 확률을 뜻하는 두 개의 함수를 분자 분모로 가지고 있으며, 분수로 구성되어 있기 때문에 무조건 양수로 이루어져 있습니다. Advantage function인 $\hat{A}_t$와 곱해져 Objective function인 $L^{CLIP}$은 Advantage function과 방향이 같아집니다.

위의 설명으로 인하여 나오는 그림이 아래의 그림입니다.

<center> <img src="https://www.dropbox.com/s/f7i97geiligfej9/Screen%20Shot%202018-07-26%20at%2011.02.17%20AM.png?dl=1" width="700"> </center>

위의 그림에서 빨간색 그래프를 보면,

- $L^{CLIP}$은 min 값들의 평균이기 때문에 평균들의 min 값보다는 더 작아집니다. 즉, 주황색 그래프와 초록색 그래프 중 작은값보다 더 작아지는 것을 볼 수 있습니다.
- $L^{CLIP}$을 최대화하는 $\theta$가 다음 $\pi_\theta$가 됩니다. 다시 말해 PPO는 기존 PG 방법들처럼 parameter space에서 parameter $\theta$를 점진적으로 업데이트하는 것이 아니라, 매번 policy를 maximize하는 방향으로 업데이트하는 것으로 볼 수 있는 것입니다.


<br><br>

# 5. Adaptive KL Penalty Coefficient

이전까지 설명했던 Clipped Surrogate Objective 방법과 달리 기존의 TRPO에서 Adaptive한 파라미터를 적용한 또 다른 방법에 대해서 알아보겠습니다. 사실 이 방법은 앞서본 clip을 사용한 방법보다는 성능이 좋지 않습니다. 하지만 baseline으로써 알아볼 필요가 있습니다.

clip에서 다뤘던 Probability ratio $r_t(\theta)$ 대신 KL divergence를 이용하여 penalty를 줍니다. 그 결과 각각의 policy update에 대해 KL divergence $d_{targ}$의 target value를 얻습니다.

- clipped surrogate objective 대신하여 or 추가적으로 사용할 수 있다고 합니다. 
- 자체 실험 결과에서는 clipped surrogate objective 보다는 성능이 안 좋았다고 합니다.
- 둘 다 사용한 실험 결과는 없습니다.

이 알고리즘에서 각각의 policy update에 적용하는 step은 다음과 같습니다. 

1. KL-penalized objective 최적화를 합니다. 기존의 TRPO의 Objective function에 $\beta$를 적용하여 다음과 같이 $\beta$를 Adaptive하게 조절하고 있습니다. 수식은 아래와 같습니다.

<center> <img src="https://www.dropbox.com/s/ojq7kfobf0f8x9n/Screen%20Shot%202018-07-26%20at%2011.22.44%20AM.png?dl=1" width="500"> </center>

2. <img src="https://www.dropbox.com/s/j9phkalrbgf45k0/Screen%20Shot%202018-07-26%20at%2011.34.25%20AM.png?dl=1" width="230">을 계산합니다.
    - 만약 $d < d_{targ} \, / \, 1.5$라면, 그 때 $\beta \leftarrow \beta \, / 2$
    - 만약 $d > d_{targ} \, x \, 1.5$라면, 그 때 $\beta \leftarrow \beta \, x 2$
    - 즉, KL-divergence 값이 일정 이상 커지게 되면 objective function에 penalty를 더 크게 부과합니다.
    - 갱신된 $\beta$ 값은 다음 policy update 때 사용합니다. 

$\beta$를 조절하는 방법은 만약 $\theta old$와 $\theta$간의 파라미터 차이가 크다면 penalty를 강하게 부여하여 차이를 작게하고, 파라미터 차이가 작다면 penalty를 완화시켜 주어 차이를 더 크게하는 것입니다.

정리하자면, KL-divergence를 constraint로 둔 것이 아니기 때문에, 간혹 excessive large policy update가 발생할 수도 있지만, KL Penaly coefficienct인 $\beta$가 KL-divergence에 따라 adpative 하게 조정됨으로써 excessive large policy update가 지속적으로 발생되는 것을 방지하는 것입니다.

<br><br>

# 6. Algorithm

앞서 다뤘던 section들은 어떻게 policy만을 업데이트 하는지에 대해 설명하고 있습니다. 이번 section에서는 value, entropy-exploration과 합쳐서 어떻게 통합적으로 업데이트 하는지에 대해서 알아보겠습니다.

먼저 Variance-reduced advantage function estimator을 계산하기 위해 learned state-value fucntion $V(s)$을 사용합니다. 여기에는 두 가지 방법이 있습니다.

- Generalized advantage estimation
- Finite horizontal estimators

만약 policy와 value function 간 parameter sharing하는 neural network architecture를 사용한다면, policy surrogate와 value function error term을 combine한 loss function을 사용해야 합니다. 또한 이 loss function에 entropy bonus term을 추가하여, 충분한 exploration이 될 수 있도록 합니다.(exploration하는 부분은 [A3C 논문](https://arxiv.org/pdf/1602.01783.pdf)에 나와있습니다.)

그래서 앞써 다뤘던 objective function을 $L^{CLIP}$라고 표현한다면 PPO에서 제시하는 통합적으로 최대화해야하는 objective function은 다음과 같이 표현할 수 있습니다.

$$L_t^{CLIP+VF+S} (\theta) = \hat{E}_t [L^{CLIP} (\theta) - c_1 L_t^{VF} (\theta) + c_2 S[\pi_\theta] (s_t)]$$

위의 수식에 대한 추가적인 설명은 다음과 같습니다.

- $c_1, c_2$ : coefficients
- $S$ : entropy bonus
- $L_t^{VF}$ : squared-error loss $(V_\theta (s_t) - V_t^{targ})^2$

위의 objective function을 최대화하면 Reinforcement Learning을 통한 policy 학습, state-value function의 학습, exploration을 할 수 있습니다.

추가적으로 A3C와 같은 요즘 인기있는 PG의 style에서는 T time steps(T는 episode length 보다 훨씬 작은 크기) 동안 policy에 따라서 sample들을 얻고, 이 sample들을 업데이트에 사용합니다. 이러한 방식은 time step T 까지만 고려하는 advantage estimator가 필요하며, A3C에서 다음과 같이 사용합니다.

$$\hat{A}_t = - V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t+1} r_{T-1} + \gamma^{T-t}V(s_T)$$

- 여기서 $t$는 주어진 length T trajectory segment 내에서 $[0, T]$에 있는 time index입니다.

위의 수식에 더하여 generalized version인 GAE의 truncated version(generalized advantage estimation)을 사용합니다.($\lambda$가 1이면 위 식과 같아집니다.) 수식은 아래와 같습니다. 

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1},$$$$where \,\,\,\, \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

그래서 고정된 length T trajectory segment를 사용한 PPO algorithm은 다음과 같은 pseudo code로 나타낼 수 있습니다.

<center> <img src="https://www.dropbox.com/s/q00y8x7eryl1ncd/Screen%20Shot%202018-07-26%20at%202.28.03%20PM.png?dl=1" width="800"> </center>

- 여기서 actor는 A3C처럼 병렬로 두어도 상관없습니다.
- advantage estimate 계산을 통해서 surrogate loss를 계산합니다.

<br><br>

# 7. Experimnet

<br>
## 7.1 Surrogate Objectives의 비교

이번 section에서는 3가지 Surrogate Objectives를 비교분석 하고 있습니다.

* No clipping or penalty: $L_t(\theta) = r_t(\theta)\hat{A}_t$
* Clipping: $L_t(\theta) = min(r_t(\theta) \, \hat{A}_t, \, clip(r_t(\theta), 1-\epsilon,1+\epsilon) \, \hat{A}_t)$
* KL penalty(fixed or adaptive): <img src="https://www.dropbox.com/s/ksnhlxz2riuns1p/Screen%20Shot%202018-07-26%20at%202.42.50%20PM.png?dl=1" width="270">

그리고 환경은 MuJoCo를 사용하며 OpenAI Gym 환경에서 실험을 진행하였습니다. 그리고 네트워크들은 2개의 hidden layer를 가지며 각각 64개의 unit들을 가지고 있습니다. 그리고 tanh를 활성화 함수로써 사용하고 있으며 네트워크는 Gaussian distribution을 출력하는 것으로 구성되어 있습니다. 그리고 Policy Network와 Value Network는 파라미터들을 공유하고 있지 않으며, 위에서 설명한 Exploration을 위한 entropy는 사용하지 않습니다.

7개의 환경에서 실험하였으며 각 환경들은 HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPerdulum, Reacher, Swimmer, Walker2d를 사용하였으며 모든 환경들은 v1 버전을 사용하였습니다.

아래의 표는 실험 결과입니다. 세 가지 Surrogate Objective에 대해 각 파라미터들을 표에 표기된 방식대로 변화하며 실험하였습니다.

<center> <img src="https://www.dropbox.com/s/4xbkxh4m82mu1vo/Screen%20Shot%202018-07-26%20at%202.45.42%20PM.png?dl=1" width="450"> </center>

<br>
## 7.2 Comparison to other Algorithms in the Continuous Domain

이번 section에서는 Clipping 버전의 PPO와 다른 알고리즘들 간의 비교를 연속적인 Action을 가지는 환경에서 실험을 한 결과를 보여주고 있습니다. 다른 알고리즘은 TRPO, cross-entropy method, vanilla policy gradient with adaptive stepsize, A2C, A2C with trust region를 사용하였습니다. 환경과 사용한 네트워크의 구성은 이전 section과 동일합니다.

<center> <img src="https://www.dropbox.com/s/tf7djngioxnxtef/Screen%20Shot%202018-07-26%20at%202.50.23%20PM.png?dl=1" width="900"> </center>

여타 다른 알고리즘보다 좋은 성능을 보이는 것을 알 수 있습니다.

<br>
## 7.3 Showcase in the Continuous Domain: Humanoid Running and Steering

이번 section에서는 Humanoid-v0, HumanoidFlagrun-v0, HumanoidFlagrunHarder-v0의 환경에서 Roboschool을 사용하여 실험하였습니다.

- Humanoid-v0는 단순히 앞으로 걸어나가는 환경
- HumanoidFlagrun-v0은 200스텝마다 혹은 목적지에 도착할 때 마다 위치가 바뀌는 목적지에 걸어서 도달하는 환경입니다. 
- HumanoidFlagrunHarder-v0은 환경이 초기화될 때 마다 특정 경계안에 Humanoid가 위치하게 되며 특정 영역 밖으로 걸어나가는 것을 수행하는 환경입니다.


<center> <img src="https://www.dropbox.com/s/y112ua7il6l0296/Screen%20Shot%202018-07-26%20at%202.53.01%20PM.png?dl=1" width="900"> </center>

- Roboschool을 사용하여 3D humanoid control task에서 PPO 알고리즘으로 학습시킨 Learning curve입니다.

<center> <img src="https://www.dropbox.com/s/09scyk6zeviyswt/Screen%20Shot%202018-07-26%20at%202.53.10%20PM.png?dl=1" width="900"> </center>

- Roboschool Humanoid Flagrun에서 학습된 policy의 frame들입니다. 

<br>
## 7.4 Comparison to Other Algorithms on the Atari Domain

이번 section에서는 Atari Domain에서 PPO, A2C, ACER(Actor Critic with Experience Replay)의 3가지 알고리즘을 비교합니다.

<center> <img src="https://www.dropbox.com/s/jxuqdncxpnoyqgi/Screen%20Shot%202018-07-26%20at%203.00.00%20PM.png?dl=1" width="600"> </center>

전체 training에 대해서는 PPO가 ACER보다 좋은 성능을 내고 있습니다. 하지만 마지막 100 에피소드만을 비교했을 때는 PPO보다 ACER이 더 좋은 성능을 보이고 있습니다. 이는 PPO가 더 빨리 최종 능력에 도달하지만, ACER이 가진 potential이 더 높다는 것입니다.

<br><br>

# 8. Conclusion

이 논문에서는 policy update를 하기 위한 방법으로 stochastic gradient ascent의 multiple epchs를 사용하는 policy optimization method들의 하나의 알고리즘은 Proximal Policy Optimization(PPO)를 소개합니다.

이 알고리즘은 trust region method의 stability와 reliability를 가집니다. 여기에 더하여 학습하기에 훨씬 더 간단하고, 더 일반적인 setting으로 적용하기에 편한 A3C로서 code를 구성하기에도 편하고, 계산량도 훨씬 덜합니다. 그리고 전반적으로 더 좋은 성능을 가집니다.