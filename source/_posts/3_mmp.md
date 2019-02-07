---
title: Maximum Margin Planning
date: 2019-02-07
tags: ["프로젝트", "GAIL하자!"]
categories: 프로젝트
author: 이동민
subtitle: Inverse RL 3번째 논문
---

<center> <img src="../../../../img/irl/mmp_1.png" width="850"> </center>

Author : Nathan D. Ratliff, J. Andrew Bagnell, Martin A. Zinkevich
Paper Link : https://www.ri.cmu.edu/pub_files/pub4/ratliff_nathan_2006_1/ratliff_nathan_2006_1.pdf
Proceeding : International Conference on Machine Learning (ICML) 2006

---

# 0. Abstract

일반적으로 Supervised learning techniques를 통해 sequential, goal-directed behavior에 대한 imitation learning은 어렵습니다. 이 논문에서는 maximum margin을 이용한 prediction problem에 대해 sequential, goal-directed behavior을 학습하는 것을 다룹니다. 더 구체적으로 말하자면, 각 features부터 cost까지 한번에 mapping하는 것을 학습해서, 이 cost에 대한 Markov Decision Process(MDP)에서의 optimal policy가 expert's behavior를 모방하도록 하는 것입니다.

또한 inference를 위해 fast algorithms을 이용하여 subgradient method의 기반인 structured maximum margin learning으로서 간단하고 효율적인 접근을 보였습니다. 비록 이러한 fast algorithm technique는 일반적으로 사용하지만, QP formulation의 한계를 벗어난 문제에서는 $A^*$나 Dynamic Programming(DP) 접근들이 policy를 learning하는 것을 다룰 수 있도록 만든다는 것을 보였습니다.

실험에서는 outdoor mobile robot들을 이용하여 route planning에 적용합니다.

<br>
## 0.1 들어가기에 앞서..

이 논문은 앞서 다뤘던 APP 논문에 더하여 좀 더 효율적인 알고리즘을 만들고자 하였습니다. 그래서 기존에 APP에서 QP를 이용한 SVM 방법에 더하여 Soft Margin term을 추가하여 슬랙변수를 가지는 SVM을 사용하였고, subgradient method를 이용하여 알고리즘을 좀 더 쉽고 빠르게 구할 수 있도록 만들었습니다.

SVM과 Soft Margin SVM에 대해 모르는 분이 계시다면 아래의 링크를 꼭 보시고 이 논문을 보시는 것을 추천해드립니다!

1) 영상 (KAIST 문일철 교수님 강의)
  - [Lecture 1 Decision boundary with Margin](https://www.youtube.com/watch?v=hK7vNvyCXWc&list=PLbhbGI_ppZISMV4tAWHlytBqNq1-lb8bz&index=23)
  - [Lecture 2 Maximizing the Margin](https://www.youtube.com/watch?v=tZy3uRv-9mY&list=PLbhbGI_ppZISMV4tAWHlytBqNq1-lb8bz&index=24)
  - [Lecture 3 SVM with Matlab](https://www.youtube.com/watch?v=sYLuJ_8Qw3s&list=PLbhbGI_ppZISMV4tAWHlytBqNq1-lb8bz&index=25)
  - [Lecture 4 Error Handling in SVM](https://www.youtube.com/watch?v=vEivqCo-LiU&list=PLbhbGI_ppZISMV4tAWHlytBqNq1-lb8bz&index=26)
  - [Lecture 5 Soft Margin with SVM](https://www.youtube.com/watch?v=5jOqc7ByMm4)

2) 블로그 글
  - [SVM (Support Vector Machine) Part 1](https://gentlej90.tistory.com/43)
  - [SVM (Support Vector Machine) Part 2](https://gentlej90.tistory.com/44)

<br><br>

# 1. Introduction

Imitation Learning에서, learner는 expert의 "behavior" or "control strategy"를 모방하려고 시도합니다. 이러한 imitation learning을 이용하여, robotics에서의 supervised learning 접근은 featrues부터 decision까지 mapping하는 것을 학습하는 것에서 엄청난 결과로서 사용되어 왔지만, Long-range나 Goal-directed behavior에서는 이러한 기술들을 사용하기 어렵습니다.

<br>
## 1.1 Perception and Planning

Mobile robotics에서는 일반적으로 **perception** subsystem과 **planning** subsystem으로 autonomy software를 partition함으로써 Long-horizon goal directed behavior를 찾습니다.
1) Perception system은 다양한 model과 환경의 features를 계산합니다.
2) Planning system은 cost-map을 input으로 두고, 그 input을 통해 minimal risk (cost) path를 계산합니다.

<br>
## 1.2 In this work

하지만 불행하게도, perception의 model부터 planner에 대한 cost까지 하는 것은 종종 어렵습니다.

따라서 이 논문에서는 새로운 방법을 제시합니다. **perception features부터 planner에 대한 cost까지 (Perception + Planning)** mapping하는 것을 자동화하는 방법입니다. 최종 목표는 features부터 cost function까지 mapping하는 것을 한번에 학습하는 것입니다. 그래서 이러한 cost function에 대한 optimal policy가 expert의 behavior을 한번에 모방하도록 하는 것입니다.

<br>
## 1.3 Three fold

정리해보면, 이 논문에서는 3가지 중요한 언급이 있습니다.

1. planning을 학습하기위한 새로운 방법을 제시합니다.
2. sutructured maximum-margin classification으로서 효율적이고 간단한 접근을 제시합니다. 또한 batch setting에서도 linear convergence하다는 것을 보여주며, 다른 QP 기법이 없이도 lerge problems에 적용될 수 있다고 합니다.
3. 실험적으로 mobile robotics 관련 문제에 적용하였습니다.

<br><br>

# 2. Preliminaries

<br>
## 2.1 Notations

### 2.1.1 modeling the planning problem with discrete MDPs

이 논문은 discrete MDPs에서의 planning problem을 modeling합니다.

$\mathcal{X}$ is state spaces. $x$ index the state spaces.

$\mathcal{A}$ is action spaces. $a$ index the action spaces.

$p(y|x,a)$ is transition probablities.

$s$ is initial state distribution.

$\gamma$ is a discount factor on rewards. if any, $\gamma$ is aborbed into the transition probabilities.

Reward $R$은 따로 두지 않고, demonstrated behavior를 모방하는 policies를 만들기 위해 supervised examples로부터 학습됩니다.

$v \in \mathcal{V}$ is primal variables of value function

$\mu \in \mathcal{G}$ is **dual state-action frequency counts**. equal to $y$.

- 여기서 $\mu$는 어떠한 상태에서 어떠한 행동을 취했는 지를 count의 개념으로 나타낸 것인데 $y$와 혼용되어 쓸 수 있습니다.

그리고 여기서는 오직 stationary policies만을 고려합니다. The generalization is straightforward.

#### state-action frequency counts란?

Inverse RL의 궁극적인 목표를 다르게 말해보면 expert가 어떠한 행동을 했을 때, 여기서의 state-action visitation frequency counts(줄여서 visitation frequency)를 구합니다. 그리고 우리의 목적은 expert의 visitation frequency과 최대한 비슷한 visitation frequency를 만들어내는 어떤 reward를 찾는 것입니다.

일반적으로, RL 문제는 reward가 주어졌을 때, 이 reward의 expected sum을 최대로 하는 policy를 찾는다고 알려져 있습니다. 그런데 이 문제의 dual problem은 visitation frequency를 찾는 것입니다.

다시 말해 optimal policy와 optimal visitation frequency counts는 1:1 관계라는 것입니다.

### 2.1.2 The input to our algorithm

The input is a set of training instances

<center> <img src="../../../../img/irl/mmp_2.png" width="300"> </center>

$p_i$ is transition probablities.

State-action pairs $(x_i, a_i) \in \mathcal{X}_i, \mathcal{A}_i$ is $d \times |\mathcal{X}||\mathcal{A}|$ Feature matrix (or mapping) $F_i \in \mathbb{R}^{d \times |\mathcal{X}||\mathcal{A}|}$.

$y_i$ is expert's demonstration (desired trajectory or full policy). equal to **dual state-action frequency counts** $\mu_i$.

$f_i (y)$ denote vector of expected feature counts $F_i \mu$ of the $i$th example.

$\mathcal{L}_i$ : Some additional loss function (heuristic)

### 2.1.3 Detail description

$\mu_i^{x, a}$ is the expected state-action frequency for state $x$ and action $a$ of example $i$.

- $\mu_i$를 자세히보면 $\mu$에 $i$가 붙은 형태로 되어있습니다. 여기서 example $i$라는 것 총 trajectory(or path)의 length 중에서 하나의 index를 말하는 것입니다. expert의 경우, 보통 전체 trajectory를 한꺼번에 취하기 때문에 이 논문에서는 구별하기 위해 $i$라는 notation을 쓴 것입니다.

$\mathcal{D} = \\{(\mathcal{X}_i, \mathcal{A}_i, p_i, f_i, y_i, \mathcal{L}_i)\\} \equiv \\{(\mathcal{X}_i, \mathcal{A}_i, \mathcal{G}_i, F_i, \mathcal{\mu}_i, \mathcal{l}_i)\\}$ 

- ($\mathcal{D}$는 $\mathcal{D}_{i=1}^n$)

Loss function is $\mathcal{L} : \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}_+$ between solutions. $\mathcal{L} (y_i, y) = \mathcal{L}_i (y) = l_i^T \mu$. 

- $l_i \in \mathbb{R}_+^{|\mathcal{X}||\mathcal{A}|}$
- $l_i = 1 -$ expert's visitation frequency

### 2.1.4 그래서 이 논문에서 하고 싶은 것 (중요)

The best policy over the resulting reward function $\mu^* = arg\max_{\mu \in \mathcal{G}_i} w^T F_i \mu$ is “**close**” to the expert's demonstrated policy $\mu_i$.

<br>
## 2.2 Loss function

위에서 말했던 loss function은 teacher가 아닌 learner가 방문한 states의 count입니다.

또한 이 논문에서는 teacher가 도달한 어떠한 states에서 teacher와 다른 action들을 고르거나 teacher가 선택하지 않은 states를 도달하는 것을 penalizing할 것입니다.

끝으로, $\mathcal{L}(y_i, y) \geq 0$을 가정합니다.

<br>
## 2.3 Quadratic Programming Formulation

### 2.3.1 Quadratic Program

Given a training set :

<center> <img src="../../../../img/irl/mmp_2.png" width="300"> </center>

Quadratic Program is

$$\min_{w, \zeta_i} \frac{1}{2} \parallel w \parallel^2 + \frac{\gamma}{n} \sum_i \beta_i \zeta_i \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, (1)$$

$$s.t. \,\, \forall i \,\,\,\, w^T f_i (y_i) \geq \max_{\mu \in \mathcal{G}_i} (w^T f_i (y) + \mathcal{L} (y_i, y)) - \zeta_i \,\,\,\,\,\,\,\,\, (2)$$

s.t.로 나오는 constraint의 intuition을 보자면, expert's policies가 margin에 대해 다른 모든 policies보다 더 높은 experted reward를 가지도록 하는 only weight vectors를 찾자는 것입니다.

다음으로 위의 수식에 나오는 notation에 대해서 자세히 알아보겠습니다.

$\frac{\gamma}{n}\sum_i \zeta_i$ is **soft margin term** from soft margin SVM. $\zeta$의 값을 되도록 최소화하여 오분류의 허용도를 낮추기 위해 추가되었습니다.

$\zeta_i$ is **slack variable**. The slack variable permit violations of these constraints for a penalty. 여유 변수라고도 하고, $\zeta_i$만큼의 오류를 인정한다는 의미로 볼 수 있습니다.

$\gamma \geq 0$ is scaled for a penalty. 보통 $c$ ($c \geq 0$)라고도 하는데, 최소화 조건을 반영하는 정도를 결정하는 값으로 우리가 적절히 정해주어야 합니다. c값이 크면 전체 margin도 커지므로 오분류를 적게 허용(엄격)한다는 뜻이고, 반대로 c값이 작으면 margin이 작아지므로 비교적 높은 오분류를 허용(관대)한다는 뜻입니다.

$\beta_i \geq 0$는 example들이 다른 length일 때 normalization하기 위해서 사용되는 data dependent scalars입니다.

$w^T f_i(y_i)$ is expert's reward.

$w^T f_i(y)$ is other's reward.

$\mathcal{L} (y_i, y)$는 $y_i$와 $y$가 일치하지 않는 상태의 수입니다.

### 2.3.2 Maximum Margin Problem(MMP)

만약 $f_i (\cdot)$ and $\mathcal{L}_i (\cdot)$가 state-action frequencies $\mu$에서 linear하다면, MMP는 다음과 같이 정의할 수 있습니다.

$$\min_{w, \zeta_i} \frac{1}{2} \parallel w \parallel^2 + \frac{\gamma}{n} \sum_i \beta_i \zeta_i \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, (3)$$

$$s.t. \,\, \forall i \,\,\,\, w^T F_i \mu_i \geq \max_{\mu \in \mathcal{G}_i} (w^T F_i \mu + l_i^T \mu) - \zeta_i \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, (4)$$

그리고 수식 (4)에서, $\mu \in \mathcal{G}_i$를 Bellman-flow constraints로 표현할 수 있습니다. 다시 말해 $\mu \geq 0$는 다음을 만족합니다. ($\mu$의 성질)

<center> <img src="../../../../img/irl/mmp_3.png" width="450"> </center>

- 나중에 GAIL 논문에서도 나오겠지만, 위의 수식으로 $\mu$인 visitation frequency를 정할 수 있습니다.

### 2.3.3 One compact quadratic program for MMP

이어서 수식 (4)에서 nonlinear, convex constraints는 오른쪽 side의 dual을 계산함으로써 linear constraints의 compact set으로 다음과 같이 변형될 수 있습니다.

$$\forall i \,\,\,\, w^T F_i \mu_i \geq \min_{v \in V_i} \, (s_i^T v) - \zeta_i \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, (5)$$

그리고 수식 (5)에서, $v \in V_i$는 Bellman primal constraints을 만족하는 value function입니다. Bellman primal constraints는 다음과 같습니다.

$$\forall i,x,a \,\,\,\, v^x \geq (w^T F_i + l_i)^{x,a} + \sum_{x'} p_i (x'|x,a) v^{x'} \,\,\,\,\,\,\,\,\,\, (6)$$

위의 constraints (5), (6)을 combining함으로써 최종적으로 다음과 같이 쓸 수 있습니다.

One compact quadratic program is

$$\min_{w, \zeta_i} \frac{1}{2} \parallel w \parallel^2 + \frac{\gamma}{n} \sum_i \beta_i \zeta_i \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, (7)$$

$$\forall i \,\,\,\, w^T F_i \mu_i \geq \min_{v \in V_i} \, (s_i^T v) - \zeta_i \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, (8)$$

$$\forall i,x,a \,\,\,\, v_i^x \geq (w^T F_i + l_i)^{x,a} + \sum_{x'} p_i (x'|x,a) v_i^{x'} \,\,\,\,\,\,\,\, (9)$$

위의 수식은 MMP 문제를 One compact quadratic program으로써 풀 수 있도록 만든 것입니다. 하지만 아쉽게도 위의 constraints의 수는 state-action pairs와 training examples에 대해 linear하게 scaling됩니다.

이러한 program을 직접적으로 optimize할 수 있도록 이미 만들어져 있는 QP software가 있지만, 뒤이어 나오는 **section 3** 에서 **subgradient methods** 의 이용함으로써 One compact quadratic program에 대해 크게 향상시킬 수 있는 다른 alternative formulation을 이용하고자 합니다.

추가적으로, **Section 4** 에서는 최종 objective function에 유용한 방법들을 생각해 볼 것입니다. 그리고 나서 path planning problems에 대한 적절한 examples를 말할 것입니다.

<br><br>

# 3. Efficient Optimization

실제로, 수식 (9)에서의 quadratic program을 해결하는 것은 적어도 single MDP의 linear programming을 해결하는 것만큼 어렵습니다. 그래서 수식 (9)를 quadratic program으로 해결하려는 것이 적절한 전략이 될 수 있지만, 다르게 보면 많은 문제들에 대해 policy iteration과 $A^*$ algorithm처럼 이론적으로나 실험적으로나 더 빠르게 해결할 수 있도록 특별하게 design된 algorithm이 존재한다는 것으로 볼 수 있습니다.

따라서 이 논문에서는 더 나아가 **fast maximization algorithm을 사용하는 iterative method 기반인 subgradient method로 접근합니다.**

<br>
## 3.1 Objective function

첫 번째 step은 optimization program(One compact quadratic program)을 "hinge-loss" form으로 변형하는 것입니다.

변형된 objective function은 다음과 같습니다.

$$c_q(w) = \frac{1}{n} \sum_{i=1}^n \beta_i \Big( \big\\{\max_{\mu \in \mathcal{G}_i} (w^T F_i + l_i^T)\mu \big\\} - w^T F_i \mu_i\Big) + \frac{1}{2} \parallel w \parallel^2 \,\,\,\,\,\,\,\,\,\,\,\, (10)$$

hinge-loss 관점에서 보면, 위의 수식에서 $\big\\{\max_{\mu \in \mathcal{G}_i} (w^T F_i + l_i^T)\mu\big\\} - w^T F_i \mu_i$은 slack variable인 $\zeta_i$과 동일합니다.

또한 기존에 있던 $\gamma$는 slack variable $\zeta_i$가 없어졌기 때문에 사라집니다.

위의 objective function은 convex하지만, **max term**이 있기 때문에 미분이 불가능합니다. 따라서 이 논문에서는 subgradient method라 불리는 gradient descent의 generalization을 이용함으로써 optimization을 합니다.

convex function $c : \mathcal{W} \rightarrow \mathbb{R}$의 subgradient는 vector $g$로 정의합니다.

<center> <img src="../../../../img/irl/mmp_4.png" width="400"> </center>

위의 수식에서 subgradient는 비록 미분 가능한 점에서 필연적으로 gradient와는 같지만, unique할 필요는 없습니다.

### 3.1.1 Subgradient Method란?

아래의 링크를 참고해주시면 감사하겠습니다.

1) [Wikipedia - Subgradient method](https://en.wikipedia.org/wiki/Subgradient_method)
2) [모두를 위한 컨벡스 최적화 - Subgradient](https://wikidocs.net/18963)
3) [모두를 위한 컨벡스 최적화 - Subgradient Method](https://wikidocs.net/18953)

<br>
## 3.2 Four well known properties for subgradient method

최종 objective function을 보기전에, 먼저 $c(w)$의 subgradient를 계산하기 위해, subgradient에 대해 잘 알려진 4가지 속성들에 대해서 알아봅시다. (**3번 중요**)
1) Subgradient operators are linear.
2) The gradient is the unique subgradient of a differentiable function.
3) Denoting $y^∗ = arg\max_y [f (x, y)]$ for differentiable $f (., y)$, $\nabla_x f(x,y∗)$ is a subgradient of the piecewise(구분적으로, 구간적으로) differentiable convex function $\max_y [f (x, y)]$.
4) An analogous chain rule holds as expected.

3번을 보면, 결국 subgradient method를 통해 하고 싶은 것은 piecewise differentiable convex function인 $f(x,y)$ 중에서 제일 큰 $\max_y [f (x, y)]$를 subgradient로 구해서, 그 중 가장 큰 값인 $y^∗ = arg\max_y [f (x, y)]$를 통해 $\nabla_x f(x,y∗)$를 하겠다는 것입니다.

<br>
## 3.3 A subgradient method for objective function

We are now equipped to compute a subgradient $g_w \in \partial c(w)$ of our objective function (10):

$$g_w = \frac{1}{n} \sum_{i=1}^n \beta_i \big( (w^T F_i + l_i^T)\mu^* - w^T F_i \mu_i \big) \cdot F_i \Delta^w \mu_i + \lambda w \,\,\,\,\,\,\,\,\,\,\,\, (12)$$

위의 수식에서 detail한 notation은 다음과 같습니다.

$$\mu^* = arg \max_{\mu \in \mathcal{G}} (w^T F_i + l_i^T)\mu \,\,\,\,\,\,\,\,\,\,\,\, (12-1)$$

$$\Delta^w \mu_i = \mu^∗ − \mu_i \,\,\,\,\,\,\,\,\,\,\,\, (12-2)$$

수식 (12-2)를 직관적으로 보면, subgradient가 현재의 reward function $w^T F_i$에 관하여 **optimal policy와 example policy 사이의 state-action visitation frequency count를 비교한다는 점**을 발견할 수 있습니다.

또한 subgradient를 계산하는 것은 $\mu^* = arg \max_{\mu \in \mathcal{G}} (w^T F_i + l_i^T)\mu$을 해결하는 것과 같습니다. 다시 말해 reward function $w^T F_i + l_i^T$를 해결한다는 것입니다.

<br>
## 3.4 Algorithm 1. Max Margin Planning

<center> <img src="../../../../img/irl/mmp_5.png" width="500"> </center>

- 5: loss augmented cost map $(w^T F_i + l_i^T)$에 대해서 각각의 input map에 대한 optimal policy $\mu^*$와 state-action visitation frequencies $\mu^i$를 계산합니다. 처음에는 w가 0에서 시작하므로 loss augmented cost map $w^T F_i + l_i^T$은 $l_i^T$로 시작하게 됩니다.
- 6: 수식 (12)에 있는 objective function $g$를 계산합니다.
- 7: w를 $\alpha_t g$에 따라 minimize합니다.
- 8: Option으로 추가적인 constraints를 둘 수도 있습니다. 자세한 내용은 section 4.4인 Incorporating Prior Knowledge를 참고하시기 바랍니다.
- No RL step!

<br><br>

# 4. Additional section

이전까지는 최종 objective function과 algorithm을 살펴봤습니다. 여기서 더 나아가 유용한 방법들을 통해 우리의 objective function과 algorithm이 더 robust하도록 만들어봅시다.

<br>
## 4.1 Guarantees in the Batch Setting

subgradient method로 구성된 algorithm들의 잘 연구된 class 중 하나는 batch setting으로 둘 수 있다는 것입니다.

batch setting에는 두 가지 key point가 존재합니다.
1) 이 method에서 step-size sequence $\\{ \alpha_t \\}$의 선택은 상당히 중요합니다. $\\{ \alpha_t \\}$에 따라서 convergence guarantee가 달라집니다.
2) 우리의 결과는 objective function을 유지하기 위해 strong convexity assumption이 필요합니다.
따라서 Given G$\mathcal{W} \subseteq \mathbb{R}^d$, a function $f: \mathcal{W} \rightarrow \mathbb{R}$ is $\eta$-strongly convex if there exists $g: \mathcal{W} \rightarrow \mathbb{R}$ such that for all $w$, $w' \in \mathcal{W}$:

<center> <img src="../../../../img/irl/mmp_6.png" width="500"> </center>

**Theorem 1. Linear convergence of constant stepsize sequence.** Let the stepsize sequence $\\{ \alpha_t \\}$ of Algorithm (1) be chosen as $\alpha_t = \alpha \leq \frac{1}{\lambda}$. Furthermore, assume for a particular region of radius $R$ around the minimum, $\forall w,g \in \partial c(w), ||g|| \leq C$. Then the algorithm converges at a linear rate to a region of some minimum point $x^*$ of $c$ bounded by

$$||x_{min} - x^*|| \leq \sqrt{\frac{a C^2}{\lambda}} \leq \frac{C}{\lambda}$$

<center> <img src="../../../../img/irl/mmp_7.png" width="450"> </center>

Theorem 1은 우리가 충분히 작고 일정한 stepsize를 통해 linear convergence rate를 얻을 수 있다는 것을 보여줍니다. 그러나 이 convergence는 오직 minimum 주변 지역에서만 가능합니다.

대안적으로, 우리는 $t \geq 1$에 대해 $\alpha_t = \frac{r}{t}$ 형태의 감소하는 step size rule을 고를 수 있습니다. 여기서 $r$은 learning rate로 생각할 수 있는 some positive constant입니다.

이러한 rule을 통해, Algorithm 1은 minimum에서 convergence가 보장되지만, 위에서 말했던 strong convexity assumption에서만 오직 sublinear rate로 수렴될 수 있습니다.

<br>
## 4.2 Optimization in an Online Setting

다양한 optimization techniques와 다르게, subgradient method는 batch setting에서 더 확장됩니다.

online setting에서는 적절하게 관련된 domain에 대한 다양한 planning problem들을 생각해볼 수 있습니다. 특히, 그 중 하나는 path를 plan하기 위해 필요로 하는 domain을 제공하는 것입니다. 더 정확하게는 "correct" path를 제공하는 것입니다.

At each time step $i$:
1) We observe $\mathcal{G}_i$ and $F_i$.
2) Select a weight vector $w_i$ and using this compute a resulting path.
3) Finally we observe the true policy $y_i$.

즉, strongly convex cost function(앞서 다뤘던 수식 (10))이 되기 위해 $c_i(w) = \frac{1}{2} \parallel w \parallel^2 + \\{ \max_{\mu \in \mathcal{G}_i} (w^T F_i + l_i)\mu\\} − w^T F_i \mu_i$를 정의할 수 있다는 것입니다. 그리고 우리는 $y_i$, $\mathcal{G}_i$, $F_i$가 주어진다면 계산할 수 있습니다.

정리하자면, 앞서 본 cost function(Equation 10)에서 $\frac{1}{n}\sum_{i=1}^n \beta_i$가 없어진 것과 같이 online setting이 가능하다는 것을 보여줍니다.

This is now an **online convex programming problem**.

<br>
## 4.3 Modifications for Acyclic Positive Costs (Think of Worst Case)

acyclic(특정 방향이 없는, 사이클이 없는, 비순환적인) domain의 infinite horizon problems에서, $A^*$와 $A^*$의 변종들은 일반적으로 좋은 paln을 찾기위한 가장 효율적인 방법입니다. 이러한 domain에서는 strictly negative한 reward를 생각해볼 필요가 있습니다(동일하게, cost는 strictly positive). 다시 말해 best case에 대해서만 생각해볼 것이 아니라 worst case에 대해서도 생각해볼 필요가 있다는 것입니다. 이렇게 하지 않으면 infinite reward path가 발생할지도 모르기 때문입니다. 이러한 negativity의 strictness는 heuristic의 존재를 더 확실히게 보장하는 것이라고 볼 수 있습니다.

$F_i \geq 0$이라고 가정하면, 이러한 negativity의 strictness는 두 가지 방법을 통해 쉽게 구현할 수 있습니다.
1) w에 component-wise negativity constraints를 추가
2) 각각의 state-action pair에 대한 보상에 negativity를 부여하는 constraints를 추가

이렇게 negativity를 추가할 수 있는 이유는 reward $w^T F_i \mu$(or $(w^T F_i + l_i^T) \mu$)에서 $F_i$이 0보다 크기 때문에, 우리는 $w, \mu$에 negativity를 추가할 수 있습니다. 1의 경우 단순히 w의 violated component를 0으로 설정하여 구현할 수 있고, 2의 경우 가장 violated constraint를 반복적으로 추정함으로써 효율적으로 구현할 수 있습니다.

<br>
## 4.4 Incorporating Prior Knowledge

이 section은 앞서 Algorithm 1의 Line 8에서와 말한 것과 같이 Option으로 prior knowledge을 통해 추가적인 constraints를 둘 수 있다는 것을 보여줍니다.
1) 0 vector 대신에 $w$에 prior belief에 대한 solution을 regularizing하는 것
2) loss function을 통해 특정한 state-action pairs를 poor choices으로 표시하는 것. algorithm이 이러한 요소로 인하여 large magin을 가지도록 강제합니다.
3) $w$에 constraint 형태로 domain knowledge를 포함시키는 것 (e.g 특정한 영역의 state를 다른 state보다 cost가 적어도 두 배가 되도록 요구.)

이러한 방법들은 training example의 사용 외에도 learner에게 expert의 knowledge를 전달하는 강력한 방법입니다.

<br><br>

# 5. Experimental Results

<br>
## 5.1 Demonstration of learning to plan based on satellite color imagery

실험에서는 실제 문제(Path planning)에서 논문 개념을 이용하여 유효성 검증할 것입니다.
1) Section 4에서 보여주었던 batch learning algorithm을 사용
2) Regularization을 위한 적당한 값을 사용하고, 위에서 다뤘던 우리의 algorithm을 사용

추가적으로 prior knowledge에서의 첫 번째 방법을 적용한 것으로 보입니다.

같은 맵의 영역에서 시연되는 다른 예제 경로는 학습 후에 hold out 영역에서 상당히 다른 결과를 이끌었습니다.

<center> <img src="../../../../img/irl/mmp_8.png" width="1200"> </center>

<center> <img src="../../../../img/irl/mmp_9.png" width="600"> </center>

- 실험 의도
  - Top : Road에 유지하도록 제안
  - Bottom : 은밀한 의도를 제시(여기서는 숲을 지나는 의도를 의미)

- 실험 결과
  - Left : Training 예제
  - Middle : Training 이후에 hold out 영역에서 학습된 cost map
  - Right : Hold out 영역에서 $A^*$를 이용하여 생성된 행동 결과

<br>
## 5.2 Data shown are MMP learned cost maps

<center> <img src="../../../../img/irl/mmp_10.png" width="1100"> </center>

Figure 2은 holdout region으로부터의 결과입니다.

그림에서 loss-augmented path (blue)은 일반적으로 마지막 학습된 경로보다 일반적으로 좋은 결과를 수행하지 못하는 것을 나타났습니다.

왜냐하면 loss-augmentation은 높은 loss의 영역을 최종 학습 지도보다 더욱 desirable하게 만들기 때문입니다. 

직관적으로, 만약 학습자가 loss-augmented cost map에 대해서 잘 수행할 수 있다면, loss-augmentation없이도 더욱 잘 수행되어야 한다는 것입니다. 이것은 margin을 가지고 학습된 개념입니다.

<br>
## 5.3 Results using two alternative approaches

<center> <img src="../../../../img/irl/mmp_11.png" width="800"> </center>

- Left : the result of a next-action classifier applied superimposed on a visualization of the second dataset.
- Right : a cost map learned by manual training of a regression.

두 개의 경우에서 학습된 경로들은 poor approximations. (not shown on left, red on right).

<br>
## 5.4 Visualization about losses

<center> <img src="../../../../img/irl/mmp_12.png" width="850"> </center>

- Left : Visualization of inverted Loss function $(1− l(x))$ for a training example path.
- Right : Comparison of holdout loss of MMP (by number of iterations) to a regression method where a teacher hand-labeled costs.

비교를 위해, 저자는 MMP에 다른 두 개의 접근방법을 사용하여 유사한 학습을 시도하였습니다.

1. (Lecun et al., 2006)$^5$에서 제안한 방법으로 state 특징들을 다음 action으로 취하는 mapping을 사용한 직접적으로 학습하는 알고리즘입니다. 이 경우, traing data에 대해 좋은 결과를 얻지 못했습니다.

2. 다소 더 성공적은 시도는 직접 label을 통해 cost를 학습시킨 알고리즘입니다. 이 알고리즘은 MMP보다 학습자에게 더 많은 명시적 정보를 제공합니다.
   - 다음을 기반하여 low, medium, high cost로 제공
     1. Expert knowledge of the planner
     2. Iterated training and observation
     3. The trainer had prior knowledge of the cost maps found under MMP batch learning on this dataset.
   - 추가 정보가 주어진 cost map은 정성적으로 올바른 것처럼 보이지만, 그림 3과 그림 4는 상당히 좋지 않은 성능을 보여줍니다..

<br><br>

# 6. Related and Future Work

Maximum Margin Planning과 직접적으로 연관된 두 가지 work가 있습니다.

그 중 하나가 바로 Inverse RL입니다.

IRL의 목표는 MDP에서 agent의 행동을 관찰하는 하여 agent의 행동으로 부터 reward function를 추출하는 것입니다. 그러나 이것은 기본적으로 ill-posed 문제로 알고 있습니다. 그럼에도 불구하고, MMP와 같이 유사한 효과를 가진 IRL 아이디어들을 시도한 몇 가지 heuristic 시도가 있었습니다.

유용한 heuristic 방법은 바로 이전 논문인 *Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning* APP 논문입니다. 학습자의 정책과 시연되는 예제간의 expected feature counts을 통해 매칭을 시도하는 학습 방법입니다.

MMP의 경우 IRL algorithm의 variant과는 다른 스타일의 algorithm입니다.

MMP는 하나의 MDP 보다 많은 정책 시연들을 허용하도록 설계되어 있습니다. 여러 특징 맵, 다른 시작 지점과 목표 지점, 완전히 새로운 맵과 목표 지점을 가지는 학습자의 목표를 이용하여 예제들을 시연했습니다. 또한 MMP는 상당히 다른 IRL적 알고리즘 접근 방법을 유도합니다.

IRL과 MMP간의 관계는 **generative and discriminative learning** 간의 구별을 연상시킵니다.

일반적인 IRL의 경우, feature matching을 시도합니다. agent가 MDP에서 (거의 최적같이) 행동하고 (거의) feature expectation에 매칭 가능할 때 학습하도록 설계되었습니다(Generative models과 같은 strong 가정). 예를 들어, feature expecatation을 매칭하는 능력은 algorithm의 행동이 feature가 선형인 모든 cost function에 대해서 near-optimal일 것이라는 것을 의미합니다.

반대로 MMP의 경우, 우리의 목표가 직접적으로 output 행동을 모방하는 것이라는 weaker 가정을 하고 실제 MDP나 reward 함수에 대해 agnostic합니다. 여기서 MDP는 output decision들을 구조화하고 경쟁을 하려고 하는 expert가 natual class을 제공합니다.

정리하자면,
Generative model : 개별 클래스의 분포를 모델링한다.
Discriminative model : Discriminative 모델은 기본 확률 분포 또는 데이터 구조를 모델링하지 않고 기본 데이터를 해당 클래스에 직접 매핑(class 경계를 통해 학습). SVM은 이러한 기준을 만족 시키므로 decision tree와 마찬가지로 discriminative model.

<br><br>

# 처음으로

## [Let's do Inverse RL Guide](https://reinforcement-learning-kr.github.io/2019/01/22/0_lets-do-irl-guide/)

<br>

# 이전으로

## [APP 여행하기]()

<br>

# 다음으로

## [MaxEnt 여행하기]()