---
title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
date: 2018-07-10 14:18:32
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 김동민, 이동민
subtitle: 피지여행 1번째 논문
---

---
# 1. Intro to Policy Gradient

이 논문은 policy gradient (PG) 기법의 효시와도 같으며 향후 많은 파생연구를 낳은 중요한 논문입니다. 이 논문을 이해하기 위해 필요한 배경지식을 먼저 설명하고 논문을 차근차근 살펴보도록 하겠습니다.
<br>

## 1.1 Value Function Approach
전통적으로 강화학습 기법은 value function을 기반으로 동작하였습니다. 특정 state에서 vaue function 또는 value function을 근사하는 함수(function approximation)를 최대화하는 action을 찾는 greedy action-selection policy가  대표적입니다.

논문에서는 이러한 방법은 deterministic한 policy를 찾는 쪽으로 나아가게 되지만 종종 최적의 policy는 stochastic한 성질을 가지기 때문에 이 방법으로는 최적의 policy를 찾을 수 없다고 언급하고 있습니다. (이 논문이 나오고 나서 한참 후 개발된 deterministic policy gradient 기법과는 반대되는 서술일 수 있습니다.) 아마도 이러한 부분은 $\epsilon$-greedy action-selection policy로 개선될 수 있을 것입니다. 또 하나의 문제는 value function의 작은 변화로 인해서 action이 크게 변할 수도 있다는 것입니다. 이것은 알고리듬의 수렴성에 문제를 야기할 수 있습니다. 이러한 문제점을 해결하기 위하여 policy search라는 새로운 기법이 고안됩니다.
<br>

## 1.2 Policy Search
policy search는 최적의 policy $\pi^*$를 직접 찾기 때문에 value function 모델링이 필요가 없습니다. (뒤에서보겠지만 사실 꼭 그렇지만도 않습니다.) policy는 좀 더 다루기 쉽도록 parameter $\theta$를 이용하여 $\pi_{\theta}$로 표현할 수 있습니다. 최적의 policy를 찾기 위하여 expected return $E\left[R|\theta\right]$이 최대화되도록 parameter를 조정합니다. 이를 간단한 수식으로 정리하면 다음과 같습니다.

$$
\pi^* = \arg \max\limits_\pi  E\left[ {\left. R \right|\pi } \right] \to {\text{original problem} } 
\\\\  {\Downarrow} \\\\
{\text{policy parameterization by } } \pi_{\theta}: \Theta  \to \Pi \\\\ {\Downarrow} $$

$$\pi^* = \arg \max\limits_{\theta}  E\left[ {\left. R \right|\theta } \right] \to {\text{policy search problem} }
$$

policy search 방법은 크게 두 가지로 분류할 수 있습니다. 첫 번째는 gradient-based optimization으로 policy gradient method가 이에 속합니다. 두 번째는 gradient-free optimization으로 진화(evolutionary) 연산을 이용하는 것이 이에 속합니다.
<br>

## 1.3 How to Obtain the Expected Return
expected return을 최대화하는 방향으로 policy를 update한다고 하였습니다. 그렇다면 expected return은 어떻게 구할 수 있을까요? 여기에도 크게 두 가지 방법이 있습니다. 첫 번째는 deterministic approximation으로 Markov decision process의 dynamics를 모델링한 후 수식적으로 구하는 것입니다. 두 번째 방법은 monte carlo estimation으로 dynamics에 대한 모델을 하지 않고 많은 sample들을 얻은 후 empirical하게 expected return을 계산하는 방법입니다. 두 번째 방법이 좀 더 현실적이지만 gradient를 구하는 것은 더 어렵습니다. 결국 gradient를 esimate하는방법을 고안해야 합니다. 가장 유명한 gradient estimation 방법이 1992년 R. J. Williams에 의해서 제안된 REINFORCE 기법입니다. REINFORCE는 Monte Carlo estimate 또는 likelihood-ratio estimate라고 부르는 방법을 이용해서 gradient와 expectation의 위치를 서로 바꿔줍니다.

### Monte Carlo Gradient Estimation
다음과 같은 parameter $\theta$를 가지는 random variable $X$가 있습니다: $X:\Omega\mapsto\mathcal{X}$. 그리고 이 $x$에 대한 함수 $f$가 있습니다: $f:\mathcal{X}\mapsto\mathbb{R}$. expected return처럼 $E[f(x)]$를 최대화하고자 합니다. 이를 위해서는 $\nabla_{\theta}E[f(x)]$를 구해야 합니다. 이 때 다음과 같은 log derivate trick을 이용하여 다음과 같이 수식을 변형시킬 수 있습니다.

$$
\begin{align}
\nabla_{\theta}E_{p(x;\theta)}\left[ {f(x)} \right] 
&= \nabla_\theta\int{f\left( x \right)p\left( {x;\theta} \right) dx}
\\\\
&= \int { {\nabla_\theta }p\left( {x;\theta } \right)f\left( x \right)dx} 
\\\\
&= \int {\frac{ {p\left( {x;\theta } \right)} }{ {p\left( {x;\theta } \right)} }{\nabla_\theta }p\left( {x;\theta } \right)f\left( x \right)dx} 
\\\\
&= \int {p\left( {x;\theta } \right){\nabla_\theta }\log p\left( {x;\theta } \right)f\left( x \right)dx} 
\\\\
&= E_{p(x;\theta)}[f(x)\nabla_\theta \log p({x;\theta})]
\end{align}
$$

이제 expectation은 많은 샘플들을 모아서 평균을 취하는 Monte Carlo 기법으로 근사화할 수 있습니다.

$${\nabla_{\theta} }{E_{p\left( {x;\theta } \right)} }\left[ {f\left( x \right)} \right] = \frac{1}{N}\sum_{n=1}^N f\left(X_n\right)\nabla_\theta\log p\left(X_n;\theta\right)$$

이 방법을 사용하기 위해서 필요한 조건은 $\log p\left(X_n;\theta\right)$가 미분가능해야 한다는 것 뿐입니다. 그러나 이 방법은 얻은 샘플들에 의존하기 때문에 경우에 따라서 큰 variance를 가질 수 있습니다.

<br><br>

---
# 2. Policy Gradient Methods
이제 본격적으로 policy gradient 기법에 대해서 알아보겠습니다.

## 2.1 Policy Gradent Approach
policy gradient는 stochastic policy를 자체적인 파리미터를 가진 function approximator를 이용해서 근사화시킵니다. 몇 가지 notation들을 정의하겠습니다.

* $\theta$: policy parameter들의 vector  
* $\rho$: 해당 policy들의 성능을 나타내는 척도 (예. average reward per step)  
* $\alpha$: positive-definite한 step size 

policy parameter는 gradient에 비례하여 업데이트됩니다.

$$\Delta\theta \approx \alpha\frac{\partial\rho}{\partial\theta}$$

local optimal policy로 수렴합니다. 논문의 가장 중요한 contribution은 특정 조건을 만족하는 function approximator를 이용하여 경험(experience, sample)을 축적하고 이것들을 이용하여 위의 gradient를 unbiased estimate할 수 있다는 것을 증명한 것입니다.
<br>

## 2.2 System Model
시스템모델은 다음과 같습니다.

* Markov decision process (MDP) at each time $t\in{0,1,2,...}$
* $S_t\in\mathcal{S}$: state
* $A_t\in\mathcal{A}$: action
* $r_t\in E$: reward
* $\theta\in\mathbb{R}^l$: vector of policy parameters, for $l\ll\left|\mathcal{S}\right|$
* $\mathcal{P}_{s s'}^a = \Pr[S_{t+1}=s'|S_t=s,A_t=a]$: state transition probabilities
* $\mathcal{R}_{s s'}^a = E[R_{t+1}|S_t=s,A_t=a]$: expected reward
* $\pi(s,a,\theta)=\Pr[A_t=a|S_t=s,\theta]$: policy, shortened as $\pi(s,a)$ or $\pi(a|s)$

다음으로는 reward를 표현하는 두 가지 방법에 대해서 알아보겠습니다.
<br>

## 2.3 Average Reward Formulation
Average rewward formulation은 시간의 흐름에 따른 reward를 표현한다기보다는 모든 시간의 reward를 평균을 내서 표현하는 방법입니다.

* Long‐term expected reward per step
$$\rho(\pi)=\lim_{n\to\infty}\frac{1}{n}E[r_1+r_2+\cdots+r_n|\pi] = \sum_s d^\pi(s)\sum_a\pi(s,a)\mathcal{R}_s^a$$

    - $d^\pi(s): \lim_{t\to\infty}\Pr[S_t=s|s_0,\pi]$: stationary distribution of states under $\pi$
    - $d^\pi(s)$가 존재한다고 가정합니다. 그리고 모든 policy에 대해서 $d^\pi(s)$는 $s_0$와 independent합니다.
    - 위의 두 표현식은 왜 같을까요? Time average와 Ensemble average가 같다는 뜻으로 ergodic한 system에서 성립합니다.

* Value of a state-action pair given a policy
$$Q^{\pi(s,a)} = \sum_{t=1}^\infty E[r_t - \rho(\pi)|S_0=s, A_0=a, \pi]$$
* State-valu function
$$V^\pi(s) = \sum_{a} \pi(s,a)Q^\pi(s,a)$$
<br>

## 2.4 Start-State Formulation
start-state formulation은 시간의 흐름에 따라 감소하는 reward들을 표현합니다.

* Long‐term expected reward per step with a designated start state $S_0$
$$\rho(\pi) = E\left[\left.\sum_{t=1}^\infty \gamma^{t-1}r_t\right|S_0,\pi\right]$$
    - $\gamma\in\left[0,1\right]$: discount rate

* Value of a state‐action pair given a policy
$$Q^\pi(s,a) = E\left[\left.\sum_{k=1}^\infty \gamma^{k-1} r_{t+k}\right|S_t=s, A_t=a, \pi\right]$$

* Alternative form of $Q^\pi(s,a)$
$$
Q^\pi(s,a) =R_s^a + \gamma\sum_{s'}P_{s s'}^a V^{\pi}(s')
$$
<br>

## 2.5 Policy Gradient Theorem
논문의 중요한 결과인 Theorem 1은 다음과 같습니다.

*For any MDP, in either the average‐reward or start‐state formulations,*
$$\frac{\partial\rho}{\partial\theta}=\sum_s d^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)$$

놀라운 부분은 expected return의 gradient를 취할 때 $\frac{\partial d^\pi(s,a)}{\partial\theta}$를 구하지 않아도 된다는 점입니다. 즉, policy의 변화가 state distribution에 영향을 주지 않는다는 것입니다. 이것은 다시 말하면 아래 수식과 같이 우리는 $\frac{\partial\rho}{\partial\theta}$에 대한 unbiased estimator를 구할 수 있다는 뜻입니다.

$$E_s\left[\sum_a\frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)\right] = \frac{\partial\rho}{\partial\theta}$$

이로인해서 우리는 gradient를 sampling을 통해서 추정할 수 있습니다. 그렇지만 이것은 sample이 아주 많을 때만 성립합니다!

또 한 가지 문제는 우리는 $Q^\pi(s,a)$의 정확한 값을 알 수 없다는 것 입니다. 이것을 estimate하기 위해서 현재의 return값 $R_t$를 이용할 수 있습니다.
<br>

## 2.6 Proof of Policy Gradient Theorem
증명을 살펴보겠습니다.

* average-reward formulation을 이용할 때의 증명은 아래와 같습니다.

$$
\begin{align}
\frac{\partial V^\pi(s)}{\partial\theta} &\overset{\underset{\mathrm{def} }{} }{=}\frac{\partial}{\partial\theta}\sum_a \pi(s,a)Q^\pi(s,a)
\\\\
&=\sum_a \frac{\partial}{\partial\theta}\left[\pi(s,a)Q^\pi(s,a)\right]\\\\
&=\sum_a \left[\frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a)\frac{\partial Q^\pi(s,a)}{\partial\theta}\right]\\\\
&=\sum_a \left[\frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a)\frac{\partial}{\partial\theta}\left[R_s^a -\rho(\pi) + \sum_{s'}P_{s s'}^a V^{\pi}(s')\right]\right]\\\\
&=\sum_a \left[\frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a)\left[-\frac{\partial\rho}{\partial\theta} + \sum_{s'}\mathcal{P}_{s s'}^a \frac{\partial V^{\pi}(s')}{\partial\theta}\right]\right]\\\\
\frac{\partial\rho}{\partial\theta}&=\sum_a \left[\frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a)\sum_{s'}\mathcal{P}_{s s'}^a \frac{\partial V^{\pi}(s')}{\partial\theta}\right] - \frac{\partial V^\pi(s)}{\partial\theta}
\end{align}
$$

양변에 stationary distribution에 대한 평균을 취합니다.
$$
\begin{align}
\sum_s d^\pi(s)\frac{\partial\rho}{\partial\theta}&=\sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a)+ \sum_s d^\pi(s)\sum_a\pi(s,a)\sum_{s'}\mathcal{P}_{s s'}^a \frac{\partial V^{\pi}(s')}{\partial\theta} 
\\\\
&- \sum_s d^\pi(s)\frac{\partial V^\pi(s)}{\partial\theta}
\end{align}
$$

모든 state $s$에서 모든 state $s'$으로 이동하는 것은 state $s'$에 존재할 확률이므로 다음과 같이 표현할 수 있습니다.

$$
\begin{align}
\sum_s d^\pi(s)\frac{\partial\rho}{\partial\theta}&=\sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a)+ \sum_s d^\pi(s') \frac{\partial V^{\pi}(s')}{\partial\theta} - \sum_s d^\pi(s)\frac{\partial V^\pi(s)}{\partial\theta}
\end{align}
$$

뒤의 두 항은 같은 식이므로 이것은 다시 다음과 같이 표현됩니다.
$$
\begin{align}
\sum_s d^\pi(s)\frac{\partial\rho}{\partial\theta}&=\sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a)
\end{align}
$$

$\frac{\partial\rho}{\partial\theta}$는 $s$에 대한 함수가 아니므로 $\sum_s d^\pi(s)\frac{\partial\rho}{\partial\theta}=\frac{\partial\rho}{\partial\theta}$입니다.
이를 이용해서 위의 식을 다시 표현하면 아래와 같고 이를 통해 증명이 완성됩니다.

$$
\begin{align}
\therefore\frac{\partial\rho}{\partial\theta}&=\sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a)
\end{align}
$$

다음으로 start-state formulation에 대한 증명입니다.

* start-state formulation은 이 논문에서도 나오고 서튼책에서도 나오는데, 기존의 논문과 다르게 좀 더 진행되는 부분이 있습니다. 그래서 논문에 있는 증명이 아닌 서튼책에 있는 증명은 아래와 같습니다. (서튼책 [Link](https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view))

$$\nabla{v_\pi}(s)=\nabla\big[\sum_a\pi(a|s)q_\pi(s,a)\big]$$ 

다음으로 $(f\cdot g)'=f'\cdot g+f\cdot g'$ (product rule of calculus)에 의해 다음과 같이 나타낼 수 있습니다.
$$=\sum_a\big[\nabla\pi(a|s)q_\pi(s,a)+\pi(a|s)\nabla q_\pi(s,a)\big]$$  

이어서 $p(s',r|s,a)\cong Pr[S_t=s', R_t=r|S_{t-1}=s,A_{t-1}=a ]$ (서튼책 3.2 공식) 에 의해 다음과 같이 나타낼 수 있습니다.

$$=\sum_a\big[\nabla\pi(a|s)q_{\pi(s,a)}+\pi(a|s)\nabla\big[\sum_{s', r}p(s',r|s,a)(r+v_{\pi}(s'))\big]\big]$$  

계속해서 $p(s'|s,a)\cong Pr[S_t=s', R_t=r|S_{t-1}=s,A_{t-1}=a ]=\sum_{r\in R}p(s',r|s,a)$ (서튼책 3.4 공식) 에 의해 다음과 같이 나타낼 수 있습니다.
$$=\sum_a\big[\nabla\pi(a|s)q_\pi(s,a)+\pi(a|s)\sum_{s'}p(s'|s,a)\nabla v_\pi(s') \big]$$ 

여기서 $\nabla v_\pi(s')$ 부분을 unrolling을 하면 다음과 같이 나타낼 수 있습니다.
$$=\sum_a\big[\nabla\pi(a|s)q_\pi(s,a)+\pi(a|s)\sum_{s'}p(s'|s,a)\sum_{a'}\big[\nabla\pi(a'|s')q_\pi(s',a')\\\\+\pi(a'|s')\sum_{s''}p(s''|s',a')\nabla v_\pi(s''))\big] \big]$$ 

unrolling을 반복하다보면 다음과 같이 나타낼 수 있습니다.
$$=\sum_{x\in S}\sum_{k=0}^{\infty}Pr(s\rightarrow x,k,\pi)\sum_a \nabla \pi(a|s)q_\pi(s,a)$$

여기에서 $Pr(s\rightarrow x,k,\pi)$ 는 policy $\pi$ 에 대해 state s에서 state x까지 k step만큼 움직일 때의 변환 확률입니다. 따라서 위의 수식을 아래처럼 다시 나타낼 수 있습니다.
$$ \begin{align} \nabla J(\theta)= &\nabla v_\pi (s_0) \\ =&\sum_s(\sum_{k=0}^{\infty}Pr(s_0\rightarrow s,k,\pi))\sum_a \nabla \pi(a|s)q_\pi(s,a) \\ =&\sum_s \eta(s) \sum_a \nabla \pi(a|s)q_\pi(s,a) \end{align}$$

- 서튼책에서 $\eta(s)$는 다음과 같이 표현할 수 있습니다.
    - $\eta(s)$ = 평균적으로 하나의 episode마다 state에서 머무른 time step의 수 (page 199)
- 논문에서는 $\eta(s)$를 다음과 같이 표현합니다.
    - $d^\pi(s)\\\\=\sum_{t=0}^{\infty}\gamma^t Pr(s_t=s|s_0,\pi)$ ($\gamma=1$ is allowed only in episodic tasks) = discounted weighting of states

여기서부터가 논문에는 없는 내용입니다. 위의 최종 수식을 다시 한 번 적으면 아래와 같습니다.
$$\sum_s \eta(s) \sum_a \nabla \pi(a|s)q_\pi(s,a)$$

위의 수식을 아래의 수식처럼 바꿀 수 있습니다.
$$\sum_{s'} \eta(s') \cdot\sum_s \frac{\eta(s)}{\sum_{s'}\eta(s')} \sum_a \nabla \pi(a|s)q_\pi(s,a)$$

그러면 이것을 새로운 기호로 다시 나타낼 수 있습니다.
$$\sum_{s'} \eta(s') \cdot\sum_s \mu(s)\sum_a \nabla \pi(a|s)q_\pi(s,a)$$ 
- 여기서 $\mu(s)$ 는 state distribution이라고 하며, weight를 state distribution으로 바꿔주는 과정이라고 생각할 수 있습니다. 쉽게 말해 어떠한 state에 agent가 머무르는 확률입니다.

위의 수식은 아래의 수식과 비례합니다. 따라서 최종 형태는 다음과 같습니다.
$$\propto \sum_s \mu(s)\sum_a \nabla \pi(a|s)q_\pi(s,a)$$
<br><br>

---
# 3. Policy Gradient with Approximation
이번 section은 앞서 다뤘던 Theorem 1 중에 $Q^\pi$에 대해서 중점적으로 다룹니다. 어떠한 $Q^\pi$가 학습된 function approximator로 근사된다고 합시다. 

$f_w$ ($f_w$는 $S × A \rightarrow \Re$)를 parameter $w$를 가지는 어떠한 $Q^\pi$ 에 대한 근사치입니다. 학습된 function approximator에 대해서 생각하기 때문에, 학습된 $f_w$는 error를 최소화하는 방향으로 parameter $w$를 업데이트된 것을 의미합니다. 그러면 하나의 rule에 의해서 $\pi$를 따르고 $w$를 업데이트하는 $f_w$에 대해서 다음과 같이 생각할 수 있습니다.
$$\Delta w_t \propto \frac{\partial}{\partial w}\big[ {\hat Q}^\pi(s_t,a_t)-f_w(s_t,a_t) \big]^2 \propto \big[{\hat Q}^\pi(s_t,a_t)-f_w(s_t,a_t) \big]\frac{\partial f_w(s_t,a_t)}{\partial w}$$ 
(여기서 ${\hat Q}^\pi(s_t,a_t)$ 는 $Q^\pi(s_t,a_t)$의 unbiased estimator $R_t$입니다.)

그러면 위와 같은 수식이 local optimum에 수렴을 했을 때, 다음과 같이 나타낼 수 있습니다.
$$\sum_s d^\pi(s)\sum_a \pi(s,a)\big[ {Q}^\pi(s_t,a_t)-f_w(s_t,a_t) \big]\frac{\partial f_w(s,a)}{\partial w}=0$$
- 위의 수식에 대한 추가 설명
    - local optimum으로 수렴을 하기 때문에 당연히 위의 수식은 0이 됩니다.
    - 또한 stochastic policy이기 때문에 local optimum으로 수렴하려면 모든 state, action에 대해 expectation을 해야합니다. 따라서 $\sum_s d^\pi(s)\sum_a \pi(s,a)$이 붙게 됩니다.
<br>

## 3.1 Theorem 2: Policy Gradient with Function Approximation
만약 $f_w$가 아래의 등식을 만족한다고 합시다.
$$\sum_s d^\pi(s)\sum_a \pi(s,a)\big[ {Q}^\pi(s_t,a_t)-f_w(s_t,a_t) \big]\frac{\partial f_w(s,a)}{\partial w}=0$$

또한 $f_w$가 policy parameterization과 양립할 수 있다고 합시다. policy parameterization에 대한 여러가지 의미가 있는데 그 중에서 이 논문에서는 다음을 의미합니다.
$$\frac{\partial f_w(s,a)}{\partial w}=\frac{\partial \pi(s,a)}{\partial \theta}\frac{1}{\pi(s,a)}$$
- 위의 수식에 대한 추가 설명
    - 이 수식은 Monte-Calro Estimation을 이용하기 위한 조건입니다.
    - Compatibility Condition이라고 부릅니다.
    - 우변(= $\nabla\log\pi(s,a)$)으로 유도함으로써 아래의 수식과 같이 sampling을 이용한 추정이 가능해집니다.

따라서 위의 두 수식을 이용하여 아래와 같이 나타낼 수 있다.
$$\frac{\partial\rho}{\partial\theta}=\sum_s d^\pi(s)\sum_a \frac{\partial\pi(s,a)}{\partial\theta}f_w(s,a)$$
<br>

## 3.2 Proof of Theorem 2
$$\sum_s d^\pi(s)\sum_a \pi(s,a)\big[ {Q}^\pi(s_t,a_t)-f_w(s_t,a_t) \big]\frac{\partial f_w(s,a)}{\partial w}=0$$

$$\frac{\partial f_w(s,a)}{\partial w}=\frac{\partial \pi(s,a)}{\partial \theta}\frac{1}{\pi(s,a)}$$

위의 두 수식을 합치면 다음과 같이 나타낼 수 있습니다.
$$\sum_s d^\pi(s)\sum_a \pi(s,a)\big[ {Q}^\pi(s_t,a_t)-f_w(s_t,a_t) \big]\frac{\partial \pi(s,a)}{\partial \theta}\frac{1}{\pi(s,a)}=0$$

분자와 분모에 있는 $\pi(s,a)$를 지우면 아래와 같습니다.
$$\sum_s d^\pi(s)\sum_a \frac{\partial \pi(s,a)}{\partial \theta}\big[ {Q}^\pi(s_t,a_t)-f_w(s_t,a_t) \big]=0$$

이어서 위의 수식이 0이기 때문에 policy gradient theorem(앞서 다뤘던 Theorem 1)에서 위의 수식을 뺄 수 있습니다.
$$\frac{\partial\rho}{\partial\theta}=\sum_s d^\pi(s)\sum_a \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)-\sum_s d^\pi(s)\sum_a \frac{\partial\pi(s,a)}{\partial\theta}\big[ Q^\pi(s_t,a_t)-f_w(s_t,a_t) \big]$$

$$\frac{\partial\rho}{\partial\theta}=\sum_s d^\pi(s)\sum_a \frac{\partial\pi(s,a)}{\partial\theta}\big[ Q^\pi(s_t,a_t)-Q^\pi(s_t,a_t)+f_w(s_t,a_t) \big]$$

$$\therefore\frac{\partial\rho}{\partial\theta}=\sum_s d^\pi(s)\sum_a \frac{\partial\pi(s,a)}{\partial\theta}f_w(s_t,a_t)$$
<br><br>

---
# 4. Application to Deriving Algorithms and Advantages

## 4.1 Application to Deriving Algorithms
feature의 linear combination에서 Gibbs distribution (softmax)를 하나의 policy로 생각해볼 수 있습니다.
$$\pi(s,a)=\frac{e^{\theta^{T}\phi_{sa} }}{\sum_b e^{\theta^{T}\phi_{sb} }}$$
- ($\phi_{sa}$: state-action pair s,a을 나타내는 l-dimensional feature vector)

compatible condition을 추가하면 다음과 같습니다. (4.2 Proof of application of compatible condition 참고)
$$\frac{\partial f_w(s,a)}{\partial w}=\frac{\partial\pi(s,a)}{\partial\theta}\frac{1}{\pi(s,a)}=\phi_{sa}-\sum_b \pi(s,b)\phi_{sb}$$ 

이어서 $f_w$을 적분한 natural parameterization은 다음과 같습니다.
$$f_w(s,a)=w^T\big[ \phi_{sa}-\sum_b \pi(s,b)\phi_{sb}\big]$$
- 위의 수식에 대한 추가 설명
    - $f_w$는 policy로서 같은 feature들에 대해 linear합니다.
    - $f_w$는 각각의 state에 대해서 평균이 0입니다. ($\sum_a\pi(s,a)f_w(s,a)=0$)
    - advantage function $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$의 하나의 근사치로서 $f_w$를 생각하는 것 좋습니다.
<br>

## 4.2 Proof of application of compatible condition
증명을 하기 전에 다음을 가정합니다.
1. $(\frac{f(x)}{g(x)})'=f'(x)\cdot \frac{1}{g(x)}-f(x)\cdot \frac{g'(x)}{g(x)^{2} }=\frac{f'(x)\cdot g(x)-f(x)\cdot g'(x)}{g(x)^{2} }$
2. $f(\theta)=e^{\theta^{T}\phi_{sa} }$, $g(\theta)=\sum_be^{\theta^{T}\phi_{sb} }$

$$
\begin{align}
\frac{\partial f_w(s,a)}{\partial w}&=\frac{\partial\pi(s,a)}{\partial\theta}\frac{1}{\pi(s,a)}
\\\\
&=\big(\frac{e^{\theta^{T}\phi_{sa} }}{\sum_be^{\theta^{T}\phi_{sb} }}\big)'\frac{1}{\pi(s,a)}
\\\\
&=\big[ \phi_{sa}(\frac{e^{\theta^{T}\phi_{sa} }}{\sum_be^{\theta^{T}\phi_{sb} }})-(\frac{e^{\theta^{T}\phi_{sa} }}{\sum_be^{\theta^{T}\phi_{sb} }})\frac{\sum_b\phi_{sb}e^{ {\theta^T}\phi_{sb} }}{\sum_be^{\theta^{T}\phi_{sb} }}\big]\frac{1}{\pi(s,a)}\\\\
&=\big[ \phi_{sa}\pi(s,a)-\pi(s,a)\sum_b\phi_{sb}\pi(s,b)\big]\frac{1}{\pi(s,a)}
\\\\
&=\pi(s,a)\big[ \phi_{sa}-\sum_b\pi(s,b)\phi_{sb}\big]\frac{1}{\pi(s,a)}
\\\\
&\therefore\phi_{sa}-\sum_b\pi(s,b)\phi_{sb}
\end{align}
$$

<br>

## 4.3 Application to Advantages
Policy Gradient with Function Approximation Theorem (Theorem 2)는 advantage function으로 확장될 수 있습니다.
$$\frac{\partial\rho}{\partial\theta}=\sum_sd^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}\big[ f_w(s,a)+v(s) \big]$$
- 위의 수식에 대한 추가설명
    - $v$ ($v$는 $S\rightarrow\Re$) 는 arbitrary function입니다.
    - 이 수식은 $\sum_a\frac{\partial\pi(s,a)}{\partial\theta}=0$이기 때문에 가능해집니다.
        - 즉, 이 수식은 $\pi(s,a)$의 gradient에만 dependent하기 때문에 advantage 역할을 하는 함수들을 넣어도 아무런 상관이 없습니다.
    - $v$의 선택은 Theorem들에 영향을 미치지 못하지만, 실질적으로 gradient estimator의 variance에 영향을 미칩니다.
    - 이러한 문제는 전체적으로 이전의 연구에 reinforcement baseline의 사용에 있어서 유사합니다.
    - (comment) 위의 수식에서 $f_w(s,a)+v(s)$와 Application to Deriving Algorithms의 $f_w(s,a)$와는 다른 것입니다. Application to Deriving Algorithms의 $f_w(s,a)$은 softmax에 의해 스스로 advantage function의 역할을 할 수 있습니다. 하지만 보통의 경우에는 그러지 못할 수도 있기 때문에 위의 수식처럼 $f_w(s,a)+v(s)$을 추가하여 zero mean만들어서 variance를 줄일 수 있습니다.
<br><br>

---
# 5. Convergence of Policy Iteration with Function Approximation

## 5.1 Theorem 3: Convergence of Policy Iteration with Function Approximation

policy iteration with function approximation의 형태는 locally optimal policy에 수렴합니다. $\pi$와 $f_w$를 

compatibility condition을 만족하는 policy와 value function에 대해 $\max_{\theta, s, a, i, j}\big| \frac{\partial^2\pi(s,a)}{\partial\theta_i\partial\theta_j}\big|<B<\infty$ 을
만족하는 어떠한 미분가능한 function approximator라고 합시다.

(comment) $\pi(s,a)$라는 function은 이계미분값이 존재하고, 임의의 상수인 (Bound) B에 유계하기 때문에 function의 그래프는 smooth하다고 볼 수 있습니다. (아래의 그림 중 빨간색 그래프 참고)

<img src="https://www.dropbox.com/s/03w8nuar9v0atls/Screen%20Shot%202018-07-10%20at%203.41.06%20PM.png?dl=0" width=400px>

${\alpha_k}$는 $\lim_{k\rightarrow\infty}\alpha_k=0$ 이며 $\sum_k\alpha_k=\infty$를 만족하는 step-size sequence라고 합시다.

그 때, bounded reward를 가진 MDP에 대해
- 1. 어떠한 $\theta_0, \pi_k=\pi(\cdot,\cdot,\theta_k)$ 
- 2. $\sum_sd^{\pi_{k} }(s)\sum_a\pi_k(s,a)\big[ Q^{\pi_{k} }(s,a)-f_w(s,a)\big]\frac{\partial f_w(s,a)}{\partial w}=0$로 인하여 $w_k=w$, $\theta_{k+1}=\theta_k+\alpha_k\sum_sd^{\pi_{k} }(s)\sum_a\frac{\partial\pi_k(s,a)}{\partial\theta}f_{w_{k} }(s,a)$

으로 정의된 sequence $\rho(\pi_k)$는 $\lim_{k\rightarrow\infty}\frac{\partial\rho(\pi_k)}{\partial\theta}=0$이기 때문에 수렴합니다.
- sequence $\rho(\pi_k)_{k=0}^\infty$에 대한 추가 설명
    - $\theta_{k+1}=\theta_k+\alpha_k\sum_sd^{\pi_{k} }(s)\sum_a\frac{\partial\pi_k(s,a)}{\partial\theta}f_{w_{k} }(s,a)$ 에 따라 $\theta$가 1, 2, ..., $\infty$로 갈텐데, 거기에 따른 objective function or performance의 sequence입니다.
    - (comment) 굳이 sequence라는 표현이 없어도 될 것 같다. 어짜피 k가 $\infty$로 가면 $\rho(\pi_k)$가 수렴한다는 의미이기 때문에 불필요해보입니다.

## 5.2 Proof of Theorem 3
- Theorem 2는 $\theta_k$ update가 gradient의 error를 최소화한다는 것을 증명했습니다.
- $\frac{\partial^{2}\pi(s,a)}{\partial\theta_i\partial\theta_j}$와 MDP의 reward에서의 bound는, $\frac{\partial^{2}\rho}{\partial\theta_i\partial\theta_j}$ 또한 bound된다는 것을 증명합니다.
- step-size 필요조건 때문에 이러한 bound된 것들은 Proposition 3.5 from page 94 of Bertsekas and Tsitsiklis(1996)에 적용하기 위해 필요한 조건입니다.
- Proposition 3.5 from page 94 of Bertsekas and Tsitsiklis(1996)은 local optimum으로 수렴한다는 것을 증명했습니다.
<br><br>

---
# 6. Summary 

논문에서 설명한 policy gradient 기법을 요약하면 다음과 같습니다.

* Original maximization problem: $\max\rho_{\theta}=E\left[R\right]$
* gradient ascent method를 이용한 parameter update: $\theta_{t+1} = \theta_{t} + \alpha\left.\frac{\partial E[R]}{\partial\theta}\right|_{\theta_t}$
    - 그러나 우리는 gradient를 모르고, 추정하기도 어렵습니다. 왜냐하면 expectation이 안에 있기 때문입니다.
    - 그렇다면 이것을 추정 가능한 형태로 바꿔야합니다. 다시 말해 expectation이 밖에 있는 형태로 바꾸는 것입니다.
    - Expectation이 밖에 있으면 왜 추정이 유리할까요? 바로 Sample mean을 취하면 되기 때문입니다.
* Approximate gradient
    - 방법1: 논문의 Theorem 1을 이용하면 gradient와 expectation의 위치를 바꿀 수 있습니다.
    $$\frac{\partial E[R]}{\partial\theta}=\sum_s d^\pi(s)\sum_a\frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)$$
    - 방법2: Log derivative trick을 이용하면 gradient와 expectation의 위치를 바꿀 수 있습니다. (Theorem 2)
    $$\theta_{t+1} = \theta_{t} + \alpha E[R\frac{\partial}{\partial\theta}\log p_{\theta}|\theta_t]$$
    - 그리고 특정 trajectory를 따라가면서 return값을 구하고 이것을 여러 번 수행하여 sample mean을 취하면 gradient를 추정하는 것이 가능합니다.
    - 이러한 기법을 제시한 기존의 연구가 REINFORCE입니다.
    - 여러 Trajectory를 이용하므로 variance가 높을 수 밖에 없습니다.
    - Advantage를 활용하거나 하는 방식으로 이후 여러 연구가 진행되었습니다.
<br>
