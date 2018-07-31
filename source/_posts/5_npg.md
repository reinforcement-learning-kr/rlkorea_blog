---
title: Natural Policy Gradient
date: 2018-06-25 11:36:45
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 김동민, 이동민, 이웅원, 차금강
subtitle: 피지여행 4번째 논문
---

<center> <img src="https://www.dropbox.com/s/yd0x14ljrhpnj1b/Screen%20Shot%202018-07-18%20at%201.08.05%20AM.png?dl=1" width="600"> </center>

논문 저자 : Sham Kakade
논문 링크 : https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf
Proceeding : Advances in Neural Information Processing Systems (NIPS) 2002
정리 : 김동민, 이동민, 이웅원, 차금강

---

# 1. 들어가며...

많은 연구들이 objective function의 gradient 값을 따라서 좋은 policy $\pi$를  찾는 노력을 하고 있습니다. 하지만 기존의 Gradient descent rule을 따르는 업데이트 방식은 non-covariant 방식입니다. gradient descent rule은 $\bigtriangleup \theta_i = \alpha \dfrac{\partial f}{\partial \theta_i}$을 이용해서 파라미터를 업데이트 하는데 $\bigtriangleup \theta_i$가 $f$를 구성하고 있는 파라미터들에 종속되어 있다는 것입니다. 본 논문에서는 $f$를 구성하고 있는 파라미터들과 독립적인 metric을 정의함으로써 covariant한 gradient을 제시하고 이를 이용하여 파라미터를 업데이트합니다.

<br><br>

# 2. Notation & Natural Gradient

<br>
## 2.1 Notation

Finite한 MDP tuple을 먼저 정의합니다. $(S, s_0, A, R, P)$로 정의됩니다. Finite한 상태의 set$(S)$, 에피소드 시작 state $(s_0)$, finite한 행동의 set $(A)$, 보상 함수 $R: S \times A \to [0, R_{max}]$, 그리고 transition 모델 $(P)$로 이루어져 있습니다. 그리고 모든 정책 $\pi$는 ergodic하다고 가정합니다. 여기서 어떠한 통계적인 모델(모집단)과 모집단으로부터 추출된 모델(표본집단)이 같은 특성을 가질 때 ergodic하다고 합니다. 고등학교 수학에서 배우는 표본집단의 표준편차, 평균은 모집단의 그것과 같다는 것과 비슷한 개념입니다. 그리고 정책은 파라미터들 $\theta$에 의해 지배될 때 상태 $s$에서 행동 $a$를 선택할 확률로 정의되며 $\pi(a;s,\theta)$와 같이 표현됩니다. [Sutton PG](../../../06/28/sutton-pg)와 같이 $Q^\pi(s,a) = E_\pi (\Sigma^\infty_{t=0}R(s_t, a_t)-\eta(\pi)|s_0 = s, a_0 = a)$를 정의합니다.

다음 다른 정책 업데이트와 같이 objective function, $\eta(\pi_{\theta})$를 정의합니다.
$$
\eta(\pi_{\theta}) = \Sigma_{s,a} \rho^{\pi}(s)\pi(a;s,\theta)Q^\pi(s,a)
$$


여기서 $\eta(\pi_\theta)$는 $\pi_\theta$에 종속되어 있으며, $\pi_\theta$는 $\theta$에 종속되어 있습니다. 결국 $\eta(\pi_\theta)$는 $\theta$에 종속되어 있다고 할 수 있으며 $\eta(\pi_{\theta})$는 $\eta(\theta)$로 표현할 수 있습니다. 

<br>
## 2.2 Natural Gradient

2.1에 정의된 objective function을 최대화 하는 방향으로 파라미터들을 업데이트 하는 것이 목적입니다. Euclidean space에서는 objective function의 일반적인  policy gradient는 [Sutton PG](../../../06/28/sutton-pg)에서 논의한 바와 같이 다음과 같이 구할 수 있습니다.

$$\bigtriangledown\eta(\pi_\theta) = \Sigma_{s,a}\rho^\pi(s)\bigtriangledown\pi(a;s,\theta)Q^\pi(s,a)$$

하지만 뉴럴 네트워크에서 사용하는 parameter들은 매우 복잡하게 얽혀 있으며 이는 보통 생각하는 Euclidean space가 아닙니다. 일반적으로 이를 리만 공간(Riemannian space)라고 합니다. 리만 공간에서는 natural gradient가 steepest direction(업데이트 방향)이며 이를 구하는 방법은 [Amari, Natural gradient works in efficiently in learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf)에서 자세히 설명을 하고 있습니다. 이를 간단히 설명하면 다음과 같습니다. 파라미터로 구성된 Matrix의 Positive-Definite Matrix인 $G(\theta)$를 이용하여 구할 수 있는데 리만 공간에서의 steepest gradient(natural gradient)를 $\widetilde \bigtriangledown \eta(\theta)$라고 한다면 다음과 같이 표현할 수 있습니다.

$$\widetilde \bigtriangledown\eta(\theta)=G(\theta)^{-1}\bigtriangledown\eta(\theta)$$

위의 식을 이용하여 리만 공간에서 파라미터를 업데이트 하면 아래의 식과 같이 표현할 수 있습니다.

$$\theta_{k+1}=\theta_k-\alpha_tG(\theta)^{-1}\bigtriangledown\eta(\theta)$$

한 가지 생각해볼 점이 있습니다. 이 문제는 뉴럴넷을 이용하여 구성하고 해결할 수 있습니다. 뉴럴넷은 여러가지 파라미터셋들로 구성될 수 있습니다. 하지만 우연의 일치로 다른 파라미터셋을 가지지만 같은 policy를 가질 수 있습니다. 이 경우 steepest  direction은 같은 policy이기 때문에 같은 방향을 가리키고 있어야 하는데 non-covariant한 경우 그렇지 못합니다. 이 이유로 느린 학습이 야기됩니다. 이 문제를 해결하기 위해 단순히 Positive-Definite Matrix $G(\theta)$를 사용하지 않고 [Fisher Information Matrix](arxiv.org/abs/1707.06347)(이하 FIM) $F_s(\theta)$를 사용합니다. FIM는 어떤 확률 변수의 관측값으로부터 확률 변수의 분포의 매개변수에 대해 유추할 수 있는 정보량입니다. 어떠한 확률변수 $X$가 미지의 매개변수 $\theta$에 의해 정의되는 분포를 따른다고 하면 $X=x$일때 FIM은 다음과 같이 정의됩니다.

$$F_x(\theta)=E\left[\left(\dfrac{\partial}{\partial\theta}\log\Pr(x|\theta)\right)^2\right]$$

강화학습에서는 정보 $x$는 에피소드에 의해 관측된 상태값 $s$이며 매개변수 $\theta$에 의해 선택될 수 있는 행동에 대한 분포가 나오게 됩니다. 이에 의해 위의 FIM는 다음과 같이 표현할 수 있습니다.

$$F_s(\theta) \equiv E_{\pi(a;s,\theta)}\left[\left(\dfrac{\partial}{\partial\theta}\log\pi(a;s,\theta)\right)^2\right] =E_{\pi(a;s,\theta)}\left[\dfrac{\partial \log\pi(a;s,\theta)}{\partial \theta_i}\dfrac{\partial \log\pi(a;s,\theta)}{\partial\theta_j}\right]$$

그리고 위의 식들을 이용하여 objective function을 정리하면 아래의 식과 같이 표현됩니다.

$$F(\theta) = E_{\rho^{\pi}(s)}[F_s(\theta)]$$

또한 이 Fisher Information Matrix에 정의된 이 metric은 다음과 같은 성질을 가지고 있습니다. 

1) 업데이트되는 파라미터에 의해 구성되는 Manifold(곡률을 가지는)에 기반한 metric입니다. 
2) 확률분포(본 논문에서는 $\pi(a;s,\theta)$)를 구성하는 파라미터($\theta$)의 변화에 독립적입니다. 어떠한 coordinate를 선택하느냐에 따라 변화하지 않습니다. 
3) 마지막으로 positive-definite한 값을 가집니다. 이렇기 때문에 steepest gradient에서 Objective function의 방향을 알기 위해 사용한 방법과 같은 방법으로 Natural gradient direction을 다음과 같이 구할 수 있습니다.

$$\widetilde{\bigtriangledown}\eta(\theta) \equiv F(\theta)^{-1}\bigtriangledown\eta(\theta)$$

<br><br>

# 3. Natural Gradient와 Policy Iteration

3장에서는 Natural gradient를 통한 policy iteration을 수행하여 실제로 정책의 향상이 있는지를 증명합니다. 본 논문에서 $Q^\pi(s,a)$는 compatible function approximator $f^\pi(s,a;w)$로 근사됩니다.

<br>
## 3.1 Compatible Function Approximation

3.1절에서는 정책이 업데이트 될 때, value function approximator, $f^\pi(s,a;w)$도 같이 실제 값과 가까워지는지를 증명합니다.

본 증명을 수행하기 전에 몇가지 가정과 정의가 필요합니다.

파라미터들의 집합, $\theta$와 선형 행렬인 $\omega$를 이용하여 다음을 정의합니다.
$$
\psi^{\pi}(s,a) = \bigtriangledown \log\pi(a;s,\theta), f^\pi(s,a;\omega) = \omega^T\psi^\pi(s,a)
$$
여기서, $[\bigtriangledown \log\pi(a;s,\theta)]_i=\partial \log\pi(a;s,\theta)/\partial\theta_i$입니다. 그리고 compatible function approximator는 실제 $Q^\pi(s,a)$에 가까워져야 하므로 오차 $\epsilon(\omega,\pi)=\Sigma_{s,a}\rho^{\pi}(s)\pi(a;s,\theta)(f^\pi(s,a;\omega)-Q^\pi(s,a))^2$가 정의됩니다. 그리고 $\widetilde \omega$는 $\epsilon(\omega, \pi_\theta)$를 최소화하는 $\omega$라고 가정합니다.

최종적으로 $\widetilde \omega = \widetilde \bigtriangledown \eta(\theta)$이면 function approximator의 gradient 방향과 정책의 gradient 방향이 같다는 의미입니다.

증명) $\widetilde \omega$ 가 squared error $\epsilon(\omega, \pi)$를 최소화 한다면, $\dfrac{\partial\epsilon}{\partial\omega_i}=0$입니다.
$$
\dfrac{\partial}{\partial\omega_i}\Sigma_{s,a}\rho^\pi(s)\pi(a;s,\theta){(\color{red}{f^\pi(s,a;w)}-Q^\pi(s,a))^2}=0
$$

$$
\dfrac{\partial}{\partial\omega_i}\Sigma_{s,a}\rho^\pi(s)\pi(a;s,\theta)(\color{red}{\psi^\pi(s,a)\widetilde\omega}-Q^\pi(s,a))^2 = 0
$$

$$
\Sigma_{s,a}\rho^\pi(s)\pi(a;s,\theta)\psi^\pi(s,a)\color{blue}{(\psi^\pi(s,a)^T\widetilde\omega-Q^\pi(s,a))} = 0
$$

$$
\Sigma_{s,a}\rho^\pi(s)\pi(a;s,\theta)\psi^\pi(s,a)\color{blue}{\psi^\pi(s,a)^T\widetilde\omega}=\Sigma_{s,a}\rho^\pi(s)\pi(a;s,\theta)\psi^\pi(s,a)\color{blue}{Q^\pi(s,a)}
$$

$$
\color{green}{\psi^\pi(s,a)\psi^\pi(s,a)^T} = \bigtriangledown \log\pi(a;s,\theta) \bigtriangledown \log\pi(a;s,\theta) = \color{green} {F_s(\theta)}
$$

$$
\Sigma_{s,a}\rho^\pi(s)\pi(a;s,\theta)\psi^\pi(s,a)\psi^\pi(s,a)^T= E_{\rho^\pi(s)}[F_s(\theta)] = F(\theta)
$$

$$
\widetilde\bigtriangledown\eta(\theta)\equiv F(\theta)^{-1}\bigtriangledown\eta(\theta)
$$

위의 수식을 따라 정리가 됩니다.

<br>
## 3.2 Greedy Policy Improvement

natrual gradient는 다른 policy iteration 방법처럼 단지 현재보다만 좋은 방법을 선택하도록 업데이트하는 것이 아니라 현재보다 좋은 것 중에 가장 좋은 방법을 선택 합니다. 특수한 정책을 가지는 상황안에서 학습속도($\alpha$)를 무한대로 가져감으로써 어떤 action을 선택하는지를 알아봅니다.

3.2절에서는 정책이 다음과 같은 형태를 가진다고 가정합니다.
$$
\pi(a;s,\theta) \propto \exp(\theta^T\phi_{sa})
$$
그리고 3.1절처럼 $\widetilde\omega$가 approximation error인 $\epsilon$을 최소화하는 파라미터라고 가정하지만 gradient마저 최소화되게 하는 파라미터는 아니라고 가정합니다.($\widetilde\bigtriangledown\eta(\theta)\neq0$)

그리고 학습속도가 무한대일 때의 정책의 notation을 다음과 같이 표현합니다.
$$
\pi_\infty(a;s) = lim_{\alpha\rightarrow\infty}\pi(a;s,\theta + \alpha\widetilde\bigtriangledown\eta(\theta))
$$
3.1절의 결과를 이용하여 function approximator($f^\pi(s,a;\omega)$)는 다음과 같이 쓸 수 있습니다.
$$
f^\pi(s,a;\omega) = \widetilde\bigtriangledown\eta(\theta)^T\psi^\pi(s,a)
$$
그리고 가정에 의해 다음과 같이 표기될 수 있습니다.
$$
\psi^\pi(s,a) = \phi_{sa}-E_{\pi(a' ;s,\theta)}[\phi_{s a'}]
$$
다음 function approximator는 다음과 같이 다시 쓸 수 있습니다.
$$
f_\pi(s,a;\omega) = \widetilde\bigtriangledown\eta(\theta)^T(\phi_{s a}-E_{\pi(a',s,\theta)}[\phi_{s a'}])
$$
Greedy improvement와 같이 function approxmiator의 가장 높은 행동을 선택하는 것을 정책으로 합니다.($\arg\max_{a'}f^\pi(s,a)=\arg\max_{a'}\alpha\widetilde\bigtriangledown\eta(\theta)^T\phi_{s a'}$)

그리고 정책의 업데이트는 다음과 같이 이루어집니다.
$$
\pi(a;s,\theta+\alpha\widetilde\bigtriangledown\eta(\theta)) \propto exp(\theta^T\phi_{sa}+\alpha\widetilde\bigtriangledown\eta(\theta)^T\phi_{sa})
$$
3.2절 초입의 가정($\widetilde\bigtriangledown\eta(\theta)\neq0$)에 의해 위의 식에서 $\alpha$가 무한대로 간다면 우측 식에서 $\theta^T\phi_{sa}$의 항은 다른 항에 비해 매우 작은 값을 가지게 되므로 $\alpha\widetilde\bigtriangledown\eta(\theta)^T\phi_{sa}$의 지배를 받게 됩니다.

그러므로 $\pi_\infty=0$ 과 $a∉\arg\max_a\widetilde\bigtriangledown\eta(\theta)^T\phi_{sa'}$은 필요충분조건이 됩니다. 이것은 NPG가 단지 더 좋은 행동이 아니라 가장 좋은 행동을 취하는 것을 뜻합니다.

<br>
## 3.3 General Parameterized Policy

일반적인 정책에서 또한 npg는 가장 좋은 행동을 선택하는 쪽으로 학습합니다.

만약 $\widetilde\omega$가 approximation error을 최소화 하는 파라미터라고 가정하고 파라미터는 $\theta'=\theta+\alpha\widetilde\bigtriangledown\eta(\theta)$의 방식으로 업데이트 합니다. 일반적으로 업데이트를 $\theta'=\theta+\bigtriangleup\theta$로 표현하며 앞 문장의 방식으로 표현을 바꾼다면 $\bigtriangleup\theta = \alpha\widetilde\bigtriangledown\eta(\theta)$가 됩니다. 그리고 3.1절의 결과에 의해 $\bigtriangleup\theta=\alpha\widetilde\omega$로 표현할 수 있습니다. 이를 이용하여 Taylor expansion에 의해 다음과 같이 전개 될 수 있습니다.
$$
\pi(a;s,\theta')=\pi(a;s,\theta)+\dfrac{\partial\pi(a;s,\theta^T)}{\partial\theta}\bigtriangleup\theta+O(\bigtriangleup\theta^2)
$$

$$
= \pi(a;s,\theta)(1+\alpha\psi(s,a)^T\bigtriangleup\theta)+O(\bigtriangleup\theta^2)
$$

$$
= \pi(a;s,\theta)+(1+\alpha\psi(s,a)^T\widetilde\omega)+O(\alpha^2)
$$

$$
=\pi(a;s,\theta) + (1+\alpha f^\pi(s,a;\widetilde\omega))+O(\alpha^2)
$$

정책 자체가 function approximator의 크기대로 업데이트가 되기 때문에 지역적으로 가장 좋은 행동의 확률은 커지고 다른 확률은 작아질 것입니다. 하지만 탐욕적으로 향상을 하더라도 그게 성능자체를 향상시키지는 않을 수도 있습니다. 그렇기 때문에 line search기법과 함께 사용할 경우 성능 향상을 보장할 수 있습니다.

<br><br>

# 4. Metrics and Curvatures

2절에서 설명하고 있는 Positive-Definite Matrix인 FIM이외의 다른 Matrix도 사용할 수 있습니다. 본 논문에서는 다음과 같이 설명하고 있습니다. 다양한 파라미터 추정에서 FIM은 Hessian Matrix에 수렴하지 않을 수 있다고 합니다. 이 말은 2nd order 수렴이 보장되지 않는다는 말입니다.  [Mackay](http://www.inference.org.uk/mackay/ica.pdf) 논문에서 Hessian에서 data independant한 term을 metric으로 가져오는 방법을 제안했습니다. 그래서 performance를 2번 미분해보면 다음과 같습니다. 하지만 다음 식에서는 모든 항이 data dependent합니다(Q가 있으니까). 첫 번째 항이 그나마 FIM과의 관련성이 있을 수 있지만 Q 값이 curvature에 weight를 주는 방식 때문에 다르다고 할 수 있습니다.

$\nabla^2\eta(\theta)=\sum_{sa}\rho^{\pi}(s)(\nabla^2\pi(a;s)Q^{\pi}(s,a)+\nabla\pi(a;s)\nabla Q^{\pi}(s,a)^T+\nabla Q^{\pi}(s,a)\nabla\pi(a;s)^T)$

Hessian Matrix는 무조건 Positive-Definite Matrix인 것이 아니기 때문에, 따라서 국부적 최대점이 될 때 까지 Hessian Matrix를 사용하는 것은 좋은 방법이 아니라고 설명하고 있습니다. 이를 극복하기 위해서 conjugate methods가 효율적이라고 설명하고 있습니다. 

<br><br>

# 5. Experiment

본 논문에서는 LQR, simple MDP, 그리고 tetris MDP에 대해서 실험을 진행했습니다. FIM은 다음과 같은 식으로 업데이트합니다.

$f\leftarrow f+\nabla \log \pi(a_t; s_t, \theta)\nabla \log \pi(a_t; s_t, \theta)^T$

$T$길이의 경로에 대해서 $f/T$를 이용해 $F$의 기대값($E$)를 구합니다.

<br>
## 5.1 LQR(Linear Quadratic Regulator)

Agent를 실험할 환경은 다음과 같은 dynamics를 가지고 있습니다.

$x(t+1) = 0.7x(t)+u(t)+\epsilon(t)$

$u(t)$는 control 신호로서 에이전트의 행동입니다. $\epsilon$은 noise distribution으로 환경에 가해지는 노이즈입니다. 에이전트의 목표는 적절한 $u(t)$를 통해 $x(t)$를 0으로 유지하는 것입니다. $x(t)$를 0으로 유지하기 위해서 필요한 소모값(cost)는 $x(t)^2$로 정의하며 cost를 최소화하도록 학습합니다. 이 논문에서는 실험할 때 복잡성을 더 해주기 위해 noise distribution인 $\epsilon$을 dynamics에 추가하였습니다. 

이 실험에서 policy는 다음과 같이 설정하였습니다. 파라미터가 2개 밖에 없는 간단한 policy입니다.

$\pi(u;x,\theta)\propto \exp(\theta_1s_1x^2+\theta_2s_2x)$

이 policy를 간단히 그래프로 그려보면 다음과 같습니다. 가로축은 $x$, 세로축은 cost입니다. $\theta_1$과 $\theta_2$를 (0.5, 0.5), (1, 0), (0, 1)로 설정하고 $s_1$, $s_2$는, 1로 두었습니다. $x$는 -1~1사이의 범위로 그렸습니다. 

<center><img src='https://www.dropbox.com/s/v69qyrwn7zurk8c/Screenshot%202018-06-08%2014.57.07.png?dl=1' width='500px'></center>

아래의 그림은 LQR을 학습한 그래프입니다. cost가 $x^2$이기 때문에 cost가 0으로 갈수록 agent는 0에서 안정적으로 머무른다고 볼 수 있습니다. 6개의 선 중에서 오른쪽 세 개가 일반적인 gradient 방법을 사용해서 학습한 결과입니다. 그리고 왼쪽의 세 개의 선이 natural policy gradient를 통해 학습한 곡선입니다. 일반 gradient 방법보다 natural gradient가 훨씬 빠르게 학습합니다(time 축이 log scale인 것을 감안하면 차이가 많이 납니다.). 하지만 문제가 있습니다. NPG를 학습한 세 개의 곡선은 $\theta$를 rescale 한 것입니다. $\theta$앞에 곱해지는 숫자에 따라 학습의 과정이 다릅니다. 이 것은 coordinate에 따라 steepest gradient가 다르게 측정된다는 것입니다. 즉, covariant gradient가 아니라는 뜻입니다. 이 논문에서는 natural gradient를 통해 gradient가 covariant하도록 만들고 싶었는데 실패한 것입니다. 

<center><img src="https://www.dropbox.com/s/fhn8cgje0rdws0i/Screenshot%202018-06-08%2023.13.37.png?dl=1" width="300px"></center>

natural gradient가 covariant하지 않은 이유는 Fisher Information Matrix가 예상했던 바와는 달리 invariant metric이 아니기 때문입니다. 또한 FIM이 invariant metric이 아닌 이유는 FIM을 계산할 때 $\rho_s$가 곱해지기 때문입니다. 하지만 여전히 의의가 있는 것은 기존 gradient 방법들보다 훨씬 빠르게 학습한다는 것입니다!

<br>
## 5.2 Simple 2-state MDP

이제 다른 예제에서 NPG를 테스트한다. 2개의 state만 가지는 MDP를 고려해봅시다. [그림출처](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1080&context=robotics). 그림으로보면 다음과 같습니다. $x=0$ 상태와 $x=1$ 상태 두 개가 존재합니다. 에이전트는 각 상태에서 다시 자신의 상태로 되돌아오는 행동을 하거나 다른 상태로 가는 행동을 할 수 있습니다. 상태 $x=0$에서 다시 자기 자신으로 돌아오면 1의 보상을 받고 상태 $x=1$에서 자기 자신으로 돌아오면 2의 보상을 받습니다. 결국 optimal policy는 상태 $x=1$에서 계속 자기 자신으로 돌아오는 행동을 취하는 것입니다. 

<img src="https://www.dropbox.com/s/g1x9yknzsrip59i/Screenshot%202018-06-08%2023.06.50.png?dl=1">

문제를 좀 어렵게 만들기 위해 state distribution을 다음과 같이 설정합니다. 즉, 대부분의 경우에 상태 x=0에서 에이전트가 시작하는 것입니다. 
$$
\rho(x=0)=0.8,  \rho(x=1)=0.2
$$
일반적인 gradient 방법을 사용하면 다음과 같은 policy gradient 식에 따라서 업데이트를 하게 됩니다. 이 때, $\rho(s)$가 gradient에 곱해지므로 상대적으로 상태 0에서의 gradient 값이 커지게 됩니다. 따라서 에이전트는 상태 0에서의 gradient(상태 0에서 스스로에게 돌아오는 행동을 취하도록 정책을 업데이트하는 gradient)를 따라 parameterized policy를 update합니다. 따라서 아래 그림의 첫번째 그림에서처럼 reward가 1에서 오랜 시간동안 머무릅니다. 즉, 에이전트가 상태 0에서 self-loop를 계속 돌고 있다는 것입니다.

$$\nabla\eta(\theta)=\sum_{s,a}\rho^{\pi}(s)\nabla\pi(a;s,\theta)Q^{\pi}(s,a)$$

<center><img src="https://www.dropbox.com/s/xtb77mfazbppnss/Screenshot%202018-06-08%2023.14.24.png?dl=1" width="300px"></center>

하지만 NPG를 사용할 경우에는 훨씬 빠르게 average reward가 2에 도달합니다. gradient 방법이 $1.7\times 10^7$정도의 시간만에 2에 도달한 반면 NPG는 2의 시간만에 도달합니다. 또한 $\rho(x=0)$가 $10^{-5}$이하로 떨어지지 않습니다.

한 가지 그래프를 더 살펴봅시다. 다음 그래프는 parameter $\theta$가 업데이트 되는 과정을 보여줍니다. 이 그래프에서는 parameter가 2개 있습니다. 일반적인 gradient가 아래 그래프에서 실선에 해당합니다. 이 실선의 그래프는 보면 처음부터 중반까지 $\theta_i$만 거의 업데이트하는 것을 볼 수 있습니다. 그에 비해 NPG는 두 개의 parameter를 균등하게 업데이트하는 것을 볼 수 있습니다. 

<center><img src="https://www.dropbox.com/s/g7pazozw2k6rd7x/Screenshot%202018-06-08%2023.23.25.png?dl=1" width="300px"></center>

policy가 $\pi(a;s,\theta)\propto \exp(\theta_{sa})$일 때, 다음과 같이 $F_{-1}$이 gradient 앞에 weight로 곱해지는데 이게 $\rho$와는 달리 각 parameter에 대해 균등합니다. 따라서 위 그래프에서와 같이 각 parameter는 비슷한 비율로 업데이트가 되는 것입니다.

$$\bar{\nabla}\eta(\theta) = F^{-1}\nabla\eta(\theta)$$

<br>
## 5.3 Tetris

NPG를 테스트할 tetris 예제는 Neuro Dynamic Programming 책에 소개되어 있습니다. 다음 그림은 tetris 예제를 보여줍니다. 보통 그림에서와 같이 state의 feature를 정해줍니다. [그림 출처](http://slideplayer.com/slide/5215520/)

<img src="https://www.dropbox.com/s/y1halso9yermy8s/Screenshot%202018-06-08%2023.44.34.png?dl=1">

이 예제에서도 exponential family로 policy를 표현합니다. $\pi(a;s,\theta) \propto \exp(\theta^T\phi_{sa})$ 로 표현합니다.

tetris는 linear function approximator와 greedy policy iteration을 사용할 경우 performance가 갑자기 떨어지는 현상이 있습니다. 밑의 그림에서 A의 spike가 있는 그래프가 이 경우입니다. 그 밑에 낮게 누워있는 그래프는 일반적인 policy gradient 방법입니다. 하지만 Natural policy gradient를 사용할 경우 B 그림에서 오른쪽 그래프와 같이 성능개선이 뚜렷합니다. Policy Iteration 처럼 성능이 뚝 떨어지지 않고 안정적으로 유지합니다. 또한 그림 C에서 보는 것처럼 오른쪽 그래프인 일반적인 gradient 방법보다 훨씬 빠르게 학습하는 것을 볼 수 있습니다.

<img src="https://www.dropbox.com/s/pr6s2qrqaic0wyj/Screenshot%202018-06-08%2023.40.16.png?dl=1">

<br><br>

# 6. Discussion

Natural Gradient Method는  Policy Iteration에서와 같이 Greedy Action을 선택하도록 학습합니다. Line search 기법과 함께 사용하면 더 Policy Iteration과 비슷해집니다. Greedy Policy Iteration에서와 비교하면 Performance Improvement가 보장됩니다. 하지만 FIM이 asymtotically Hessian으로 수렴하지 않습니다. 그렇기 때문에 Conjugate Gradient Method로 구하는 방법이 더 좋을수 있습니다.

살펴봤듯이 본 논문의 NPG는 완벽하지 않습니다. 위의 몇가지 문제점을 극복하기 위한 추가적인 연구가 필요하다고 할 수 있습니다.

