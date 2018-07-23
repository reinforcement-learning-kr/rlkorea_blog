---
title: Deterministic Policy Gradient Algorithms
date: 2018-06-27 17:21:48
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 장수영, 공민서
subtitle: 피지여행 2번째 논문
---

<center> <img src="https://www.dropbox.com/s/etwa3hn5c0pegdv/Screen%20Shot%202018-07-18%20at%2012.51.35%20AM.png?dl=1" width="800"> </center>

논문 저자 : David Silver, Guy Lever, Nicloas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller
논문 링크 : http://proceedings.mlr.press/v32/silver14.pdf
Proceeding : International Conference on Machine Learning (ICML) 2014
정리 : 장수영, 공민서

---

## 1. Summary
- Deterministic Policy Gradient (DPG) Theorem 제안합니다 [[Theorem 1](#deterministic-policy-gradient-theorem)]
    1) DPG는 존재하며,
    2) DPG는 Expected gradient of the action-value function의 형태를 띕니다.
- Policy variance가 0에 수렴할 경우, DPG는 Stochastic Policy Gradient(SPG)와 동일해짐을 보임 [[Theorem 2](#dpg-는-spg-의-limiting-case-임)]
    - Theorem 2 로 인해 기존 Policy Gradient (PG) 와 관련된 기법들을 DPG 에 적용할 수 있게 됨
        - 예. Sutton PG, natural gradients, actor-critic, episodic/batch methods
- 적절한 exploration 을 위해 model-free, off-policy Actor-Critic algorithm 을 제안함
    - Action-value function approximator 사용으로 인해 policy gradient가 bias되는 것을 방지하기 위해 compatibility conditions을 제공 [[Theorem 3](##deterministic-actor-critic-algorithms)]
- DPG 는 SPG 보다 성능이 좋음
    - 특히 high dimensional action spaces 을 가지는 tasks에서의 성능 향상이 큼
        - SPG의 policy gradient는 state와 action spaces 모두에 대해서, DPG의 policy gradient는  state spaces에 대해서만 평균을 취함
        - 결과적으로, action spaces의 dimension이 커질수록 data efficiency가 높은 DPG의 학습이 더 잘 이뤄지게 됨
        - 무한정 학습을 시키게 되면, SPG도 최적으로 수렴할 것으로 예상되기에 위 성능 비교는 일정 iteration 내로 한정함
    - 기존 기법들에 비해 computation 양이 많지 않음
        - Computation 은 action dimensionality 와 policy parameters 수에 비례함
<br>

---
## 2. Background
### 2.1 Performance objective function

$$
\begin{align}
J(\pi_{\theta}) &= \int_{S}\rho^{\pi}(s)\int_{A}\pi_{\theta}(s,a)r(s,a)dads = E_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}[r(s,a)]
\end{align}
$$

### 2.2 SPG Theorem
- State distribution $ \rho^{\pi}(s) $ 은 policy parameters에 영향을 받지만, policy gradient 를 계산할 때는 state distribution 의 gradient 를 고려할 필요가 없다.
- $$\begin{eqnarray}\nabla_{\theta}J(\pi_{\theta}) &=& \int_{S}\rho^{\pi}(s)\int_{A}\nabla_{\theta}\pi_{\theta}(a|s)Q^{\pi}(s,a)dads \nonumber \\ &=& E_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a)]
\end{eqnarray}$$

### 2.3 Stochastic Actor-Critic Algorithms
- Actor 와 Critic 이 번갈아가면서 동작하며 stochastic policy 를 최적화하는 기법
- Actor: $ Q^{\pi}(s,a) $ 를 근사한 $ Q^w(s,a) $ 를 이용해 stochastic policy gradient 를 ascent 하는 방향으로 policy parameter $ \theta $ 를 업데이트함으로써 stochastic policy 를 발전시킴
    - $ \nabla_{\theta}J(\pi_{\theta}) = E_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}[\nabla_{\theta}log\pi_{\theta}(a|s)Q^{w}(s,a)] $
- Critic: SARSA 나 Q-learning 같은 Temporal-difference (TD) learning 을 이용해 action-value function의 parameter, $ w $ 를 업데이트함으로써 $ Q^w(s,a) $ 가 $ Q^{\pi}(s,a) $ 과 유사해지도록 함
- 실제 값인 $ Q^{\pi}(s,a) $ 대신 이를 근사한 $ Q^w(s,a) $ 를 사용하게 되면, 일반적으로 bias 가 발생하게 된다. 하지만, compatible condition 에 부합하는 $ Q^w(s,a) $ 를 사용하게 되면, bias 가 발생하지 않는다.

### 2.4 Off-policy Actor-Critic
- Distinct behavior policy $ \beta(a|s) ( \neq \pi_{\theta}(a|s) ) $ 로부터 샘플링된 trajectories 를 이용한 Actor-Critic
- Performance objective function
    - $\begin{eqnarray}
        J_{\beta}(\pi_{\theta}) 
        &=& \int_{S}\rho^{\beta}(s)V^{\pi}(s)ds \nonumber \\\\
        &=& \int_{S}\int_{A}\rho^{\beta}(s)\pi_{\theta}(a|s)Q^{\pi}(s,a)dads
        \end{eqnarray} $
- off-policy policy gradient
    - $ \begin{eqnarray}
        \nabla_{\theta}J_{\beta}(\pi_{\theta}) &\approx& \int_{S}\int_{A}\rho^{\beta}(s)\nabla_{\theta}\pi_{\theta}(a|s)Q^{\pi}(s,a)dads \nonumber \end{eqnarray} $
        $=E_{s \sim \rho^{\beta}, a \sim \beta}[\frac{\pi_{\theta}(a|s)}{\beta_{\theta}(a|s)}\nabla_{\theta}log\pi_{\theta}(a|s)Q^{\pi}(s,a)]$
    - off-policy policy gradient 식에서의 물결 표시는 [Degris, 2012b] 논문에 근거함
        - [Degris, 2012b] "Linear off-policy actor-critic," ICML 2012
        - Exact off-policy policy gradient 와 이를 approximate 한 policy gradient 는 아래와 같음. (빨간색 상자에 있는 항목을 삭제함으로써 근사함)
            - <img src="https://www.dropbox.com/s/xzpv3okc139c1fs/Screenshot%202018-06-16%2017.48.51.png?dl=1" width=500px>
        - [Degris, 2012b] Theorem 1 에 의해 policy parameter 가 approximated policy gradient ( $\nabla_{u}𝑄^{\pi,\gamma}(𝑠,𝑎)$ term 제거)에 따라 업데이트되어도 policy 는 improve 가 됨이 보장되기에 exact off-policy policy gradient 대신 approximated off-policy policy gradient 를 사용해도 괜찮음.
            - <img src="https://www.dropbox.com/s/mk13931r4scjngo/Screenshot%202018-06-16%2017.49.24.png?dl=1" width=500px>
    - off-policy policy gradient 식에서 $ \frac{\pi_{\theta}(a|s)}{\beta_{\theta}(a|s)} $ 는 importance sampling ratio 임
        - off-policy actor-critic 에서는 $ \beta $ 에 의해 샘플링된 trajectory 를 이용해서 stochastic policy $ \pi $ 를 예측하는 것이기 때문에 imnportance sampling 이 필요함
<br>

---
## 3. Gradient of Deterministic Policies
### 3.1 Regulariy Conditions
- 어떠한 이론이 성립하기 위한 전제 조건
- Regularity conditions A.1
    - $ p(s'|s,a), \nabla_{a}p(s'|s,a), \mu_{\theta}(s), \nabla_{\theta}\mu_{\theta}(s), r(s,a), \nabla_{a}r(s,a), p_{1}(s) $ are continuous in all parameters and variables $ s, a, s' $ and $ x $.
- regularity conditions A.2
    - There exists a $ b $ and $ L $ such that $ \sup_{s}p_{1}(s) < b $, $ \sup_{a,s,s'}p(s'|s,a) < b $, $ \sup_{a,s}r(s,a) < b $, $ \sup_{a,s,s'}\|\nabla_{a}p(s'|s,a)\| < L $, and $ \sup_{a,s}\|\nabla_{a}r(s,a)\| < L $.

### 3.2 Deterministic Policy Gradient Theorem
- Deterministic policy
    - $ \mu_{\theta} : S \to A $ with parameter vector $ \theta \in \rm I \! R^n $
- Probability distribution
    - $ p(s \to s', t, \mu) $
- Discounted state distribution
    - $ \rho^{\mu}(s) $
- Performance objective

$$
J(\mu_{\theta}) = E[r^{\gamma}_{1} | \mu] 
$$

$$
= \int_{S}\rho^{\mu}(s)r(s,\mu_{\theta}(s))ds 
= E_{s \sim \rho^{\mu}}[r(s,\mu_{\theta}(s))]
$$

- DPG Theorem
    - MDP 가 A.1 만족한다면, 아래 식(9)이 성립함
    $\nabla_{\theta}J(\mu_{\theta}) = \int_{S}\rho^{\mu}(s)\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)\vert_{a=\mu_{\theta}(s)}ds \nonumber$
    $= E_{s \sim \rho^{\mu}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)\vert_{a=\mu_{\theta}(s)}]   \nonumber $ (9)
    
	- DPG는 State space 에 대해서만 평균을 취하면 되기에, State와 Action space 모두에 대해 평균을 취해야 하는 [SPG](#spg-theorem)에 비해 data efficiency가 좋다. 즉, 더 적은 양의 데이터로도 학습이 잘 이뤄지게 된다.

    
### 3.3 DPG 형태에 대한 informal intuition
- Generalized policy iteration
    - 정책 평가와 정책 발전을 한 번 씩 번갈아 가면서 실행하는 정책 iteration
        - 위와 같이 해도 정책 평가에서 예측한 가치함수가 최적 가치함수에 수렴함
- 정책 평가
    - action-value function $ Q^{\pi}(s,a) $ or $ Q^{\mu}(s,a) $ 을 estimate 하는 것
- 정책 발전
    - 위 estimated action-value function 에 따라 정책을 update 하는 것
    - 주로 action-value function 에 대한 greedy maximisation 사용함
        - $ \mu^{k+1}(s) = \arg\max\limits_{a}Q^{\mu^{k}}(s,a) $
        - greedy 정책 발전은 매 단계마다 global maximization을 해야하는데, 이는 continuous action spaces 에서 계산량이 급격히 늘어나게 됨.
    - 그렇기에 policy gradient 방법이 나옴
        - policy 를 $ \theta $ 에 대해서 parameterize 함
        - 매 단계마다 global maximisation 수행하는 대신, 방문하는 state $ s $ 마다 policy parameter 를 action-value function $ Q $ 의 $ \theta $ 에 대한 gradient $ \nabla_{\theta}Q^{\mu^{k}}(s,\mu_{\theta}(s)) $ 방향으로 proportional 하게 update 함
        - 하지만 각 state 는 다른 방향을 제시할 수 있기에, state distribution $ \rho^{\mu}(s) $ 에 대한 기대값을 취해 policy parameter 를 update 할 수도 있음
            - $ \theta^{k+1} = \theta^{k} + \alpha \rm I\!E_{s \sim \rho^{\mu^{k}}} [\nabla_{\theta}Q^{\mu^{k}}(s,\mu_{\theta}(s))] $
        - 이는 chain-rule 에 따라 아래와 같이 분리될 수 있음
            - $ \theta^{k+1} = \theta^{k} + \alpha \rm I\!E_{s \sim \rho^{\mu^{k}}} [\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu^{k}}(s,a)\vert_{a=\mu_{\theta}(s)}] $ (7)
            - chain rule: $ \frac{\partial Q}{\partial \theta} = \frac{\partial a}{\partial \theta} \frac{\partial Q}{\partial a} $
        - 하지만 state distribution $ \rho^{\mu} $ 은 정책에 dependent 함
            - 정책이 바꾸게 되면, 바뀐 정책에 따라 방문하게 되는 state 가 변하게 되기 때문에 state distribution이 변하게 됨
        - 그렇기에 정책 update 시 state distribution에 대한 gradient 를 고려하지 않는데 정책 발전이 이뤄진다는 것은 직관적으로 와닿지 않을 수 있음
        - deterministic policy gradient theorem 은 state distribution에 대한 gradient 계산없이 위 식(7) 대로만 update 해도 performance objective 의 gradient 를 정확하게 따름을 의미한다.


### 3.4 DPG 는 SPG 의 limiting case 임
- stochastic policy parameterization
    - $ \pi_{\mu_{\theta},\sigma} $ by a deterministic policy $ \mu_{\theta} : S \to A $ and a variance parameter $ \sigma $
    - $ \sigma = 0 $ 이면, $ \pi_{\mu_{\theta},\sigma} \equiv \mu_{\theta} $
- Theorem 2. Policy 의 variance 가 0 에 수렴하면, 즉, $ \sigma \to 0 $, stochastic policy gradient 와 deterministic policy gradient 는 동일해짐
    - 조건 : stochastic policy $ \pi_{\mu_{\theta},\sigma} = \nu_{\sigma}(\mu_{\theta}(s),a) $
        - $ \sigma $ 는 variance
        - $ \nu_{\sigma}(\mu_{\theta}(s),a) $ 는 conditions B.1 만족
        - MDP 는 conditions A.1 및 A.2 만족
    - 결과 :
        - $ \lim\limits_{\sigma\downarrow0}\nabla_{\theta}J(\pi_{\mu_{\theta},\sigma}) = \nabla_{\theta}J(\mu_{\theta})  $
            - 좌변은 standard stochastic gradient 이며, 우변은 deterministic gradient.
    - 의미 :
        - deterministic policy gradient 는 stochastic policy gradient 의 특수 case 임
        - 기존 유명한 policy gradients 기법들에 deterministic policy gradients 를 적용할 수 있음
            - 기존 기법들 예: compatible function approximation (Sutton, 1999), natural gradients (Kakade, 2001), actor-critic (Bhatnagar, 2007) or episodic/batch methods (Peters, 2005)
<br>

---
## 4. Deterministic Actor-Critic Algorithms
1. 살사 critic 을 이용한 on-policy actor-critic
    - 단점
        - deterministic policy 에 의해 행동하면 exploration 이 잘 되지 않기에, sub-optimal 에 빠지기 쉬움
    - 목적
        - 교훈/정보제공
        - 환경에서 충분한 noise 를 제공하여 exploration을 시킬 수 있다면, deterministic policy 를 사용한다고 하여도 좋은 학습 결과를 얻을 수도 있음$
            - 예. 바람이 agent의 행동에 영향(noise)을 줌
    - Remind: 살사(SARSA) update rule
        - $ Q(s_{t},a_{t}) \leftarrow Q(s_{t},a_{t}) + \alpha(r_{t} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_{t},a_{t})) $
    - Algorithm
        - Critic 은 MSE 를 $ \bf minimize $ 하는 방향, 즉, action-value function 을 stochastic gradient $ \bf descent $ 방법으로 update 함
            - $ MSE = [Q^{\mu}(s,a) - Q^{w}(s,a)]^2 $
                - critic 은 실제 $ Q^{\mu}(s,a) $ 대신 미분 가능한 $ Q^{w}(s,a) $ 로 대체하여 action-value function 을 estimate 하며, 이 둘 간 Mean Square Error 를 최소화하는 것이 목표
            - $ \nabla_{w}MSE \approx -2 * [r + \gamma Q^{w}(s',a') - Q^{w}(s,a)]\nabla_{w}Q^{w}(s,a)  $
                - $ \nabla_{w}MSE = -2 * [Q^{\mu}(s,a) - Q^{w}(s,a)]\nabla_{w}Q^{w}(s,a)  $
                - $ Q^{\mu}(s,a) $ 를 $ r + \gamma Q^{w}(s',a') $ 로 대체
                    - $ Q^{\mu}(s,a) = r + \gamma Q^{\mu}(s',a') $
            - $ w_{t+1} = w_{t} + \alpha_{w}\delta_{t}\nabla_{w}Q^{w}(s_{t},a_{t}) $
                - $w_{t+1} = w_{t} - \alpha_{w}\nabla_{w}MSE  \nonumber$
                $ \approx w_{t} - \alpha_{w} * (-2 * [r + \gamma Q^{w}(s',a') - Q^{w}(s,a)] \nabla_{w}Q^{w}(s,a)$
                - $ \delta_{t} = r_{t} + \gamma Q^{w}(s_{t+1},a_{t+1}) - Q^{w}(s_{t},a_{t}) $
        - Actor 는 식(9)에 따라 보상이 $ \bf maximize $ 되는 방향, 즉, deterministic policy 를 stochastic gradient $ \bf ascent $ 방법으로 update함
            - $ \theta_{t+1} = \theta_{t} + \alpha_{\theta} \nabla_{\theta}\mu_{\theta}(s_{t})\nabla_{a}Q^{w}(s_{t},a_{t})\vert_{a=\mu_{\theta}(s)} $
2. Q-learning 을 이용한 off-policy actor-critic
    - stochastic behavior policy $ \beta(a|s) $ 에 의해 생성된 trajectories 로부터 deterministic target policy $ \mu_{\theta}(s) $ 를 학습하는 off-policy actor-critic
    - performance objective
        - $ J_{\beta}(\mu_{\theta}) = \int_{S}\rho^{\beta}(s)V^{\mu}(s)ds \nonumber \\\\$
          $= \int_{S}\rho^{\beta}(s)Q^{\mu}(s,\mu_{\theta}(s))ds \nonumber \\\\$
          $= E_{s \sim \rho^{\beta}}[Q^{\mu}(s,\mu_{\theta}(s))]$
    - off-policy deterministic policy gradient
        - $ \nabla_{\theta}J_{\beta}(\mu_{\theta}) = E_{s \sim \rho^{\beta}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)\vert_{a=\mu_{\theta}(s)}] $
            - 논문에는 아래와 같이 나와있는데, 물결 표시 부분은 오류로 판단됨.
            - $ \begin{eqnarray}
                \nabla_{\theta}J_{\beta}(\mu_{\theta}) &\approx& \int_{S}\rho^{\beta}(s)\nabla_{\theta}\mu_{\theta}(a|s)Q^{\mu}(s,a)ds \nonumber \\
                &=& E_{s \sim \rho^{\beta}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)\vert_{a=\mu_{\theta}(s)}]
                \end{eqnarray} $
            - 근거: Action이 deterministic 하기에 stochastic 경우와는 다르게 performance objective 에서 action 에 대해 평균을 구할 필요가 없음. 그렇기에, 곱의 미분이 있을 필요가 없고, [Degris, 2012b] 에서처럼 곱의 미분을 통해 생기는 action-value function 에 대한 gradient term 를 생략할 필요가 사라짐. [참고](#off-policy-actor-critic)
    - Remind: 큐러닝(Q-learning) update rule
        - $ Q(s_{t},a_{t}) \leftarrow Q(s_{t},a_{t}) + \alpha(r_{t} + \gamma \max\limits_{a}Q(s_{t+1},a) - Q(s_{t},a_{t})) $
    - algorithm: OPDAC (Off-Policy Deterministic Actor-Critic)
        - 살사를 이용한 on-policy deterministic actor-critic 과 아래 부분을 제외하고는 같음
            - target policy 는 $ \beta(a|s) $ 에 의해 생성된 trajectories 를 통해 학습함
            - 업데이트 목표 부분에 실제 행동 값 $ a_{t+1} $ 이 아니라 정책으로부터 나온 행동 값 $ \mu_{\theta}(s_{t+1}) $ 사용함
                - $ \mu_{\theta}(s_{t+1}) $ 는 가장 높은 Q 값을 가지는 행동. 즉, Q-learning.
        - $ \delta_{t} = r_{t} + \gamma Q^{w}(s_{t+1},\mu_{\theta}(s_{t+1})) - Q^{w}(s_{t},a_{t}) $
        - $ w_{t+1} = w_{t} + \alpha_{w}\delta_{t}\nabla_{w}Q^{w}(s_{t},a_{t}) $
        - $ \theta_{t+1} = \theta_{t} + \alpha_{\theta} \nabla_{\theta}\mu_{\theta}(s_{t})\nabla_{a}Q^{w}(s_{t},a_{t})\vert_{a=\mu_{\theta}(s)} $
    - Stochastic off-policy actor-critic 은 대개 actor 와 critic 모두 importance sampling을 필요로 하지만, deterministic policy gradient 에선 importance sampling이 필요없음
        - Actor 는 deterministic 이기에 sampling 자체가 필요없음
            - Stochastic policy 인 경우, Actor 에서 importance sampling이 필요한 이유는 상태 $ s $ 에서의 가치 함수 값 $ V^{\pi}(s) $ 을 estimate 하기 위해 $ \pi $ 가 아니라 $ \beta $ 에 따라 sampling을 한 후, 평균을 내기 때문임.
            - 하지만 Deterministic policy 인 경우, 상태 $ s $ 에서의 가치 함수 값 $ V^{\pi}(s) = Q^{\pi}(s,\mu_{\theta}) $ 즉, action 이 상태 s 에 대해 deterministic 이기에 sampling 을 통해 estimate 할 필요가 없고, 따라서 importance sampling도 필요없어짐.
            - stochastic vs. deterministic performance objective
                - stochastic : $ J_{\beta}(\mu_{\theta}) = \int_{S}\int_{A}\rho^{\beta}(s)\pi_{\theta}(a|s)Q^{\pi}(s,a)dads $
                - deterministic : $ J_{\beta}(\mu_{\theta}) = \int_{S}\rho^{\beta}(s)Q^{\mu}(s,\mu_{\theta}(s))ds $
        - Critic 이 사용하는 Q-learning 은 importance sampling이 필요없는 off policy 알고리즘임.
            - Q-learning 도 업데이트 목표를 특정 분포에서 샘플링을 통해 estimate 하는 것이 아니라 Q 함수를 최대화하는 action을 선택하는 것이기에 위 actor 에서의 deterministic 경우와 비슷하게 볼 수 있음
3. compatible function approximation 및 gradient temporal-difference learning 을 이용한 actor-critic
    - 위 살사/Q-learning 기반 on/off-policy 는 아래와 같은 문제가 존재
        - function approximator 에 의한 bias
            - 일반적으로 $ Q^{\mu}(s,a) $ 를 $ Q^{w}(s,a) $ 로 대체하여 deterministic policy gradient 를 구하면, 그 gradient 는 ascent 하는 방향이 아닐 수도 있음
        - off-policy learning 에 의한 instabilities
    - 그래서 stochastic 처럼 $ \nabla_{a}Q^{\mu}(s,a) $ 를 $ \nabla_{a}Q^{w}(s,a) $ 로 대체해도 deterministic policy gradient 에 영향을 미치지 않을 compatible function approximator $ Q^{w}(s,a) $ 를 찾아야 함
    - Theorem 3. 아래 두 조건을 만족하면, $ Q^{w}(s,a) $ 는 deterministic policy $ \mu_{\theta}(s) $ 와 compatible 함. 즉, $ \nabla_{\theta}J_{\beta}(\mu_{\theta}) = E_{s \sim \rho^{\beta}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{w}(s,a)\vert_{a=\mu_{\theta}(s)}] $
        - $ \nabla_{a}Q^{w}(s,a)\vert_{a=\mu_{\theta}(s)} = \nabla_{\theta}\mu_{\theta}(s)^{\top}w $
        - $ w $ 는 $ MSE(\theta, w) = E[\epsilon(s;\theta,w)^{\top}\epsilon(s;\theta,w)] $ 를 최소화함
            - $ \epsilon(s;\theta,w) = \nabla_{a}Q^{w}(s,a)\vert_{a=\mu_{\theta}(s)} - \nabla_{a}Q^{\mu}(s,a)\vert_{a=\mu_{\theta}(s)}  $
    - Theorem 3 은 on-policy 뿐만 아니라 off-policy 에도 적용 가능함
    - $ Q^{w}(s,a) = (a-\mu_{\theta}(s))^{\top}\nabla_{\theta}\mu_{\theta}(s)^{\top} w + V^{v}(s) $
        - 어떠한 deterministic policy 에 대해서도 위 형태와 같은 compatible function approximator 가 존재함.
        - 앞의 term 은 advantage 를, 뒤의 term 은 value 로 볼 수 있음
    - $ Q^{w}(s,a) = \phi(s,a)^{\top} w + v^{\top}\phi(s) $
        - 정의 : $ \phi(s,a) \overset{\underset{\mathrm{def}}{}}{=} \nabla_{\theta}\mu_{\theta}(s)(a-\mu_{\theta}(s)) $
        - 일례 : $ V^{v}(s) = v^{\top}\phi(s) $
        - Theorem 3 만족 여부
            - 첫 번째 조건 만족.
            - 두 번째 조건은 대강 만족.
                - $ \nabla_{a}Q^{\mu}(s,a) $ 에 대한 unbiased sample 을 획득하기는 매우 어렵기에, 일반적인 정책 평가 방법들로 $ w $ 학습.
                - 이 정책 평가 방법들을 이용하면 $ Q^{w}(s,a) \approx Q^{\mu}(s,a) $ 인 reasonable solution을 찾을 수 있기에 대강 $ \nabla_{a}Q^{w}(s,a) \approx \nabla_{a}Q^{\mu}(s,a) $ 이 될 것.
        - action-value function 에 대한 linear function approximator 는 큰 값을 가지는 actions 에 대해선 diverge할 수 있어 global 하게 action-values 예측하기에는 좋지 않지만, local critic 에 사용할 때는 매우 유용하다.
            - 즉, 절대값이 아니라 작은 변화량을 다루는 gradient method 경우엔 $ A^{w}(s,\mu_{\theta}(s)+\delta) = \delta^{\top}\nabla_{\theta}\mu_{\theta}(s)^{\top}w $ 로, diverge 하지 않고, 값을 얻을 수 있음
    - COPDAC-Q algorithm (Compatible Off-Policy Deterministic Actor-Critic Q-learning critic)
        - Critic: 실제 action-value function 에 대한 linear function approximator 인 $ Q^{w}(s,a) = \phi(s,a)^{\top}w $ 를 estimate
            - $ \phi(s,a) = a^{\top}\nabla_{\theta}\mu_{\theta} $
            - Behavior policy $ \beta(a|s) $ 로부터 얻은 samples를 이용하여 Q-learning 이나 gradient Q-learning 과 같은 off-policy algorithm 으로 학습 가능함
        - Actor: estimated action-value function 에 대한 gradient 를 $ \nabla_{\theta}\mu_{\theta}(s)^{\top}w $ 로 치환 후, 정책을 업데이트함
        - $ \delta_{t} = r_{t} + \gamma Q^{w}(s_{t+1},\mu_{\theta}(s_{t+1})) - Q^{w}(s_{t},a_{t}) $
        - $ w_{t+1} = w_{t} + \alpha_{w}\delta_{t}\phi(s_{t},a_{t}) $
        - $ \theta_{t+1} = \theta_{t} + \alpha_{\theta} \nabla_{\theta}\mu_{\theta}(s_{t})(\nabla_{\theta}\mu_{\theta}(s_{t})^{\top} w_{t}) $
    - off-policy Q-learning은 linear function approximation을 이용하면 diverge 할 수도 있음
        - $ \mu_{\theta}(s_{t+1}) $ 이 diverge 할 수도 있기 때문으로 판단됨.
        - 그렇기에 simple Q-learning 대신 다른 기법이 필요함.
    - 그렇기에 critic 에 gradient Q-learning 사용한 COPDAC-GQ (Gradient Q-learning critic) algorithm 제안
        - gradient temporal-difference learning 에 기반한 기법들은 true gradient descent algorithm 이며, converge가 보장됨. (Sutton, 2009)
            - 기본 아이디어는 stochastic gradient descent 로 Mean-squared projected Bellman error (MSPBE) 를 최소화하는 것
            - critic 이 actor 보다 빠른 time-scale 로 update 되도록 step size 들을 잘 조절하면, critic 은 MSPBE 를 최소화하는 parameters 로 converge 하게 됨
            - critic 에 gradient temporal-difference learning 의 일종인 gradient Q-learning 사용한 논문 (Maei, 2010)
    - COPDAC-GQ algorithm
        - $ \delta_{t} = r_{t} + \gamma Q^{w}(s_{t+1},\mu_{\theta}(s_{t+1})) - Q^{w}(s_{t},a_{t}) $
        - $ \theta_{t+1} = \theta_{t} + \alpha_{\theta} \nabla_{\theta}\mu_{\theta}(s_{t})(\nabla_{\theta}\mu_{\theta}(s_{t})^{\top} w_{t}) $
        - $ w_{t+1} = w_{t} + \alpha_{w}\delta_{t}\phi(s_{t},a_{t}) - \alpha_{w}\gamma\phi(s_{t+1}, \mu_{\theta}(s_{t+1}))(\phi(s_{t},a_{t})^{\top} u_{t}) $
        - $ v_{t+1} = v_{t} + \alpha_{v}\delta_{t}\phi(s_{t}) - \alpha_{v}\gamma\phi(s_{t+1})(\phi(s_{t},a_{t})^{\top} u_{t}) $
        - $ u_{t+1} = u_{t} + \alpha_{u}(\delta_{t}-\phi(s_{t}, a_{t})^{\top} u_{t})\phi(s_{t}, a_{t}) $
    - stochastic actor-critic 과 같이 매 time-step 마다 update 시 필요한 계산의 복잡도는 $ O(mn) $
        - m 은 action dimensions, n 은 number of policy parameters
    - Natural policy gradient 를 이용해 deterministic policies 를 찾을 수 있음
        - $ M(\theta)^{-1}\nabla_{\theta}J(\mu_{\theta}) $ 는 any metric $ M(\theta) $ 에 대한 our performance objective (식(14)) 의 steepest ascent direction 임 (Toussaint, 2012)
        - Natural gradient 는 Fisher information metric $ M_{\pi}(\theta) $ 에 대한 steepest ascent direction 임
            -  $ M_{\pi}(\theta) = E_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}[\nabla_{\theta}\log\pi_{\theta}(a|s)\nabla_{\theta}\log\pi_{\theta}(a|s)^{\top}] $
            - Fisher information metric 은 policy reparameterization 에 대해 불변임 (Bagnell, 2003)
        - deterministic policies 에 대해 metric 으로 $ M_{\mu}(\theta) = E_{s \sim \rho^{\mu}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{\theta}\mu_{\theta}(s)^{\top}] $ 을 사용.
        	- 이는 variance 가 0 인 policy 에 대한 Fisher information metric 으로 볼 수 있음
        	- $ \frac{\nabla_{\theta}\pi_{\theta}(a\vert s)}{\pi_{\theta}(a\vert s)}$ 에서 policy variance 가 0 이면, 특정 s 에 대한 $ \pi_{\theta}(a|s)$ 만 1 이 되고, 나머지는 0 임
        - deterministic policy gradient theorem 과 compatible function approximation 을 결합하면 $ \nabla_{\theta}J(\mu_{\theta}) = E_{s \sim \rho^{\mu}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{\theta}\mu_{\theta}(s)^{\top}w] $ 이 됨
            - $ \nabla_{\theta}J(\mu_{\theta}) = E_{s \sim \rho^{\mu}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q^{\mu}(s,a)\vert_{a=\mu_{\theta}(s)}] $
            - $ \nabla_{a}Q^{\mu}(s,a)\vert_{a=\mu_{\theta}(s)} \approx \nabla_{a}Q^{w}(s,a)\vert_{a=\mu_{\theta}(s)} = \nabla_{\theta}\mu_{\theta}(s)^{\top}w $
        - 그렇기에 steepest ascent direction 은 $ M_{\mu}(\theta)^{-1}\nabla_{\theta}J_{\beta}(\mu_{\theta}) = w $ 이 됨
            - $ E_{s \sim \rho^{\mu}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{\theta}\mu_{\theta}(s)^{\top}]^{-1}E_{s \sim \rho^{\mu}}[\nabla_{\theta}\mu_{\theta}(s)\nabla_{\theta}\mu_{\theta}(s)^{\top}w] = w $
        - 이 알고리즘은 COPDAC-Q 혹은 COPDAC-GQ 에서 $ \theta_{t+1} = \theta_{t} + \alpha_{\theta} \nabla_{\theta}\mu_{\theta}(s_{t})(\nabla_{\theta}\mu_{\theta}(s_{t})^{\top} w_{t}) $ 식을 $ \theta_{t+1} = \theta_{t} + \alpha_{\theta}w_{t} $ 로 바꿔주기만 하면 됨

## Experiments
### Continuous Bandit
- Stochastic Actor-Critic (SAC)과 COPDAC 간 성능 비교 수행
    - Action dimension이 커질수록 성능 차이가 심함
    - 빠르게 수렴하는 것을 통해 DPG의 data efficiency가 SPG에 비해 좋다는 것을 확인할 수 있지만, 반면, time-step이 증가할수록 SAC와 COPDAC 간 성능 차이가 줄어드는 것을 통해 성능 차이가 심하다는 것은 일정 time step 내에서만 해당하는 것이라고 유추해볼 수 있음
    - <img src="https://www.dropbox.com/s/hrkyq0s2f24z66r/Screenshot%202018-06-16%2017.47.38.png?dl=1">

### Continuous Reinforcement Learning
- COPDAC-Q가 SAC와 off-policy stochastic actor-critic(OffPAC-TD) 간 성능 비교 수행
    - COPDAC-Q의 성능이 약간 더 좋음
    - COPDAC-Q의 학습이 더 빨리 이뤄짐
    - <img src="https://www.dropbox.com/s/qdca4augapmzsxi/Screenshot%202018-06-16%2017.47.07.png?dl=1">
### Octopus Arm
- 목표: 6 segments octopus arm (20 action dimensions & 50 state dimensions)을 control하여 target을 맞추는 것
    - COPDAC-Q 사용 시, action space dimension이 큰 octopus arm을 잘 control하여 target을 맞춤
    - <img src="https://www.dropbox.com/s/xrxb0a52wntekld/Screenshot%202018-06-16%2017.46.28.png?dl=1" width=600px>
    - 기존 기법들은 action spaces 혹은 action 과 state spaces 둘 다 작은 경우들에 대해서만 실험했다고 하며, 비교하고 있지 않음.
        - 기존 기법들이 6 segments octopus arm에서 동작을 잘 안 했을 것 같긴한데, 그래도 실험해서 결과를 보여주지...왜 안 했을까?
    - 8 segment arm 동영상이 저자 홈페이지에 있다고 하는데, 안 보임.
- [참고] Octopus Arm 이란?
    - [OctopusArm Youtube Link](https://www.youtube.com/watch?v=AxeeHif0euY)
    - <img src="https://www.dropbox.com/s/950ycj06sudakjx/Screenshot%202018-06-16%2017.45.52.png?dl=1">
