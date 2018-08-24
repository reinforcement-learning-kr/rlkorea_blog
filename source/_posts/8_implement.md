---
title: PG Travel implementation story
date: 2018-08-23 14:18:32
tags: ["프로젝트", "피지여행"]
categories: 프로젝트
author: 이웅원, 장수영, 공민서, 양혁렬
subtitle: 피지여행 구현 이야기
---


# PG Travel implementation story

- 구현 코드 링크 : [https://github.com/reinforcement-learning-kr/pg_travel](https://github.com/reinforcement-learning-kr/pg_travel)


피지여행 프로젝트에서는 다음 7개 논문을 살펴보았습니다. 각 논문에 대한 리뷰는 이전 글들에서 다루고 있습니다. 

<a name="1"></a>

* [1] R. Sutton, et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation", NIPS 2000.
<a name="2"></a>
* [2] D. Silver, et al., "Deterministic Policy Gradient Algorithms", ICML 2014.
<a name="3"></a>
* [3] T. Lillicrap, et al., "Continuous Control with Deep Reinforcement Learning", ICLR 2016.
<a name="4"></a>
* [4] S. Kakade, "A Natural Policy Gradient", NIPS 2002.
<a name="5"></a>
* [5] J. Schulman, et al., "Trust Region Policy Optimization", ICML 2015.
<a name="6"></a>
* [6] J. Schulman, et al., "High-Dimensional Continuous Control using Generalized Advantage Estimation", ICLR 2016.
<a name="7"></a>
* [7] J. Schulman, et al., "Proximal Policy Optimization Algorithms", arXiv, https://arxiv.org/pdf/1707.06347.pdf.

강화학습 알고리즘을 이해하는데 있어서 논문을 보고 이론적인 부분을 알아가는 것이 좋습니다. 하지만 실제 코드로 돌아가는 것은 논문만 보고는 알 수 없는 경우가 많습니다. 따라서 피지여행 프로젝트에서는 위 7개 논문 중에 DPG와 DDPG를 제외한 알고리즘을 구현해보았습니다. 구현한 알고리즘은 다음 4개입니다. 이 때, TRPO와 PPO 구현에는 GAE(General Advantage Estimator)가 함께 들어갑니다. 
 
* Vanilla Policy Gradient [[1](#1)]
* TNPG(Truncated Natural Policy Gradient) [[4](#4)]
* TRPO(Trust Region Policy Optimization) [[5](#5)]
* PPO(Proximal Policy Optimization) [[7](#7)].

바닥부터 저희가 구현한 것은 아니며 다음 코드들을 참고해서 구현하였습니다. Vanilla PG의 경우 RLCode의 깃헙을 참고하였습니다.

* [OpenAI Baseline](https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)
* [Pytorch implemetation of TRPO](https://github.com/ikostrikov/pytorch-trpo)
* [RLCode Actor-Critic](https://github.com/rlcode/reinforcement-learning-kr/tree/master/2-cartpole/2-actor-critic)

GAE와 TRPO, PPO 논문에서는 Mujoco라는 물리 시뮬레이션을 학습 환경으로 사용합니다. 따라서 저희도 Mujoco로 처음 시작을 하였습니다. 하지만 Mujoco는 1달만 무료이고 그 이후부터 유료이며 확장성이 떨어집니다. Unity ml-agent는 기존 Unity를 그대로 사용하면서 쉽게 강화학습 에이전트를 붙일 수 있도록 설계되어 있습니다. Unity ml-agent에서는 저희가 살펴본 알고리즘 중에 가장 최신 알고리즘은 PPO를 적용해봤습니다. 기본적으로 제공하는 환경 이외에 저희가 customize 한 환경에서도 학습해봤습니다.  

* mujoco-py: [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)
* Unity ml-agent: [https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)

코드를 구현하고 환경에서 학습을 시키면서 여러가지 이슈들이 있었고 해결해내가는 과정이 있었습니다. 그 과정을 간단히 정리해서 공유하면 PG를 공부하는 분들께 도움일 될 것 같습니다. 저희가 구현한 순서대로 1. Mujoco 학습 2. Unity ml-agent 학습 3. Unity Curved Surface 로 이 포스트가 진행됩니다.

<br/>
## 1. Mujoco 학습
일명 "Continuous control" 문제는 action이 discrete하지 않고 continuous한 경우를 말합니다. Mujoco는 continuous control에 강화학습을 적용한 논문들이 애용하는 시뮬레이터입니다. 저희가 리뷰한 논문 중에서도 TRPO, PPO, GAE에서 Mujoco를 사용합니다. 따라서 저희가 처음 피지여행 알고리즘을 적용한 환경으로 Mujoco를 선택했습니다. 

Mujoco에는 Ant, HalfCheetah, Hopper, Humanoid, HumanoidStandup, InvertedPendulum, Reacher, Swimmer, Walker2d 과 같은 환경이 있습니다. 그 중에서 Hopper에 맞춰서 학습이 되도록 코드를 구현하였습니다. Mujoco 설치와 관련된 내용은 Wiki에 있습니다.

</br>
### 1.1 Hopper
Hopper는 외다리로 뛰어가는 것을 학습하는 것이 목표입니다. Hopper는 다음과 같이 생겼습니다. 
<img src="https://www.dropbox.com/s/wjxrelxyp014j3g/Screenshot%202018-08-23%2000.55.54.png?dl=1">

환경을 이해하려면 환경의 상태와 행동, 보상 그리고 학습하고 싶은 목표를 알아야합니다. 

- 상태 : 관절의 위치, 각도, 각속도
- 행동 : 관절의 가해지는 토크
- 보상 : 앞으로 나아가는 속도
- 목표 : 최대한 앞으로 많이 나아가기

즉 에이전트는 time step마다 관절의 위치와 각도를 받아와서 그 상태에서 어떻게 움직여야 앞으로 나아갈 수 있는지를 학습해야 합니다. 행동은 discrete action이 아닌 continuous action으로 -1과 1사이의 값을 가집니다. 만약 행동이 -1이라면 해당 관절에 시계반대방향으로 토크를 주는 것이고 행동이 +1이라면 해당 관절에 시계방향으로 토크를 주는 것입니다. 

continuous action을 주는 방법은 네트워크(Actor)의 output layer에서 activation function으로 tanh와 같은 것을 사용해서 continuous한 값을 출력하는 것이 있습니다. 하지만 피지여행 코드 구현에서는 action을 gaussian distribution에서 sampling 하였습니다. 이렇게 하면 분산을 일정하게 유지하면서 지속적인 exploration을 할 수 있습니다. 간단하게 그림으로 보자면 다음과 같습니다. 
<img src="https://www.dropbox.com/s/94g01zdxyf5oxu1/Screenshot%202018-08-23%2001.20.21.png?dl=1">

네트워크 구조와 행동을 선택하는 부분은 다음과 같습니다. Hidden Layer의 activation function으로 tanh를 사용했으며(ReLU를 테스트해보지는 않았습니다. 기존 TRPO, PPO 구현들과 논문에서 tanh를 사용하기 때문에 저희도 사용했습니다. 뒤에 유니티 환경에서는 Swish라는 것을 사용합니다.) log std를 0으로 고정함으로서 일정한 폭을 가지는 분포를 만들어낼 수 있습니다. 이 분포로부터 action을 sampling 합니다.

<img src="https://www.dropbox.com/s/xfl9zxies0lmpm1/Screenshot%202018-08-23%2001.20.44.png?dl=1">

</br>
### 1.2 Vanilla PG
Vanilla PG는 Actor-Critic의 가장 간단한 형태입니다. Vanilla PG는 이후의 구현에 대한 baseline이 됩니다. 구현이 가장 간단하면서 학습이 안되는 것은 아닙니다. 따라서 코드 전체 구조를 잡는데 Vanilla PG를 짜는 것이 도움이 됩니다. 전반적인 코드 구조는 다음과 같습니다.

- iteration 마다 일정한 step 수만큼 환경에서 진행해서 샘플을 모은다
- 모은 샘플로 Actor와 Critic을 학습한다
- 반복한다
 

```python
episodes = 0
for iter in range(15000):
    actor.eval(), critic.eval()
    memory = deque()
    
    while steps < 2048:
        episodes += 1
        state = env.reset()
        state = running_state(state)
        score = 0
        for _ in range(10000):
            mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
            action = get_action(mu, std)[0]
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)

            if done:
                mask = 0
            else:
                mask = 1

            memory.append([state, action, reward, mask])
            state = next_state

            if done:
                break
                
    actor.train(), critic.train()
    train_model(actor, critic, memory, actor_optim, critic_optim)
```

memory에 sample을 저장할 때 sample은 state와 action, reward, mask(마지막 state일 경우 0 나머지 1)입니다. mask의 경우 뒤에서 return이나 advantage를 계산할 때 사용됩니다. 또 하나 염두에 두어야할 것은 running_state 입니다. running_state는 input으로 들어오는 state의 scale이 일정하지 않기 때문에 사용합니다. 즉 state의 각 dimension을 평균 0 분산 1로 standardization 하는 것입니다. 따라서 모델을 저장할 때 각 dimension 마다의 평균과 분산도 같이 저장해서 테스트할 때 불러와서 사용해야 합니다.

Vanilla PG의 경우 학습 부분이 상당히 간단합니다. 다음 코드를 보시면 메모리에서 state, action, reward, mask를 꺼냅니다. reward와 mask를 통해 return을 구할 수 있고 이 return을 통해 actor를 업데이트 할 수 있습니다 (REINFORCE 알고리즘을 떠올리시면 됩니다). 여기서 critic이 하는 일은 없지만 뒤의 알고리즘들과 코드의 통일성을 위해 fake로 넣어놨습니다. Return은 평균을 빼고 분산으로 나눠서 standardize 합니다. 

```python
def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    returns = get_returns(rewards, masks)
    train_critic(critic, states, returns, critic_optim)
    train_actor(actor, returns, states, actions, actor_optim)
    return returns
```

이 코드로 Hopper 환경에서 학습한 그래프는 다음과 같습니다. 
<img src="https://www.dropbox.com/s/asoysfuk76zs1dk/Screenshot%202018-08-23%2001.30.58.png?dl=1">

</br>
### 1.3 TNPG
NPG를 이용한 parameter update 식은 다음과 같습니다. 

$$\bar{w}=F(\theta)^{-1}\nabla\eta(\theta)$$

NPG를 구현하려면 KL-divergence의 Hessian의 inverse를 구해야하는 문제가 생깁니다. 현재와 같이 Deep Neural Network를 쓰는 경우에 Hessian의 inverse를 직접적으로 구하는 것은 computationally inefficient 합니다. 따라서 직접 구하지 않고 Conjugate gradient 방법을 사용해서 Fisher Vector Product ($$F^{-1}g$$)를 구합니다. 이러한 알고리즘을 Truncated Natural Policy Gradient(TNPG)라고 부릅니다. 

TNPG에서 parameter update를 구하는 과정은 다음과 같습니다. 
1. Return 구하기
2. Critic 학습하기
3. logp * return --> loss 구하기
4. loss의 미분과 kl-divergence의 2차 미분을 통해 step direction 구하기
5. 구한 step direction으로 parameter update

```python
def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    # ----------------------------
    # step 1: get returns
    returns = get_returns(rewards, masks)

    # ----------------------------
    # step 2: train critic several steps with respect to returns
    train_critic(critic, states, returns, critic_optim)

    # ----------------------------
    # step 3: get gradient of loss and hessian of kl
    loss = get_loss(actor, returns, states, actions)
    loss_grad = torch.autograd.grad(loss, actor.parameters())
    loss_grad = flat_grad(loss_grad)
    step_dir = conjugate_gradient(actor, states, loss_grad.data, nsteps=10)

    # ----------------------------
    # step 4: get step direction and step size and update actor
    params = flat_params(actor)
    new_params = params + 0.5 * step_dir
    update_model(actor, new_params)
    
```

conjugate gradient 코드는 OpenAI baseline에서 가져왔습니다. 이 코드는 원래 John schulmann 개인 repository에 있는 그대로 사용하는 것입니다. nsteps 만큼 iterataion을 반복하며 결국 x를 구하는 것인데 이 x가 step direction 입니다. 

```python
# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py
def conjugate_gradient(actor, states, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = fisher_vector_product(actor, states, p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x
```

fisher_vector_product는 kl-divergence의 2차미분과 어떠한 vector의 곱인데 p는 처음에 gradient 값이었다가 점차 업데이트가 됩니다. kl-divergence의 2차 미분을 구하는 과정은 다음과 같습니다. 일단 kl-divergence를 현재 policy에 대해서 구한 다음에 actor parameter에 대해서 미분합니다. 이렇게 미분한 gradient를 일단 flat하게 핀 다음에 p라는 벡터와 곱해서 하나의 값으로 만듭니다. 그 값을 다시 actor의 parameter로 만듦으로서 따로 KL-divergence의 2차미분을 구하지않고 Fisher vector product를 구할 수 있습니다.

```python
def fisher_vector_product(actor, states, p):
    p.detach()
    kl = kl_divergence(new_actor=actor, old_actor=actor, states=states)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)  # check kl_grad == 0

    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + 0.1 * p
```

TNPG 학습 결과는 다음과 같습니다. 
<img src="https://www.dropbox.com/s/uc4c0s00qbs33nr/Screenshot%202018-08-23%2001.53.17.png?dl=1">

</br>
### 1.4 TRPO
TRPO와 NPG가 다른 점은 surrogate loss 사용과 trust region 입니다. 하지만 실제로 구현해서 학습을 시켜본 결과 trust region을 넘어가서 back tracking line search를 하는 경우는 거의 없습니다. 따라서 주된 변화는 surrogate loss에 있다고 보셔도 됩니다. Surrogate loss에서 advantage function을 사용하는데 본 코드 구현에서는 GAE를 사용하였습니다. TRPO 업데이트 식은 다음과 같습니다. Q function 위치에 GAE가 들어갑니다.

$$
\begin{align}
\max\_\theta\quad &E\_{s\sim\rho\_{\theta\_\mathrm{old} },a\sim q}\left[\frac{\pi\_\theta(a\vert s)}{q(a\vert s)}Q\_{\theta\_\mathrm{old} }(s,a)\right] \\\\
\mathrm{s.t.\ }&E\_{s\sim\rho\_{\theta\_\mathrm{old} }}\left[D\_\mathrm{KL}\left(\pi\_{\theta\_\mathrm{old} }(\cdot\vert s) \parallel \pi\_\theta(\cdot\vert s)\right)\right] \leq \delta
\end{align}
$$


GAE를 구하는 코드는 다음과 같습니다. GAE는 td-error의 discounted summation이라고 볼 수 있습니다. 마지막에 advants를 standardization 하는 것은 return에서 하는 것과 같은 효과를 봅니다. 하지만 standardization을 안하고 실험을 해보지는 않았습니다.

```python
def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - \
                    values.data[t]
        running_advants = running_tderror + hp.gamma * hp.lamda * \
                          running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants
```

Surrogate loss를 구하는 코드는 다음과 같습니다. Advantage function(GAE)를 구하고 나면 이전 policy와 현재 policy 사이의 ratio를 구해서 advantage function에 곱하면 됩니다. 이 때 사실 old policy와 new policy는 값은 같지만 old policy는 clone()이나 detach()를 사용해서 update가 안되게 만들어줍니다.

```python
def surrogate_loss(actor, advants, states, old_policy, actions):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    advants = advants.unsqueeze(1)

    surrogate = advants * torch.exp(new_policy - old_policy)
    surrogate = surrogate.mean()
    return surrogate

```

Actor의 step direction을 구하는 것은 TNPG와 동일합니다. TNPG에서는 step direction으로 바로 업데이트 했지만 TRPO는 다음과 같은 작업을 해줍니다. Full step을 구하는 과정이라고 볼 수 있습니다. 

```python
# ----------------------------
# step 4: get step direction and step size and full step
params = flat_params(actor)
shs = 0.5 * (step_dir * fisher_vector_product(actor, states, step_dir)
             ).sum(0, keepdim=True)
step_size = 1 / torch.sqrt(shs / hp.max_kl)[0]
full_step = step_size * step_dir

```

이렇게 full step을 구하고 나면 Trust region optimization 단계에 들어갑니다. expected improvement는 구한 step 만큼 parameter space에서 움직였을 때 예상되는 performance 변화입니다. 이 값은 kl-divergence와 함께 trust region 안에 있는지 밖에 있는지 판단하는 근거가 됩니다. expected improve는 출발점에서의 gradient * full step으로 구합니다. 그리고 10번을 돌아가며 Back-tracking line search를 실시합니다. 처음에는 full step 만큼 가본 다음에 kl-divergence와 emprovement를 통해 trust region 안이면 루프 탈출, 밖이면 full step을 반만큼 쪼개서 다시 이 과정을 반복합니다.  

```python
# ----------------------------
# step 5: do backtracking line search for n times
old_actor = Actor(actor.num_inputs, actor.num_outputs)
update_model(old_actor, params)
expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
expected_improve = expected_improve.data.numpy()

flag = False
fraction = 1.0
for i in range(10):
    new_params = params + fraction * full_step
    update_model(actor, new_params)
    new_loss = surrogate_loss(actor, advants, states, old_policy.detach(),
                              actions)
    new_loss = new_loss.data.numpy()
    loss_improve = new_loss - loss
    expected_improve *= fraction
    kl = kl_divergence(new_actor=actor, old_actor=old_actor, states=states)
    kl = kl.mean()

    print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
          'number of line search: {}'
          .format(kl.data.numpy(), loss_improve, expected_improve[0], i))

    # see https: // en.wikipedia.org / wiki / Backtracking_line_search
    if kl < hp.max_kl and (loss_improve / expected_improve) > 0.5:
        flag = True
        break

    fraction *= 0.5
```

Critic의 학습은 단순히 value function과 return의 MSE error를 계산해서 loss로 잡고 loss를 최소화하도록 학습합니다. TRPO 학습 결과는 다음과 같습니다.
<img src="https://www.dropbox.com/s/rc9hxsx1kvokcrv/Screenshot%202018-08-23%2013.36.51.png?dl=1">

</br>
### 1.5 PPO
PPO의 장점을 꼽으라면 GPU 사용하기 좋고 sample efficiency가 늘어난다는 것입니다. TNPG와 TRPO의 경우 한 번 모은 sample은 모델을 단 한 번 업데이트하는데 사용하지만 PPO의 경우 몇 mini-batch로 epoch를 돌리기 때문입니다. GAE를 사용한다는 것은 같고 Conjugate gradient나 Fisher vector product나 back tracking line search가 다 빠집니다. 대신 loss function clip으로 monotonic improvement를 보장하게 학습합니다. 따라서 코드가 상당히 간단해집니다. 

다음 코드 부분이 PPO의 전체라고 봐도 무방합니다. PPO는 다음과 같은 순서로 학습합니다. 

- batch를 random suffling하고 mini batch를 추출
- value function 구하기
- critic loss 구하기 (clip을 사용해도 되고 TRPO와 같이 그냥 학습시켜도 됌)
- surrogate loss 구하기
- surrogate loss clip해서 actor loss 만들기
- actor와 critic 업데이트

Actor의 loss를 구하는 것은 다음 식의 값을 구하는 것입니다. 이 식을 구하려면 ratio에 한 번 클립하고 loss 값을 한 번 min을 취하면 됩니다.

$$L^{CLIP}(\theta) = \hat{E}_t [min(r_t(\theta) \, \hat{A}_t,  clip(r_t(\theta), 1-\epsilon, 1+\epsilon) \, \hat{A}_t)]$$

이 코드 구현에서는 actor와 critic을 따로 모델로 만들어서 따로 따로 업데이트를 하지만 하나로 만든다면 loss로 한 번만 업데이트하면 됩니다. 또한 entropy loss를 최종 loss에 더해서 regularization 효과를 볼 수도 있습니다. Critic loss에 clip 해주는 것은 OpenAI baseline의 ppo2 코드를 참조하였습니다.

```python
# step 2: get value loss and actor loss and update actor & critic
for epoch in range(10):
    np.random.shuffle(arr)

    for i in range(n // hp.batch_size):
        batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
        batch_index = torch.LongTensor(batch_index)
        inputs = torch.Tensor(states)[batch_index]
        returns_samples = returns.unsqueeze(1)[batch_index]
        advants_samples = advants.unsqueeze(1)[batch_index]
        actions_samples = torch.Tensor(actions)[batch_index]
        oldvalue_samples = old_values[batch_index].detach()

        loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                     old_policy.detach(), actions_samples,
                                     batch_index)

        values = critic(inputs)
        clipped_values = oldvalue_samples + \
                         torch.clamp(values - oldvalue_samples,
                                     -hp.clip_param,
                                     hp.clip_param)
        critic_loss1 = criterion(clipped_values, returns_samples)
        critic_loss2 = criterion(values, returns_samples)
        critic_loss = torch.max(critic_loss1, critic_loss2).mean()

        clipped_ratio = torch.clamp(ratio,
                                    1.0 - hp.clip_param,
                                    1.0 + hp.clip_param)
        clipped_loss = clipped_ratio * advants_samples
        actor_loss = -torch.min(loss, clipped_loss).mean()

        loss = actor_loss + 0.5 * critic_loss

        critic_optim.zero_grad()
        loss.backward(retain_graph=True)
        critic_optim.step()

        actor_optim.zero_grad()
        loss.backward()
        actor_optim.step()
```

PPO의 학습 결과는 다음과 같습니다. 
<img src="https://www.dropbox.com/s/rkxa836ap931kbd/Screenshot%202018-08-23%2013.50.57.png?dl=1">



## 2. Unity ml-agent 학습
Mujoco Hopper(half-cheetah와 같은 것도)에 Vanilla PG, TNPG, TRPO, PPO를 구현해서 적용했습니다. Mujoco의 경우 이미 Hyper parameter와 같은 정보들이 논문이나 블로그에 있기 때문에 상대적으로 continuous control로 시작하기에는 좋습니다. 맨 처음에 말했듯이 Mujoco는 1달만 무료이고 그 이후부터 유료이며 확장성이 떨어집니다. 좀 더 general한 agent를 학습시키기에 좋은 환경이 필요합니다. 따라서 Unity ml-agent를 살펴봤습니다. Repository는 다음과 같습니다. 
- [Unity ml-agent repository](https://github.com/Unity-Technologies/ml-agents)
- [Unity ml-agent homepage](https://unity3d.com/machine-learning/)
<img src="https://www.dropbox.com/s/lapholj8r4nxmb1/Screenshot%202018-08-24%2013.41.31.png?dl=1">

Unity ml-agent는 기존 Unity를 그대로 사용하면서 쉽게 강화학습 에이전트를 붙일 수 있도록 설계되어 있습니다. Unity ml-agent에서는 저희가 살펴본 알고리즘 중에 가장 최신 알고리즘은 PPO를 적용해봤습니다. 기본적으로 제공하는 환경 이외에 저희가 customize 한 환경에서도 학습해봤습니다.  

### 2.1 환경 만들기 : 

유니티를 한번도 접해보지않은 상황에서 환경을 어떻게 만들어야할지 감이 잡히지 않았다.
우선적으로 3DBall 예제가 있는 듯 했지만, 친절한듯 친절하지않은 예제 문서로 인해 인터넷에서 선구자들의 지식을 구했다. 예를 들면, 파이썬으로 브레인을 제어하기위해선 External 속성으로 꼭 바꿔줘야하는 것 등이다. 3DBall 환경 구축에 성공했을 때의 화면이다.
<img src="https://i.imgur.com/5S5qGUI.gif" width=500px/>

아래의 링크페이지에서 유니티 ML-agent 환경 구축에 대한 정보를 얻었다.
[https://medium.com/@indiecontessa/setting-up-a-python-environment-with-tensorflow-on-macos-for-training-unity-ml-agents-faf19d71201](https://medium.com/@indiecontessa/setting-up-a-python-environment-with-tensorflow-on-macos-for-training-unity-ml-agents-faf19d71201)

또한 브레인, 아카데미 등 ML-agent에 대한 보다 자세한 사항은 장수영님께서 설명해주셨다. 한번 프로세스를 밟고나면 모든게 쉽다. 우리의 본 목표인 Walker환경은 기본 템플릿으로 주어진 것만해도 충분했다.

</br>
### 2.2 ML-agent + PyTorch
ML-agent와 PyTorch를 연결하는 데에는 이웅원님께서 사전에 작성해놓은 코드가 있었다.아래의 사진은 이웅원님이 코드 로직을 설명해주는 상황이다.
<img src="https://i.imgur.com/1aR2Z77.png" width=500px>

Walker환경에 적용했을 당시, 웅원님과 구현팀의 이슈는 동일한 학습 시간이 흘러도 Unity Baseline의 점수만큼 도달하기가 힘들다는 것이었다. 1차적으로 생각할 수 있었던 것은 웅원님의 코드는 단일 에이전트를 학습시키는 코드였고, Unity Baseline의 코드는 다수의 에이전트를 학습시키는 코드였다. 그 외의 코드 상의 버그나 PPO 알고리즘의 구현에서 문제가 있다고 판단하기가 어려워 Unity Baseline과 로직 비교를 하는 시간이 필요했다.

</br>
### 2.3 Unity Baseline Code-Review
Unity-Baseline코드를 보며 어려웠던 점은, Unity ML agent로 제공하는 예제 환경에 모두 적용가능한 에이전트를 구현하기 위해 코드의 양과 복잡도가 상당했다는 점이었다. 사실, 팀에 필요한 로직은 정말정말 일부에 불과했으나, 헤매고 헤매다 발견하고나서야 알아낸 사실이었다. 아래의 그림은 코드리뷰를 할 당시에 작성했던 마인드맵이다.
<img src="https://i.imgur.com/YeaEntG.png">

Unity-Baseline을 보면서 크게 두가지에 초점을 맞춰 코드리뷰를 진행하였다. 첫째로 멀티 에이전트 학습 로직을 알아내기, 둘째로 세부적으로 어떤 차이가 있는지 면밀히 캐내보기 였고 Swish라는 활성화 함수, 레이어 개수나 노드 개수, learning_rate 같은 하이퍼 파라미터의 차이가 있었고 양혁렬님께서 멀티에이전트 학습 로직을 파악해 PyTorch로 구현에 성공했다.



## 3. Unity Curved Surface 제작 및 학습기
### 3.1 도입
처음부터 만들 수는 없으니, 적당한 것을 찾아볼까? (**중요** Price : 0)
<center><img src="https://www.dropbox.com/s/e0tsp3e3c9uq2zh/Screenshot%202018-08-23%2000.19.14.png?dl=1"></center>

오...괜찮아보이네. 점 찍어서 대강 굴곡 형태 잡아주면 알아서 곡면 만들어주는 듯?
<center><img src="https://www.dropbox.com/s/3ppmxotrf6qhzaf/Screenshot%202018-08-23%2000.20.25.png?dl=1"></center>

일단 곡면이 잘 만들어지는 지 제대로 돌아가는 지 보면....
<center><img src="https://www.dropbox.com/s/2e8yqvqj1a4th27/slope_walker_ball.gif?dl=1"></center>
잘 되네.

### 3.2 실전
그럼 이제 여러 에이전트들이 학습할 수 있도록 학습 환경을 만들자~!
#### 환경 구축
[초기]
<center><img src="https://www.dropbox.com/s/m492xsfp4bolmz5/Screenshot%202018-08-23%2000.36.06.png?dl=1">

[후기]
<img src="https://www.dropbox.com/s/idbov4wtd6jeqb2/Screenshot%202018-08-23%2000.36.54.png?dl=1">
- 초기에서 후기로 가며 변경 사항들
    - Slope 길이 - 길이를 기존 plane 과 동일하게 했더니, 오르막 올라가는 부분이 학습이 잘 안 됨. 그래서 좀 길이를 줄임.
    - 내리막 경사 - 너무 경사지면 학습이 잘 안 되고, 너무 완만하니 내리막 티가 잘 안 나고...
    - 오르막 경사 - 막연히 오르막이 좀 더 어려울테니, 오르막 길이를 길게, 높이를 낮게 해서 경사를 내리막보다는 완만히 하자.
    - 색 - 좀 밝은 색으로 하자.

#### 에이전트 튜닝
너무 빨리 쓰러져서...혹시나 나아질까...발 각도도 변경해보고...
<img src="https://www.dropbox.com/s/znvikbeoj7gku0u/Screenshot%202018-08-23%2000.38.22.png?dl=1">

#### 3.3 결과
그래도 생각보다는 잘 못 걸음..
<img src="https://www.dropbox.com/s/t5ngr0io4xeex6y/curved-736-overview.gif?dl=1">

<audio src="https://www.dropbox.com/s/rvtsgj5w3kmh5y1/Hand-in-Hand.mp3?dl=1" controls="controls" loop="" preload="auto"></audio>

### 3.4 부푼 꿈
#### 환경을 멋있게
건강엔 등산이지. 산을 타보자. (이건 Asset Store 에 있는 "Fire Progation(by Lewis Ward)"" 을 활용)
병렬 학습은 시켜야겠고...
한 평면에 여러 에이전트를 위치시키기는 힘드니, 쌓자!
<img src="https://www.dropbox.com/s/uom140d4w3ir4e2/env-mountain.png?dl=1">

#### 포기
급경사로 인해 학습 망함...
시간 관계 상 생략하기로...