---
title: Let's do Inverse RL Guide
date: 2019-01-22
tags: ["프로젝트", "GAIL하자!"]
categories: 프로젝트
author: 이동민, 이승현
subtitle: Let's do Inverse RL Guide
---

---

# 0. Inverse RL의 세계로

반갑습니다! 저희는 Inverse RL을 흐름을 살펴보기 위해 모인 IRL 프로젝트 팀입니다.

강화학습에서 reward라는 요소는 굉장히 중요합니다. 왜냐하면 agent라는 아이가 유일하게 학습할 수 있는 요소이기 때문입니다. 일반적으로 강화학습에서는 사람이 reward를 일일히 정해주지만, 실제로 그 reward에 따라 "desirable"  action이 나오지 않을 수도 있습니다. 여기서 생각해볼 수 있는 것이 바로 "expert"의 행동을 통해 reward를 찾는 것입니다.

저희는 Andrew Ng의 논문인 Linear IRL과 Pieter Abbeel의 논문인 APP를 필두로 하여 MMP, MaxEnt, 그리고 보통 IRL을 통해 얻어진 reward로 다시 RL을 풀어서 policy를 얻어야하지만, 이 과정을 한번에 풀어버리는 GAIL, 최근 들어 GAIL을 뛰어넘는 VAIL까지 살펴보고자 합니다.

<center> <img src="../../../../img/irl/rl_irl.png" width="1100"> </center>

논문의 순서는 다음과 같습니다.

1. [Linear_IRL](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
2. [APP](http://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Abbeel+Ng:2004.pdf)
3. [MMP](https://www.ri.cmu.edu/pub_files/pub4/ratliff_nathan_2006_1/ratliff_nathan_2006_1.pdf)
4. [MaxEnt](http://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)
5. [GAIL](https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf)
6. [VAIL](https://arxiv.org/pdf/1810.00821.pdf)

위와 같이 총 6가지 논문들을 리뷰하여 블로그로 정리하였습니다. 각 순서에 맞춰 보시는 것을 권장해드립니다.

<br><br>

# 1. \[Linear IRL\] Algorithms for Inverse Reinforcement Learning

[Linear IRL 여행하기](https://reinforcement-learning-kr.github.io/2019/01/28/1_linear-irl/)

Inverse RL(IRL)은 expert의 demonstrations(trajectories)가 있을 때 이것을 통해 expert의 optimal policy $\pi$를 찾고, 그 policy로 IRL을 진행하여 reward function $R$을 찾는 것을 말합니다. 다른 방법 중에는 어떠한 상태에서 어떠한 행동을 할지를 직접 모델링하는 Behavioral Cloning(BC)이라는 것이 있지만, 충분한 data의 양이 필요하고 시간이 지남에 따라 에러가 누적되어 그 누적된 에러 때문에 시간이 지남에 따라 성능이 많이 떨어지게 됩니다. 쉽게 말해 정해진 경로가 있을 때 경로에 조금만 틀어져도 에러가 생기는데 이 에러가 계속 누적되기 때문에 나중에는 크게 달라져버린다는 것입니다. 이러한 단점 때문에 reward function을 모델링하는 IRL 방법이 개발되었습니다.

Andrew Y. Ng 교수님이 저자로 쓴 이 논문부터 공식적으로 Inverse RL(IRL)을 언급합니다. Imitation Learning은 무엇인지, 그 중에서도 IRL이 무엇인지, 장점은 무엇인지, 왜 필요한지에 대해서 말하는 논문입니다. 또한 IRL을 통해 reward를 얻어 RL을 하는 실질적인 학습을 말하는 논문보다는 reward function을 어떻게 찾을 지에 대해서 말하고 있고 이에 따른 알고리즘들을 다루는 논문입니다. 논문에는 따로 언급되어 있지 않지만 IRL은 **ill-posed problem** 이라고 말할 수 있습니다. ill-posed problem에 대해서는 논문을 정리한 블로그를 참고해주시면 감사하겠습니다.

제안하는 알고리즘으로는 총 3가지 입니다.

1) state space가 finite할 때, model(dynamics)을 알고 있을 때
2) state space가 large or infinite할 때, model(dynamics)을 알고 있을 때
3) expert의 trajectories를 통해서만 policy를 알 때, model(dynamics)을 모를 때

이 논문에서 제안하는 아이디어는 Linear Programming(LP)입니다. 정리한 블로그에 간단하게 LP를 소개해두었으니 참고바랍니다.

자! 그럼 우리 모두 같이 IRL 여행을 시작해볼까요?

[Linear IRL 여행하기](https://reinforcement-learning-kr.github.io/2019/01/28/1_linear-irl/)

<br><br>

# 2. \[APP\] Apprenticeship Learning via Inverse Reinforcement Learning

[APP 여행하기](https://reinforcement-learning-kr.github.io/2019/02/01/2_app/)
[APP Code](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/app)

APP는 자동차 주행 혹은 헬리콥터의 주행의 경로 탐색을 문제에서 reward shaping의  어려움을 해결하고자 시작한 논문입니다. reward shaping은 domain에 대한 많은 knowledge가 필요합니다. 따라서 어떤 상태일때 더 많은 보상을 줄지에 대한 기준이 되는 feature들을 reward designer는 직관적으로 알고 있습니다. 예를들어 안전한 자동차 주행이란 task에서는 '앞차와의 거리', '현재 차선', '자동차의 충돌 여부' 등을 반영하여 직접 manual 하게 reward를 만들 어 나갑니다. 하지만 환경과 agent에 따라 매번 어느 정도의 거리나 차선이 좋은지의 중요도를 학습하는것은 매우 힘들고도 비효율적인 일입니다. 

이를 해결하고자 APP는 reward를 domain knowlege를 활용한 feature들의 선형조합으로 표현한다면, expert의 시연만으로도 feature들 간의 weight를 쉽게 자동으로 학습할 수 있을것이라고 가정하고 이를 **feature expectation**이라는 개념을 도입하여 해결합니다.

이 feature를 사용해 reward를 학습하는 개념은 이후 나오는 imitation learning 논문들이 제안하는 많은 접근방식의 근간이 되므로 IRL이라는 먼 길을 가기에 앞서 APP에서 자세히 이해하고 넘어갈것을 추천드립니다! 

[APP 여행하기](https://reinforcement-learning-kr.github.io/2019/02/01/2_app/)
[APP Code](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/app)

<br><br>

# 3. \[MMP\] Maximum Margin Planning

[MMP 여행하기](https://reinforcement-learning-kr.github.io/2019/02/07/3_mmp/)

Robotics의 관점에서 IRL을 바라본 이 논문은 APP에서 더 나아가 어떻게 하면 효율적으로 expert의 behavior을 모방할 수 있을 지를 고민하였습니다. Robotics에서는 일반적으로 perception subsystem과 planning subsystem으로 autonomy software를 partition함으로써 Long-horizon goal directed behavior를 찾습니다. 여기서 Perception system은 다양한 model과 환경의 features를 계산합니다. 그리고 Planning system은 cost-map을 input으로 두고, 그 input을 통해 minimal risk (cost) path를 계산합니다. 하지만 perception의 model부터 planner에 대한 cost까지 학습하는 것은 어렵기 때문에 새로운 방법인 perception features부터 planner에 대한 cost까지 (Perception + Planning) mapping하는 것을 자동화하는 방법을 제시합니다.

또한 일반적으로 Supervised learning techniques를 통해 sequential, goal-directed behavior에 대한 imitation learning은 어렵기 때문에 APP에서 제시했던 QP(or SVM)방법에 Soft Margin term을 추가하여 슬랙변수를 가지는 SVM을 사용하였고, 더 나아가 subgradient method를 이용하여 알고리즘을 좀 더 쉽고 빠르게 구할 수 있도록 만들었습니다.

논문에서 가장 중요한 개념은 **state-action visitation frequency counts** 라는 것입니다. 지금까지는 어떠한 상태에서 어떠한 행동을 할 확률인 policy를 이용했다면 앞으로는 확률의 개념이 아니라 얼마나 방문 했는지를 말하는 빈도수, 즉 count의 개념으로 접근 하는 것입니다. IRL의 궁극적인 목표를 다르게 말해보면, expert가 어떠한 행동을 했을 때 여기서의 state-action visitation frequency를 구하고 expert와 최대한 비슷한 visitation frequency를 만들어내는 reward를 찾는 것입니다. 또한 RL의 problem은 reward가 주어졌을 때 이 reward의 expected sum을 최대로 하는 policy를 찾는 것인데 RL의 dual problem은 visitation frequency를 찾는 것이라고도 말할 수 있다. 다시 말해 optimal policy와 optimal visitation frequency는 1:1관계이라고 말할 수 있습니다.

논문의 이론과 이에 따른 내용의 양이 상당히 많은 논문입니다. 이 논문을 보시는 분들이 저희가 만든 자료가 도움이 되어 끝까지 보실 수 있었으면 좋겠습니다!

[MMP 여행하기](https://reinforcement-learning-kr.github.io/2019/02/07/3_mmp/)

<br><br>

# 4. \[MaxEnt\] Maximum Entropy Inverse Reinforcement Learning

[MaxEnt 여행하기](https://reinforcement-learning-kr.github.io/2019/02/10/4_maxent/)
[MaxEnt Code](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/maxent)

SVM 전성시대가 막이 내리고 IRL도 이 논문부터 본격적으로 확률적인 개념을 이용하여 learning하는 방법을 제시합니다. 이 논문에서는 먼저 그 당시에 크게 두 갈래로 나눠지는 방법론인 "MMP"와 "APP"에 대해 설명하고 각 방법론에 대한 단점들을 말해줍니다. 개인적으로 다시 한 번 remind를 해주는 부분이 있어서 논문을 읽기가 더 좋았던 것 같습니다. MaxEnt의 경우 MMP의 궤가 아닌 APP의 궤로써 APP에서의 ambiguity를 어떻게 하면 해결할 수 있을 지를 말하는 논문입니다. 그 방법으로, **the principle of maximum entropy**에 기반한 확률적인 접근을 이용합니다.

결국 무엇을 하고 싶은 것이냐면, state visitation frequency count의 개념을 이용하여 이전의 논문처럼 expert와 learner 사이의 frequency count를 matching하고 싶은 것인데, IRL은 전형적으로 ill-posed problem이기 때문에 각각의 policy는 많은 reward function에 대해 optimal 할 수 있고, 이에 따라 많은 policy들은 같은 feature count들을 유도하기 때문에 ambiguity가 발생하게 됩니다. 따라서 the principle of maximum entropy 이론을 통해 어떠한 distribution의 parameter가 되는 $\theta$를 maximization하는 쪽으로 잡아나가겟다는 것입니다.

뒤이어 state visitation frequency를 더 효율적으로 구하기 위한 algorithm이 나옵니다만, dynamics를 알 때에 쓰이는 것이므로 저희는 구현할 때 q-learning을 통한 sampling을 하는 방법을 사용하였기 때문에 깊게 다루지 않았습니다.

다음 논문인 GAIL 논문을 보시기전에 이 논문을 꼭 이해하고 GAIL 논문을 보시는 것을 추천해드립니다!

[MaxEnt 여행하기](https://reinforcement-learning-kr.github.io/2019/02/10/4_maxent/)
[MaxEnt Code](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/maxent)

<br><br>

# 5. \[GAIL\] Generative Adversarial Imitation Learning

[GAIL 여행하기](https://reinforcement-learning-kr.github.io/2019/02/13/5_gail/)
[GAIL Code](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mujoco/gail)

페이스북 인공지능 연구팀의 리더이자 딥러닝의 압지라 불리는 얀 르쿤(Yann Lecun) 교수님은 Ian Goodfellow가 2014년에 내놓은 **Generative Adversarial Networks (GAN)** 을 가리켜 최근 10년간 머신러닝 분야에서 가장 혁신적인 아이디어라고 말했습니다. 이후 많은 곳에 GAN을 사용할 수 있겠다고 생각한 연구자들은 GAN을 활용한 연구들을 통해 뛰어난 성능을 보여주며 각종 인공지능 학회를 뜨겁게 달궜습니다. 이는 Imitation learning에도 마찬가지였으며, 이 시기에 나온 논문이 바로 GAIL입니다. 

APP, MMP 등의 아주 초기 IRL 논문에서부터 핵심적으로 사용되는 알고리즘은 SVM과 같이 expert와 learner의 performance margin을 최대화 하는 방향으로 reward function을 학습하는것입니다. 이는 어떻게 보면 두 policy를 더 잘 구분하고자 하는 것이며 GAN에서 말하는 discriminator와 generator의 성질과 매우 유사합니다. 저자는 이 점을 활용하여 새로운 cost regularizer를 제안함으로써 Immitation learning과 GAN을 연결짓습니다. 

GAIL의 또한 가지 특징은 최근에 나온 논문인 만큼 policy approximator로서 Neural network를 사용한다는 점입니다. 이 때문에 딥러닝에 익숙하신 분이 지금까지 IRL 여행의 수많은 머신러닝 수식들로 힘드셨다면, GAIL은 기술적으로 무르익은 단계의 논문임에도 오히려 더 명쾌하다는 느낌을 받으실 수 도 있습니다.

자, 이제 IRL 여행에 막바지에 이르렀습니다. 최적화와 관련해서 어려운 개념들이 많이 나오겠지만 힘을내서 끝까지 마무리 지어 봅시다!

[GAIL 여행하기](https://reinforcement-learning-kr.github.io/2019/02/13/5_gail/)
[GAIL Code](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mujoco/gail)

<br><br>

# 6. \[VAIL\] Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow

[VAIL 여행하기](https://reinforcement-learning-kr.github.io/2019/02/25/6_vail/)
[VAIL Code](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mujoco/vail)

수정중..

[VAIL 여행하기](https://reinforcement-learning-kr.github.io/2019/02/25/6_vail/)
[VAIL Code](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mujoco/vail)

<br><br>

# Team

이동민 : [Github](https://github.com/dongminleeai), [Facebook](https://www.facebook.com/dongminleeai)

윤승제 : [Github](https://github.com/sjYoondeltar), [Facebook](https://www.facebook.com/seungje.yoon)

이승현 : [Github](https://github.com/Clyde21c), [Facebook](https://www.facebook.com/Clyde21c)

이건희 : [Github](https://github.com/Geonhee-LEE), [Facebook](https://www.facebook.com/Geonheeee)

김준태 : [Github](https://github.com/OPAYA), [Facebook](https://www.facebook.com/kjt7889)

김예찬 : [Github](https://github.com/suhoy901), [Facebook](https://www.facebook.com/suhoy90)
