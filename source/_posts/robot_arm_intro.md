---
title: 각잡고 로봇팔 프로젝트 소개
date: 2018-11-20 17:22:40
tags: ["프로젝트", "각잡고로봇팔"]
categories: 프로젝트
author: 유지원
subtitle: 각잡고 로봇팔 프로젝트 소개
---


## 프로젝트 소개
본 게시글은 **Reinforcement Learning Korea** 커뮤니티의 2회 프로젝트인 **각잡고 로봇팔** 을 소개하는 글입니다

<br>

## 프로젝트 목표

강화학습을 로봇 컨트롤에 적용함

강화학습의 시작은 게임의 승리 혹은 discrete한 상황의 goal 달성을 목표로 한 task가 주를 이루었습니다. 하지만 절대적인 승패가 존재하지 않는 일반적인 상황에서는 판단하기가 힘듭니다. 이를 극복하기 위해, 정책 자체를 근사화 하는 PG(Policy Gradient)가 고안되었습니다. 현재는 이 기법을 기본으로한 연속적인 동작 제어에 관한 연구가 활발히 진행되고 있습니다. 특히, 우리는 Open AI와 BAIR의 놀라운 연구성과를 토대로 로봇팔 제어에 강화학습을 적용하고자 합니다.

현재는, 하기 2개의 논문의 알고리즘을 변형시켜 Pytorch로 구현 할 예정입니다.

- [Data-Efficient HRL(Data-Efficient Hierarchical Reinforcement Learning)](https://arxiv.org/abs/1805.08296)
- [Deepmimic](https://arxiv.org/abs/1804.02717)

또한, 구현을 위해 하기 논문을 리뷰하였습니다.
 
 - [HER(Handsight Experiece Replay)](https://arxiv.org/abs/1707.01495)

<br>

## 프로젝트 설립 취지

강화학습을 연구하는데 있어, 단순히 강화학습 자체를 연구하기보다 특정 산업 혹은 실물에 적용하는 노력은 학문적인 탐개와 별개로 지속적으로 진행되어야 한다고 생각합니다. 그 가운데, 로봇팔을 제어하는 것을 목표로 삼은 이유는 사람에게 가장 도움이 될 기술이라고 생각하였기 때문입니다. 사람의 팔을 대신할 로봇 혹은 자신의 생각을 말로 표현하는데 어려움이 있는 사람들을 위해 로봇팔의 자연스러움 움직임을 제어하고자 합니다.

<br>

## 프로젝트 연구 일정
**2018.10.27 ~ 2019.1.20 진행**

- (첫째달)첫 보름은 환경과 Task 선정을 합니다.
- (첫째달)동시에, 각자 1개의 논문(강화학습 혹은 로봇팔 제어관련)을 리뷰합니다. -여기까지 왔습니다!
- (첫째달)1주일에 1번씩 행아웃을 통해 함께 논문을 리뷰합니다.
- (둘째달)리뷰한 논문 중 직접 구현할 2개의 논문을 추립니다. 
- (둘째달)2명당 하나의 논문을 담당하여 함께 구현합니다 - 강화학습과 제어에 대한 이해가 더 필요하다면 하나의 논문을 다같이 구현합니다.
- (셋째달)구현을 완료하고, 튜토리얼을 작성합니다.
  - 코드 구현은 github을 통해 협업할 것입니다. 
  - 자료 정리는 ppt 혹은 markdown 형태로 공유할 예정입니다.

<br>

## 프로젝트 현황

- 환경 선정 및 셋업 : [Mujuco](http://www.mujoco.org/)
- 리뷰할 논문 선정
- 현재 두 가지의 논문으로 의견을 모아 논문을 리뷰중입니다.

<br>

## 프로젝트 team

**김재윤** [github](https://github.com/jangikim2) [facebook](https://www.facebook.com/jangikim)
**류연훈** [github](https://github.com/yhryu0409) [facebook](https://www.facebook.com/yeonhun.ryu)
**류(유)지원** [github](https://github.com/AshleyRyu) [facebook](https://www.facebook.com/profile.php?id=100001622442143)
**전준형** [github](https://github.com/junhyeongjeon) [facebook](https://www.facebook.com/Jsobu)
**정의진** [github](https://github.com/jinPrelude) [facebook](https://www.facebook.com/profile.php?id=100011176712221&fref=gs&dti=1890180054554559&hc_location=group_dialog)
