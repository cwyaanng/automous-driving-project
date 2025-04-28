import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import *
import cv2
import traceback

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town05", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    args = parser.parse_args()
    
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def make_video_from_frames(folder_path, output_file='output.mp4', fps=20, prefix="third_frame"):

    images = sorted([img for img in os.listdir(folder_path) if img.startswith(prefix) and img.endswith(".png")])

    if not images:
        print(f"[WARNING] No {prefix} images found in {folder_path}")
        return

    # First valid image
    for img in images:
        first_image_path = os.path.join(folder_path, img)
        first_image = cv2.imread(first_image_path)
        if first_image is not None:
            break
    else:
        print("[ERROR] All images failed to load")
        return

    height, width, _ = first_image.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in images:
        img_path = os.path.join(folder_path, img)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARNING] Skipping unreadable image: {img}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    print(f"[DEBUG] 🎬 Video saved to {output_file}")


def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    print("[DEBUG] Starting runner()")

    args = parse_args()
    exp_name = args.exp_name
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init

    print(f"[DEBUG] Parsed args: {args}")
    try:
        if exp_name == 'ppo':
            run_name = "PPO"
        else:
            """
            
            Here the functionality can be extended to different algorithms.

            """ 
            sys.exit() 
    except Exception as e:
        print(e.message)
        sys.exit()
    
    if train == True:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    action_std_decay_rate = 0.05
    min_action_std = 0.05   
    action_std_decay_freq = 5e5
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================

    try:
        print("[DEBUG] Connecting to CARLA server...")
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except Exception as e:
        print(f"[ERROR] Connection to CARLA failed: {e}")
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError
    if train:
        env = CarlaEnvironment(client, world, town)
        env.save_image = True
        env.image_save_dir = os.path.join("video_frames", town, f"train_episode_{episode:04d}")
        print(f"[DEBUG] save_image: {env.save_image}")
        print(f"[DEBUG] Initial image_save_dir: {env.image_save_dir}")
        os.makedirs(env.image_save_dir, exist_ok=True)
    else:
        env = CarlaEnvironment(client, world, town, checkpoint_frequency=None)
        env.save_image = True
        env.image_save_dir = os.path.join("video_frames", town, f"test_episode_{episode:04d}")
        print(f"[DEBUG] save_image: {env.save_image}")
        print(f"[DEBUG] Initial image_save_dir: {env.image_save_dir}")
        os.makedirs(env.image_save_dir, exist_ok=True)

    encode = EncodeState(LATENT_DIM)


    #========================================================================
    #                           ALGORITHM
    #========================================================================
    try:
        time.sleep(0.5)
        
        if checkpoint_load:
            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1
            chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']
                action_std_init = data['action_std_init']
            agent = PPOAgent(town, action_std_init)
            agent.load()
        else:
            if train == False:
                agent = PPOAgent(town, action_std_init)
                agent.load()
                for params in agent.old_policy.actor.parameters():
                    params.requires_grad = False
            else:
                agent = PPOAgent(town, action_std_init)
        if train:
            #Training
            while timestep < total_timesteps:
               
                try:
                    # 환경 리셋 
                    observation = env.reset()
                    observation = encode.process(observation)
                except Exception as e:
                    print(f"[ERROR] env.reset() failed: {e}")
                    traceback.print_exc()
                    break

                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):
                
                    # 정책에 따라 행동 선택 
                    action = agent.get_action(observation, train=True)

                    # agent 가 선택한 action을 환경에 적용 
                    # 다음 상태, reward, done, info 리턴 
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    # 다음 상태도 인코딩 
                    observation = encode.process(observation)
                    # 해당 상태에서 얻은 경험을 agent memory에 저장 
                    agent.memory.rewards.append(reward)
                    agent.memory.dones.append(done)
                    
                    timestep +=1
                    current_ep_reward += reward
                    
                    if timestep % action_std_decay_freq == 0:
                        action_std_init =  agent.decay_action_std(action_std_decay_rate, min_action_std)
                    
                    # 전체 학습 타임스탬프의 마지막 단계 => 체크포인트 저장 
                    if timestep == total_timesteps -1:
                        agent.chkpt_save()

                    # break; if the episode is over
                    if done:
                        episode += 1  
                        os.makedirs(env.image_save_dir, exist_ok=True)
                        # make_video_from_frames(
                        #     folder_path=os.path.join("video_frames", town, f"train_episode_{episode:04d}"),
                        #     output_file=os.path.join("video_frames", town, f"train_episode_{episode:04d}.mp4"),
                        #     prefix="third_frame"
                        # )         
                        
                        env.frame_count = 0
                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                
                # 중앙에서 벗어난 정도 
                deviation_from_center += info[1]
                
                # 누적 거리 합하기 
                distance_covered += info[0]
                
                # 에피소드에서 받은 총 보상을 기록 
                scores.append(current_ep_reward)
                
                if checkpoint_load:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)


                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                # 10 에피소드마다 학습 및 체크포인트 저장 
                if episode % 10 == 0:
                    # PPO 에이전트 학습 
                    agent.learn()
                    # 모델 가중치 저장 
                    agent.chkpt_save()
                    
                    # 가장 최근의 체크포인트 파일 가져오기 
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    if chkt_file_nums != 0:
                        chkt_file_nums -=1
                    # 체크포인트 메타데이터 파일 경로 구성 및 체크포인트 정보 저장 
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {
                        'cumulative_score': cumulative_score, 
                        'episode': episode, 
                        'timestep': timestep, 
                        'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                    
                
                if episode % 5 == 0:

                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center/5, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center/5, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered/5, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered/5, timestep)

                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                if episode % 100 == 0:
                    
                    agent.save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                        
            print("Terminating the run.")
            sys.exit()
        else:
            #Testing
            print("[DEBUG] Starting testing loop")
            while timestep < args.test_timesteps:
                print(f"[DEBUG] Resetting environment at timestep {timestep}")
                
                try:
                    observation = env.reset()
                    observation = encode.process(observation)
                    print("[DEBUG] Environment reset successful")
                except Exception as e:
                    print(f"[ERROR] env.reset() failed111: {e}")
                    traceback.print_exc()
                    break
                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(args.episode_length):
                    # select action with policy
                    print(f"[DEBUG] Step {t} in episode {episode}, timestep {timestep}")
                    try : 
                        action = agent.get_action(observation, train=False)
                        print(f"[DEBUG] Action taken: {action}")
                        observation, reward, done, info = env.step(action)
                        if observation is None:
                            print("[WARNING] Observation is None. Breaking...")
                            break
                    except Exception as e:
                        print(f"[ERROR] env.step() failed: {e}")
                        traceback.print_exc()
                        break
                    observation = encode.process(observation)
                    
                    timestep +=1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        print(f"[DEBUG] Episode {episode} ended11. Setting up next image_save_dir.")
                        episode += 1

                        try:
                            print(f"[DEBUG] Calling make_video_from_frames()")
                            env.image_save_dir = os.path.join("video_frames", town, f"test_episode_{episode:04d}")
                            os.makedirs(env.image_save_dir, exist_ok=True)
                            # make_video_from_frames(
                            #     folder_path=os.path.join("video_frames", town, f"test_episode_{episode:04d}"),
                            #     output_file=os.path.join("video_frames", town, f"test_episode_{episode:04d}.mp4"),
                            #     prefix="third_frame"
                            # )
                            print("[DEBUG] Video created successfully.")
                        except Exception as e:
                            print(f"[ERROR] make_video_from_frames failed: {e}")
                            traceback.print_exc()

                        print(f"[DEBUG] New image_save_dir: {env.image_save_dir}")
                        print(f"[DEBUG] Making video from: video_frames/{town}/train_episode_{episode:04d}")
                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break


                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                
                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run.")
            sys.exit()
        os.makedirs(env.image_save_dir, exist_ok=True)
        env.frame_count = 0  # 매 에피소드마다 프레임 번호 초기화
    except Exception as e:
        print("[ERROR] Uncaught Exception in runner():")
        traceback.print_exc()
        raise  # <- 이게 중요! 메인으로 전달됨

    finally:
        print("[DEBUG] runner() 함수 종료됨")



if __name__ == "__main__":
    try:
        print("[DEBUG] Starting runner()")
        runner()
    except KeyboardInterrupt:
        print("[INFO] Caught KeyboardInterrupt, exiting cleanly.")
        sys.exit()
    except Exception as e:
        print("[ERROR] Uncaught Exception in __main__:")
        print(traceback.format_exc())  # 전체 스택 트레이스 출력
        raise 
    finally:
        print('\nExit')

