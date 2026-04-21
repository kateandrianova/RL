import sys, os
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
from IPython.display import clear_output

env = gym.make("CartPole-v0", render_mode="rgb_array").env
env.reset()
n_actions   = env.action_space.n
state_dim   = env.observation_space.shape[0]

print(f"state vector dim = {state_dim}")
print(f"n_actions = {n_actions}")

# Инициализируем агента
agent = MLPClassifier(
    hidden_layer_sizes=(20, 20),
    activation="tanh",
    max_iter=1,
    warm_start=True,      # обязательно — иначе веса сбрасываются каждый раз
    solver="adam",
    learning_rate_init=1e-3,
)
agent.partial_fit(
    [env.reset()[0]] * n_actions,
    range(n_actions),
    classes=range(n_actions),
)


def generate_session(env, agent, t_max=1000):
    states, actions = [], []
    total_reward = 0.0

    s, _ = env.reset()

    for t in range(t_max):
        # predict_proba возвращает матрицу (1, n_actions) → берём первую строку
        probs = agent.predict_proba([s])[0]

        assert probs.shape == (env.action_space.n,), \
            "Вероятности должны быть вектором длины n_actions"

        a = np.random.choice(env.action_space.n, p=probs)

        new_s, r, terminated, truncated, _ = env.step(a)

        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if terminated or truncated:
            break

    return states, actions, total_reward

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    threshold = np.percentile(rewards_batch, percentile)

    elite_states  = []
    elite_actions = []

    for states, actions, reward in zip(states_batch, actions_batch, rewards_batch):
        if reward > threshold:
            elite_states.extend(states)
            elite_actions.extend(actions)

    return elite_states, elite_actions

def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10]):
    mean_reward = np.mean(rewards_batch)
    threshold   = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])

    clear_output(True)
    print(f"mean reward = {mean_reward:.3f},  threshold = {threshold:.3f}")
    plt.figure(figsize=[8, 4])

    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label="Mean rewards")
    plt.plot(list(zip(*log))[1], label="Reward thresholds")
    plt.legend(); plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(rewards_batch, range=reward_range, bins=20)
    plt.vlines(
        [np.percentile(rewards_batch, percentile)],
        [0], [100],
        label="percentile", color="red",
    )
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.show()


def train_cartpole(n_sessions=100, percentile=70, n_iter=100):
    global agent, env

    log = []

    for i in range(n_iter):
        sessions = [generate_session(env, agent) for _ in range(n_sessions)]
        states_batch, actions_batch, rewards_batch = zip(*sessions)
        states_batch  = list(states_batch)
        actions_batch = list(actions_batch)
        rewards_batch = np.array(rewards_batch)

        elite_states, elite_actions = select_elites(
            states_batch, actions_batch, rewards_batch, percentile
        )

        if len(elite_states) == 0:
            continue

        agent.partial_fit(
            elite_states,
            elite_actions,
            classes=range(n_actions),
        )

        show_progress(
            rewards_batch, log, percentile,
            reward_range=[0, np.max(rewards_batch)]
        )

        if np.mean(rewards_batch) > 190:
            break

    return log


# HOMEWORK PART I 
def run_taxi_experiments():
    def initialize_policy(n_states, n_actions):
        return np.ones((n_states, n_actions)) / n_actions

    def generate_session_taxi(env, policy, t_max=1000):
        states, actions = [], []
        total_reward = 0.0
        s, _ = env.reset()
        for _ in range(t_max):
            a = np.random.choice(env.action_space.n, p=policy[s])
            new_s, r, terminated, truncated, _ = env.step(a)
            states.append(s); actions.append(a); total_reward += r
            s = new_s
            if terminated or truncated:
                break
        return states, actions, total_reward

    def select_elites_taxi(states_batch, actions_batch, rewards_batch, percentile):
        threshold = np.percentile(rewards_batch, percentile)
        elite_states, elite_actions = [], []
        for states, actions, reward in zip(states_batch, actions_batch, rewards_batch):
            if reward > threshold:
                elite_states.extend(states); elite_actions.extend(actions)
        return elite_states, elite_actions

    def update_policy(elite_states, elite_actions, n_states, n_actions, laplace=1.0):
        policy = np.ones((n_states, n_actions)) * laplace
        for s, a in zip(elite_states, elite_actions):
            policy[s, a] += 1.0
        policy /= policy.sum(axis=1, keepdims=True)
        return policy

    taxi_env = gym.make("Taxi-v3")
    n_states  = taxi_env.observation_space.n
    n_actions = taxi_env.action_space.n

    configs = [
        {"n_sessions": 50,  "percentile": 50, "label": "n=50,  pct=50"},
        {"n_sessions": 100, "percentile": 70, "label": "n=100, pct=70"},
        {"n_sessions": 200, "percentile": 80, "label": "n=200, pct=80"},
        {"n_sessions": 500, "percentile": 90, "label": "n=500, pct=90"},
    ]

    plt.figure(figsize=(14, 5))

    for cfg in configs:
        policy = initialize_policy(n_states, n_actions)
        mean_rewards_log = []

        for iteration in range(50):
            sessions = [generate_session_taxi(taxi_env, policy) for _ in range(cfg["n_sessions"])]
            s_batch, a_batch, r_batch = zip(*sessions)
            s_batch = list(s_batch); a_batch = list(a_batch)
            r_batch = np.array(r_batch)

            elite_s, elite_a = select_elites_taxi(s_batch, a_batch, r_batch, cfg["percentile"])

            if len(elite_s) > 0:
                policy = update_policy(elite_s, elite_a, n_states, n_actions)

            mean_rewards_log.append(np.mean(r_batch))

        plt.plot(mean_rewards_log, label=cfg["label"])

    plt.title("Taxi-v3: влияние percentile и n_sessions")
    plt.xlabel("Итерация"); plt.ylabel("Средняя награда")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("taxi_hyperparams.png", dpi=120)
    plt.show()

    print("\n Tuned Taxi (цель: положительная средняя награда)")
    policy = initialize_policy(n_states, n_actions)
    for iteration in range(80):
        sessions = [generate_session_taxi(taxi_env, policy) for _ in range(500)]
        s_batch, a_batch, r_batch = zip(*sessions)
        s_batch = list(s_batch); a_batch = list(a_batch)
        r_batch = np.array(r_batch)
        mean_r = np.mean(r_batch)

        elite_s, elite_a = select_elites_taxi(s_batch, a_batch, r_batch, percentile=90)
        if len(elite_s) > 0:
            policy = update_policy(elite_s, elite_a, n_states, n_actions, laplace=0.5)

        print(f"Iter {iteration:3d}: mean_reward = {mean_r:.2f}")
        if mean_r > 0:
            print("Положительная награда достигнута")
            break

    taxi_env.close()

# HOMEWORK PART II
def make_mountain_car_agent():
    return MLPClassifier(
        hidden_layer_sizes=(100, 100, 50),
        activation="relu",
        solver="adam",
        learning_rate_init=5e-4,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )


def generate_session_mc(env, agent, t_max=10_000):
    states, actions = [], []
    total_reward = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        probs = agent.predict_proba([s])[0]
        a = np.random.choice(env.action_space.n, p=probs)

        new_s, r, terminated, truncated, _ = env.step(a)
        states.append(s); actions.append(a); total_reward += r
        s = new_s
        if terminated or truncated:
            break

    return states, actions, total_reward


def select_elites_mc(states_batch, actions_batch, rewards_batch, percentile=70):
    threshold = np.percentile(rewards_batch, percentile)
    elite_states, elite_actions = [], []
    for states, actions, reward in zip(states_batch, actions_batch, rewards_batch):
        if reward > threshold:
            elite_states.extend(states)
            elite_actions.extend(actions)
    return elite_states, elite_actions


def train_mountain_car(
    n_sessions   = 100,
    percentile   = 70,
    n_iter       = 150,
    n_jobs       = 4,        
    history_len  = 4,          
    target_reward= -150,
):
    """
    Улучшения:
      • joblib для параллельной генерации сессий
      • буфер из последних history_len итераций при выборе элит
    """
    mc_env = gym.make("MountainCar-v0").env
    n_actions_mc = mc_env.action_space.n

    mc_agent = make_mountain_car_agent()

    # Инициализация агента
    init_state, _ = mc_env.reset()
    mc_agent.partial_fit(
        [init_state] * n_actions_mc,
        range(n_actions_mc),
        classes=range(n_actions_mc),
    )

    log = []
    # Буфер истории: список (states, actions, reward) за прошлые итерации
    history_buffer: list = []

    def _run_session(_):
        #Вспомогательная функция для joblib
        _env = gym.make("MountainCar-v0").env
        result = generate_session_mc(_env, mc_agent)
        _env.close()
        return result

    for i in range(n_iter):
        # Параллельная генерация сессий
        sessions = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_session)(None) for _ in range(n_sessions)
        )

        states_batch, actions_batch, rewards_batch = zip(*sessions)
        states_batch  = list(states_batch)
        actions_batch = list(actions_batch)
        rewards_batch = list(rewards_batch)

        # История: добавляем в буфер
        history_buffer.append((states_batch, actions_batch, rewards_batch))
        if len(history_buffer) > history_len:
            history_buffer.pop(0)

        # Объединяем текущие + исторические сессии для выбора элит
        all_states  = []
        all_actions = []
        all_rewards = []
        for s_b, a_b, r_b in history_buffer:
            all_states.extend(s_b)
            all_actions.extend(a_b)
            all_rewards.extend(r_b)

        elite_states, elite_actions = select_elites_mc(
            all_states, all_actions, all_rewards, percentile
        )

        if len(elite_states) == 0:
            print(f"Iter {i:3d}: нет элит — пропускаем")
            continue

        mc_agent.partial_fit(
            elite_states, elite_actions,
            classes=range(n_actions_mc),
        )

        mean_r   = np.mean(rewards_batch)
        thresh_r = np.percentile(rewards_batch, percentile)
        log.append([mean_r, thresh_r])

        print(
            f"Iter {i:3d}: mean={mean_r:.1f}, "
            f"threshold={thresh_r:.1f}, "
            f"max={np.max(rewards_batch):.1f}, "
            f"elites={len(elite_states)}"
        )

        if mean_r >= target_reward:
            print(f"Цель достигнута: mean_reward = {mean_r:.1f} >= {target_reward}")
            break

    mc_env.close()

    # ── График обучения ──
    if log:
        means, thresholds = zip(*log)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(means, label="Mean reward"); plt.plot(thresholds, label="Threshold")
        plt.axhline(target_reward, color="red", linestyle="--", label=f"Target {target_reward}")
        plt.legend(); plt.grid(); plt.title("MountainCar-v0 обучение")

        plt.subplot(1, 2, 2)
        plt.hist(rewards_batch, bins=20)
        plt.vlines([thresh_r], [0], [n_sessions], color="red", label="threshold")
        plt.legend(); plt.grid(); plt.title("Распределение наград (последняя итерация)")
        plt.tight_layout()
        plt.savefig("mountaincar_training.png", dpi=120)
        plt.show()

    return mc_agent, log


def visualize_mountain_car(env, agent):
    xs = np.linspace(env.min_position, env.max_position, 100)
    vs = np.linspace(-env.max_speed, env.max_speed, 100)

    grid      = np.dstack(np.meshgrid(xs, vs[::-1])).transpose(1, 0, 2)
    grid_flat = grid.reshape(len(xs) * len(vs), 2)
    probs     = (
        agent.predict_proba(grid_flat)
              .reshape(len(xs), len(vs), 3)
              .transpose(1, 0, 2)
    )

    f, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        probs,
        extent=(env.min_position, env.max_position, -env.max_speed, env.max_speed),
        aspect="auto",
    )
    ax.set_title("Learned policy: red=left, green=nothing, blue=right")
    ax.set_xlabel("position (x)"); ax.set_ylabel("velocity (v)")

    states, actions, total_r = generate_session_mc(env, agent)
    states = np.array(states)
    ax.plot(states[:, 0], states[:, 1], color="white", linewidth=2, label=f"reward={total_r:.0f}")

    for (x, v), a in zip(states[::3], actions[::3]):
        if a == 0:
            ax.arrow(x, v, -0.05, 0, color="white", head_length=0.01)
        elif a == 2:
            ax.arrow(x, v,  0.05, 0, color="white", head_length=0.01)

    ax.legend(); plt.tight_layout()
    plt.savefig("mountaincar_policy.png", dpi=120)
    plt.show()

if __name__ == "__main__":

    print("=" * 60)
    print("CARTPOLE-v0  (основная часть)")
    print("=" * 60)
    log_cartpole = train_cartpole(n_sessions=100, percentile=70, n_iter=100)

    print("\n" + "=" * 60)
    print("HOMEWORK PART I — Taxi-v3")
    print("=" * 60)
    run_taxi_experiments()

    print("\n" + "=" * 60)
    print("HOMEWORK PART II — MountainCar-v0")
    print("=" * 60)
    mc_agent, log_mc = train_mountain_car(
        n_sessions=100,
        percentile=70,
        n_iter=200,
        n_jobs=4,        # параллельность
        history_len=4,   # буфер истории
        target_reward=-150,
    )

    # Визуализация политики
    with gym.make("MountainCar-v0") as vis_env:
        visualize_mountain_car(vis_env.unwrapped, mc_agent)
