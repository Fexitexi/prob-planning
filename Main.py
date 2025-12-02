import gymnasium as gym
import time
import random

from env.ASPTransformer import ASPTransformer
from env.GardenerQAgent import GardenerQAgent
from env.state import ObservationState

gym.envs.registration.register(
    id="GardenerEnv-v0",
    entry_point="env.GardenerEnv:GardenerEnv",
)

if __name__ == "__main__":
    env = gym.make("GardenerEnv-v0")
    gar = env.unwrapped
    numTraining = 0
    numTesting = 10
    horizon = 3
    sample_size = 100
    q_agent = GardenerQAgent()
    asp_transformer = ASPTransformer(q_agent)
    if numTraining == 0:
        q_agent.stopLearning()
        q_agent.load_weights("weights.pkl")


    while numTraining > 0 or numTesting > 0:
        if numTraining > 0:
            numTraining -= 1
            if numTraining == 0:
                q_agent.stopLearning()
                q_agent.save_weights("weights.pkl")
        else:
            numTesting -= 1
        obs, info = env.reset()
        done = False

        # RL agent test
        state = ObservationState.from_obs(obs)
        static = asp_transformer.build_static(state, horizon)

        q_agent.registerInitialState(state)

        print("Starting episode {}\n".format(numTraining + numTesting))

        interceptions = 0
        frog_kills = 0

        while not done:
            #sampling
            samples = gar.sample(horizon, sample_size)
            worlds = asp_transformer.build_worlds(samples)

            #clingo
            dynamic = asp_transformer.build_dynamic(state)
            action = asp_transformer.call_clingo(static, dynamic, worlds)


            test_action = q_agent.getBestActions(state)
            #action = random.choice(test_action)
            if action not in test_action:
                interceptions += 1
                clean = [x.item() for x in test_action]
                msg = f"ASP overruled Q-learning action: {action} vs {clean}"
                print(f"\033[31m{msg}\033[0m")
                for i in range(5):
                    try:
                        value = q_agent.getQValue(state, i)
                        print(f"Action {i}: {value}")
                    except:
                        pass

            for frog in state.frogs:
                if state.agent[0] == frog[0] and state.agent[1] == frog[1]:
                    frog_kills += 1

            obs, reward, terminated, truncated, info = env.step(action)
            state = ObservationState.from_obs(obs)
            q_agent.observeTransition(action, state, reward)
            env.render()
            if numTraining == 0:
                time.sleep(0.5)
            done = terminated or truncated

    print(f"Interceptions: {interceptions}, Frog kills: {frog_kills}")
    env.close()