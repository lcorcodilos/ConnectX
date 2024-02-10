from kaggle_environments import evaluate, make, utils


def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])



if __name__ == "__main__":
    env = make("connectx", debug=True)
    env.reset()

    print(evaluate("connectx", [my_agent, "random"], num_episodes=100))
    # env.run([my_agent, "random"])

    # trainer = env.train([None, "random"])
    # observation = trainer.reset()
    # my_action = my_agent(observation, env.configuration)
    # trainer.step(my_action)