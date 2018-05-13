from retro_contest.local import make


def main():
    env = make(game='SonicAndKnuckles3-Genesis', state='AngelIslandZone.Act1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
