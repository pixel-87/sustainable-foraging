from sustainable_foraging.foraging import AECForagingEnv

env = AECForagingEnv(
    players=2,
    field_size=(8, 8),
    max_num_food=2,
    sight=8,
    max_episode_steps=500,
)

env.reset()

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    
    if terminated or truncated:
        action = None
    else:
        action = env.action_space(agent).sample()
        
    env.step(action)
    env.render()

env.close()
