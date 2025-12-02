from gym.envs.registration import register

register(
    id='Overcooked-v1',
    entry_point='envs.wrappers:OvercookedEnvWrapper',
)

register(
    id='Overcooked-v2',
    entry_point='envs.wrappers:OvercookedEnvWrapperMLP',
)
