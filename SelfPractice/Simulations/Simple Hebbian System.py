import jax
import jax.numpy as jnp

class Hebbian:
    def __init__(self, states, corrupt_state, dimension_N, num_states):
        self.states = states
        self.corrupt_state = corrupt_state
        self.dim = dimension_N
        self.num_states = num_states
        self.key = jax.random.PRNGKey(40)

        self.mod_state_arr = states.reshape(num_states, dimension_N)
        self.W_unmod_arr = 1/dimension_N * (self.mod_state_arr.T @ self.mod_state_arr)
        self.W_arr = self.W_unmod_arr.at[jnp.diag_indices(N)].set(0)
        

    def update(self):
        W_arr = self.W_arr
        dim = self.dim
        num_states = self.num_states
        s_arr = self.corrupt_state
        key_new, self.key = jax.random.split(self.key)
        
        @jax.jit
        def step_fn(state, key):

            h_arr_0 = jnp.dot(W_arr, state)
            h_arr_sign = jnp.sign(h_arr_0)
            h_arr_end = jnp.where(h_arr_0 == 0, state, h_arr_sign)

            probability = 0.5
            mask = jax.random.bernoulli(key, p=probability, shape=h_arr_end.shape)
            new_state = jnp.where(mask, h_arr_end, state)

            return new_state, key

        
