import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
n_samples = 81
key_v, key_w = jax.random.split(key)
v0 = jax.random.uniform(key_v, (n_samples,), minval=-2.0, maxval=2.0)
w0 = jax.random.uniform(key_w, (n_samples,), minval=-1.0, maxval=1.5)
y0_pre = jnp.stack([v0, w0], axis=-1).astype(jnp.float32)
y0 = y0_pre[:80]

print(y0_pre.shape)