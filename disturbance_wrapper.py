import gymnasium as gym
from pynput import keyboard


class DisturbanceWrapper(gym.Wrapper):
    def __init__(self, env, strength=0.5, impulse=False):
        super().__init__(env)
        self.strength = strength
        self.impulse = impulse  # if True, nudge fires once per keypress; if False, holds while key is down
        self.nudge = 0.0

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def _on_press(self, key):
        if key == keyboard.Key.left:
            self.nudge = -self.strength
        elif key == keyboard.Key.right:
            self.nudge = self.strength

    def _on_release(self, key):
        if key in (keyboard.Key.left, keyboard.Key.right):
            self.nudge = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.nudge != 0.0:
            self._apply_nudge()
            if self.impulse:
                self.nudge = 0.0

        return obs, reward, terminated, truncated, info

    def _apply_nudge(self):
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, 'state'):
            # CartPole: state is [x, x_dot, theta, theta_dot]
            unwrapped.state[1] += self.nudge
        elif hasattr(unwrapped, 'lander'):
            # LunarLander: apply impulse to the Box2D lander body
            lander = unwrapped.lander
            vx, vy = lander.linearVelocity
            lander.linearVelocity = (vx + self.nudge, vy)

    def close(self):
        self._listener.stop()
        super().close()
