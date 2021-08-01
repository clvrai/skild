import matplotlib.pyplot as plt

from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.data.maze.src.maze_agents import MazeAgent
from skild.rl.agents.skild_agent import SkiLDAgent



class MazeSkiLDAgent(SkiLDAgent, MazeAgent):
    def __init__(self, *args, **kwargs):
        SkiLDAgent.__init__(self, *args, **kwargs)
        self.vis_replay_buffer = UniformReplayBuffer({'capacity': 1e7})    # purely for logging purposes

    def visualize(self, logger, rollout_storage, step):
        MazeAgent._vis_replay_buffer(self, logger, step)
        MazeSkiLDAgent._vis_replay_buffer(self, logger, step)

    def _vis_replay_buffer(self, logger, step):
        # visualize discriminator rewards
        if 'discr_reward' in self.vis_replay_buffer:
            # get data
            size = self.vis_replay_buffer.size
            start = max(0, size-5000)
            states = self.vis_replay_buffer.get().observation[start:size, :2]
            rewards = self.vis_replay_buffer.get().discr_reward[start:size]

            fig = plt.figure()
            plt.scatter(states[:, 0], states[:, 1], s=5, c=rewards, cmap='RdYlGn')
            plt.axis("equal")
            logger.log_plot(fig, "discr_reward_vis", step)
            plt.close(fig)
