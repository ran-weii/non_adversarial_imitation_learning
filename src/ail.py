import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# model imports
from src.nn_models import MLP, DoubleQNetwork
from src.rl_utils import ReplayBuffer

class DAC(nn.Module):
    """ Discriminator actor critic for discrete actions """
    def __init__(
        self, obs_dim, act_dim, hidden_dim, num_hidden, activation, 
        algo="nail", gamma=0.9, beta=0.2, polyak=0.995, norm_obs=False, 
        buffer_size=int(1e6), batch_size=100, d_steps=50, a_steps=50, 
        lr_d=1e-3, lr_c=1e-3, decay=0., grad_clip=None, grad_penalty=1., grad_target=1.
        ):
        """
        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            algo (str, optional): imitation algorithm. choices=[ail, nail]
            gamma (float, optional): discount factor. Default=0.9
            beta (float, optional): softmax temperature. Default=0.2
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            norm_obs (bool, optional): whether to normalize observations. Default=False
            buffer_size (int, optional): replay buffer size. Default=1e6
            batch_size (int, optional): discriminator and critic batch size. Default=100
            d_steps (int, optional): discriminator update steps per training step. Default=50
            a_steps (int, optional): actor critic update steps per training step. Default=50
            lr (float, optional): learning rate. Default=1e-3
            decay (float, optional): weight decay. Default=0.
            grad_clip (float, optional): gradient clipping. Default=None
            grad_penalty (float, optional): discriminator gradient norm penalty. Default=1.
            grad_target (float, optional): discriminator gradient norm target. Default=1.
        """
        super().__init__()
        assert algo in ["ail", "nail"]
        self.algo = algo
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.beta = beta
        self.polyak = polyak
        self.norm_obs = norm_obs
    
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.d_steps = d_steps
        self.a_steps = a_steps
        self.lr_d = lr_d
        self.lr_c = lr_c
        self.decay = decay
        self.grad_clip = grad_clip
        self.grad_penalty = grad_penalty
        self.grad_target = grad_target
        
        # extra dimension for absorbing state flag
        self.discriminator = MLP(
            input_dim=obs_dim + act_dim + 1, 
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
        )
        self.critic = DoubleQNetwork(
            obs_dim + 1, act_dim, hidden_dim, num_hidden, activation
        )
        self.critic_target = deepcopy(self.critic)

        # freeze target parameters
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # reference critic in nail
        self.critic_ref = deepcopy(self.critic) 
        for param in self.critic_ref.parameters():
            param.requires_grad = False
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, weight_decay=decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_c, weight_decay=decay
        )

        self.real_buffer = ReplayBuffer(obs_dim, act_dim, int(1e6))
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        
        self.plot_keys = ["eps_return_avg", "eps_len_avg", "d_loss_avg", "critic_loss_avg"]
    
    def __repr__(self):
        s_critic = self.critic.__repr__()
        s_discriminator = self.discriminator.__repr__()
        s = "{}(algo={}, gamma={}, beta={}, polyak={}, norm_obs={}, "\
            "buffer_size={}, batch_size={}, a_steps={}, d_steps={}, "\
            "lr_d={}, lr_c={}, decay={}, grad_clip={}, grad_penalty={}"\
            "\n    discriminator={}\n    critic={}\n)".format(
            self.__class__.__name__, self.algo, self.gamma, self.beta, self.polyak, self.norm_obs,
            self.replay_buffer.max_size, self.batch_size, self.a_steps, self.d_steps,
            self.lr_d, self.lr_c, self.decay, self.grad_clip, self.grad_penalty,
            s_discriminator, s_critic
        )
        return s

    def fill_real_buffer(self, dataset):
        for i in range(len(dataset)):
            batch = dataset[i]
            obs = np.array(batch["obs"])
            act = np.array(batch["act"]).reshape(-1, 1)
            next_obs = np.array(batch["next_obs"])
            rwd = np.zeros((len(obs), 1))
            done = np.array(batch["done"]).reshape(-1, 1)
            self.real_buffer.push(obs, act, next_obs, rwd, done)

    def update_normalization_stats(self):
        if self.norm_obs:
            mean = torch.from_numpy(self.replay_buffer.moving_mean).to(torch.float32)
            variance = torch.from_numpy(self.replay_buffer.moving_variance).to(torch.float32)

            self.obs_mean.data = mean
            self.obs_variance.data = variance
    
    def normalize_obs(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_variance**0.5
        return obs_norm
    
    def compute_action_dist(self, critic, obs, absorb, beta=1.):
        critic_input = torch.cat([obs, absorb], dim=-1)
        q1, q2 = critic.forward(critic_input)
        q = torch.min(q1, q2)
        pi = torch.softmax(q / beta, dim=-1)
        return pi

    def choose_action(self, obs):
        obs = torch.from_numpy(obs).view(1, -1).to(torch.float32)
        absorb = torch.zeros(len(obs), 1)
        with torch.no_grad():
            pi = self.compute_action_dist(self.critic_ref, obs, absorb, self.beta)
        a = torch.multinomial(pi, 1)[0].item()
        return a
    
    def compute_gradient_penalty(self, real_inputs, fake_inputs):
        # interpolate data
        alpha = torch.rand(len(real_inputs), 1)
        interpolated = alpha * real_inputs + (1 - alpha) * fake_inputs
        interpolated = Variable(interpolated, requires_grad=True)

        prob = torch.sigmoid(self.discriminator(interpolated))
        
        grad = torch_grad(
            outputs=prob, inputs=interpolated, 
            grad_outputs=torch.ones_like(prob),
            create_graph=True, retain_graph=True
        )[0]

        grad_norm = torch.linalg.norm(grad, dim=-1)
        grad_pen = torch.pow(grad_norm - self.grad_target, 2).mean()
        return grad_pen

    def compute_discriminator_loss(self): 
        real_batch = self.real_buffer.sample(self.batch_size, prioritize=False)
        fake_batch = self.replay_buffer.sample(self.batch_size, prioritize=False)
        
        real_obs = real_batch["obs"]
        real_act = real_batch["act"]
        real_act = F.one_hot(real_act.long().squeeze(-1), self.act_dim).to(torch.float32)
        real_absorb = real_batch["absorb"]
        fake_obs = fake_batch["obs"]
        fake_act = fake_batch["act"]
        fake_act = F.one_hot(fake_act.long().squeeze(-1), self.act_dim).to(torch.float32)
        fake_absorb = fake_batch["absorb"]
        
        # normalize obs
        real_obs_norm = self.normalize_obs(real_obs)
        fake_obs_norm = self.normalize_obs(fake_obs)
        
        # mask absorbing state observation with zeros
        real_obs_norm[real_absorb.flatten() == 1] *= 0
        fake_obs_norm[fake_absorb.flatten() == 1] *= 0
        
        real_inputs = torch.cat([real_obs_norm, real_act, real_absorb], dim=-1)
        fake_inputs = torch.cat([fake_obs_norm, fake_act, fake_absorb], dim=-1)
        inputs = torch.cat([real_inputs, fake_inputs], dim=0)

        real_labels = torch.zeros(self.batch_size, 1)
        fake_labels = torch.ones(self.batch_size, 1)
        labels = torch.cat([real_labels, fake_labels], dim=0)

        d_out = torch.sigmoid(self.discriminator(inputs))
        d_loss = F.binary_cross_entropy(d_out, labels)

        gp = self.compute_gradient_penalty(real_inputs, fake_inputs)
        return d_loss, gp
    
    def compute_reward(self, obs, act_oh, absorb):
        # compute reward as the negative probability of being fake
        inputs = torch.cat([obs, act_oh, absorb], dim=-1)
        log_r = self.discriminator(inputs)
        r = -log_r
        
        if self.algo == "nail":
            # add reference policy likelihood to reward
            # use beta = 1. for reference policy instead of actual beta
            pi = self.compute_action_dist(self.critic_ref, obs, absorb, beta=1.)
            log_pi = torch.log(pi + 1e-6)
            log_pi = torch.sum(act_oh * log_pi, dim=-1, keepdim=True)
            r += log_pi
        return r

    def compute_critic_loss(self):
        real_batch = self.real_buffer.sample(self.batch_size)
        fake_batch = self.replay_buffer.sample(self.batch_size)
        
        real_obs = real_batch["obs"]
        real_absorb = real_batch["absorb"]
        real_act = real_batch["act"]
        real_next_obs = real_batch["next_obs"]
        real_next_absorb = real_batch["next_absorb"]
        real_done = real_batch["done"]
        
        fake_obs = fake_batch["obs"]
        fake_absorb = fake_batch["absorb"]
        fake_act = fake_batch["act"]
        fake_next_obs = fake_batch["next_obs"]
        fake_next_absorb = fake_batch["next_absorb"]
        fake_done = fake_batch["done"]

        obs = torch.cat([real_obs, fake_obs], dim=-2)
        absorb = torch.cat([real_absorb, fake_absorb], dim=-2)
        act = torch.cat([real_act, fake_act], dim=-2)
        next_obs = torch.cat([real_next_obs, fake_next_obs], dim=-2)
        next_absorb = torch.cat([real_next_absorb, fake_next_absorb], dim=-2)
        done = torch.cat([real_done, fake_done], dim=-2)
        
        act_oh = F.one_hot(act.long().squeeze(-1), self.act_dim).to(torch.float32)
        
        # normalize observation
        obs_norm = self.normalize_obs(obs)
        next_obs_norm = self.normalize_obs(next_obs)

        # mask absorbing state observation with zeros
        obs_norm[absorb.flatten() == 1] *= 0
        next_obs_norm[next_absorb.flatten() == 1] *= 0
        
        critic_input = torch.cat([obs_norm, absorb], dim=-1)
        critic_next_input = torch.cat([next_obs_norm, next_absorb], dim=-1)
        with torch.no_grad():    
            # compute reward
            r = self.compute_reward(obs_norm, act_oh, absorb)
            
            # compute absorbing reward
            inputs_a = torch.cat([torch.zeros(1, self.obs_dim + self.act_dim), torch.ones(1, 1)], dim=-1)
            r_a = -self.discriminator(inputs_a)

            # compute value target
            q1_next, q2_next = self.critic_target(critic_next_input)
            q_next = torch.min(q1_next, q2_next)
            v_next = torch.logsumexp(q_next / self.beta, dim=-1, keepdim=True) * self.beta
            v_absorb = self.gamma / (1 - self.gamma) * r_a
            q_target = r + (1 - next_absorb) * self.gamma * v_next + next_absorb * v_absorb
        
        q1, q2 = self.critic(critic_input)
        q1 = torch.gather(q1, -1, act.long())
        q2 = torch.gather(q2, -1, act.long())
        q1_loss = torch.pow(q1 - q_target, 2).mean()
        q2_loss = torch.pow(q2 - q_target, 2).mean()
        q_loss = (q1_loss + q2_loss) / 2 
        return q_loss

    def take_gradient_step(self, logger=None):
        self.discriminator.train()
        self.critic.train()
        self.update_normalization_stats()
        
        d_loss_epoch = []
        for i in range(self.d_steps):
            # train discriminator
            d_loss, gp = self.compute_discriminator_loss()
            d_total_loss = d_loss + self.grad_penalty * gp
            d_total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()

            d_loss_epoch.append(d_loss.data.item())
            
            if logger is not None:
                logger.push({"d_loss": d_loss.data.item()})

        critic_loss_epoch = []
        for i in range(self.a_steps):
            # train critic
            critic_loss = self.compute_critic_loss()
            critic_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            self.d_optimizer.zero_grad()
            
            critic_loss_epoch.append(critic_loss.data.item())
            
            # update target networks
            with torch.no_grad():
                for p, p_target in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    p_target.data.mul_(self.polyak)
                    p_target.data.add_((1 - self.polyak) * p.data)

            if logger is not None:
                logger.push({
                    "critic_loss": critic_loss.cpu().data.item(),
                })

        stats = {
            "d_loss": np.mean(d_loss_epoch),
            "critic_loss": np.mean(critic_loss_epoch),
        }
        
        self.discriminator.eval()
        self.critic.eval()
        return stats

    def on_epoch_end(self):
        """ Update reference critic on epoch end """
        with torch.no_grad():
            for p, p_ref in zip(
                self.critic.parameters(), self.critic_ref.parameters()
            ):
                p_ref.data = p
        