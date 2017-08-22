
# coding: utf-8

# <div align="right"><a href="http://mybinder.org/repo/benureau/recode/benureau2015_gb/benureau2015_gb.ipynb">run online</a> | <a href="http://fabien.benureau.com/recode/benureau2015_gb/benureau2015_gb.html">html</a> | <a href="https://github.com/benureau/recode/tree/master/benureau2015_gb">github</a> | <a href="https://dx.doi.org/10.6084/m9.figshare.3081352">10.6084/m9.figshare.3081352</a></div>

# # Recode: Goal Babbling

# This notebook proposes a general introduction to *goal babbling*, and how it differs from *motor babbling*, in the context of robotics. Goal babbling is a way for robots to discover their body and environment on their own. While representations of those could be pre-programmed, there are many reasons not to do so: environments change, robotic bodies are becoming more complex, and flexible limbs, for instance, are difficult and expensive to simulate. By allowing robots to discover the world by themselves, we use the world itself—the best physic engine we know—for robots to conduct their own experiments, and observe and learn the consequence of their actions, much like infants do on their way to becoming adults.
# 
# This notebook requires no previous knowledge beyond some elementary trigonometry and a basic grasp of the Python language. The spirit behind this notebook is to show *all the code* of the algorithms in a simple manner, without relying on any library beyond [numpy](http://www.numpy.org/) (and even, just a very little of it). Only the plotting routines, using the [bokeh](http://bokeh.pydata.org/) library, have been abstracted away in the [graphs.py](https://github.com/benureau/recode/blob/master/benureau2015_gb/graphs.py) file.
# 
# The algorithms and results exposed here were originally presented in the chapter 0 and chapter 3 of [my Ph.D. thesis](http://fabien.benureau.com/docs/phd_benureau.pdf). They were implemented using the [explorers](https://github.com/benureau/explorers) library then. Here, as explained above, we don't rely on the library, and the code has been exposed in its simplest form. It is entirely available under the [Open Science License](http://fabien.benureau.com/openscience.html). A citable version of this notebook is available at [figshare](https://dx.doi.org/10.6084/m9.figshare.3081352). You can contact me for questions or remarks at `fabien.benureau@gmail.com`.
# 
# ## A Bit of Context
# 
# The core idea behind goal babbling is to explore an unknown environment by deciding which goals to try to pursue in it, rather than deciding which action to try. For robots, it means directing the exploration by choosing which effects to produce in the environments, rather than choosing directly which motor commands to execute. Of course, once a goal has been chosen, the robot must find a motor command to reach it. But the decision on what to explore has been made on which effect to try to produce, not in the motor space, and we will see that it makes an important difference.
# 
# The idea of goal babbling was proposed by [Oudeyer and Kaplan (2007)](#References) (p. 8). Computation implementations were then proposed by [Baranes and Oudeyer (2010)](#References); [Rolf et al. (2011)](#References) and [Jamone et al. (2011)](#References). Formal frameworks were described by [Baranes and Oudeyer (2013)](#References) and [Moulin-Frier and Oudeyer (2013)](#References); the work we present here can be understood as an implementation of SAGG-Random. 

# In[ ]:


import random
import numpy as np
import graphs

SEED = 0
random.seed(SEED)                                                                              # reproducible results.


# ## Robotic Arms

# We consider four different robotic arms, with 2, 7, 20, and 100 joints respectively. All arms measure one meter in length, and all the segments between the joints of the same arm are of equal length. The 7-joint arm, therefore, has segments of 1/7th of a meter, as shown in the figure below. Each joint can move between -150° and +150°. The arm receives a *motor commands* composed of 2, 7, 20 or 100 angle values, and returns an *effect*, the position of the tip of the arm after positioning each joint at the required angle, as a cartesian coordinate $x, y$.

# In[ ]:


class RoboticArm(object):
    """A simple robotic arm.

    Receives an array of angles as a motor commands.
    Return the position of the tip of the arm as a 2-tuple.
    """

    def __init__(self, dim=7, limit=150):
        """`dim` is the number of joints, which are able to move in (-limit, +limit)"""
        self.dim, self.limit = dim, limit
        self.posture = [(0.0, 0.0)]*dim                # hold the last executed posture (x, y position of all joints).

    def execute(self, angles):
        """Return the position of the end effector. Accepts values in degrees."""
        u, v, sum_a, length = 0, 0, 0, 1.0/len(angles)
        self.posture = [(u, v)]
        for a in angles:
            sum_a += np.radians(a)
            u, v = u + length*np.sin(sum_a), v + length*np.cos(sum_a)           # at zero pose, the tip is at x=0,y=1.
            self.posture.append((u, v))
        return u, v


# Let's create our four arms.

# In[ ]:


arm2   = RoboticArm(dim=2,   limit=150)   # you can create new arms here, or modify the parameters of an existing one.
arm7   = RoboticArm(dim=7,   limit=150)   # for instance, the `limit` parameter can have important consequences.
arm20  = RoboticArm(dim=20,  limit=150)
arm100 = RoboticArm(dim=100, limit=150)


# This is what the 7-joint arm looks like in a few handpicked postures. The grey circle represents the approximate limits of the reachable space, i.e. the space the tip of the arm can reach. [[1]](#Footnotes)

# In[ ]:


fig = graphs.postures(arm7, [[  0,   0,   0,   0,   0,   0,   0],
                             [ 80, -70,  60, -50,  40, -30,  20],
                             [150, -10, -10, -10, -10, -10, -10]], disk=True)


# We desire to *explore* the possibilities offered by the different arms, that is, understand where we can place the tip of the end-effector.
# 
# And we want to do that from an agnostic standpoint: the arm is a black box, receiving inputs (the angles of the joints) and producing outputs (the position of the tip of the arm). We can't assume any knowledge about the arm, nor deduce anything about the order of the inputs (which can be shuffled). In particular, zero can't be used as a special value. This constraint of ignorance allows to reuse the strategies developed on the arm for any other black box (in theory).
# 
# We add a second constraint: time. We can only execute motor commands a limited number of times on the arm. Here, we limit ourselves to 5000 executions. 5000 executions may seem a lot, but it is not compared to the size of the motor space: if we wanted sample all the motor commands whose angle values are multiples of 10° (-150°, -140°, ..., 0°, ..., 150° for each joint), we would need $31^7 = 27,512,614,111$ executions, more than 5 million times the budget we have.
# 
# So, given these two constraints, what are possible strategies to explore the arm?

# In[ ]:


N = 5000                                                                                       # the execution budget.


# ## Motor Babbling

# The simplest strategy is to try random motor commands. We call this strategy *random motor babbling* (RMB). 'Babbling', here, means that we execute those commands because we don't know—and we want to know—what they will produce.

# In[ ]:


def motor_babbling(arm):
    """Return a random (and legal) motor command."""
    return [random.uniform(-arm.limit, arm.limit) for _ in range(arm.dim)]    


# In[ ]:


def explore_rmb(arm, n):
    """Explore the arm using random motor babbling (RMB) during n steps."""
    history = []

    for t in range(n):
        m_command = motor_babbling(arm)                        # choosing a motor command; does not depend on history.
        effect = arm.execute(m_command)                        # executing the motor command and observing its effect.
        history.append((m_command, effect))                                                        # updating history.

    return history


# This is a simple strategy. By using it for 5000 motor commands, we obtain the following distribution of effects.

# In[ ]:


figures = []

for arm in [arm2, arm7, arm20, arm100]:
    history = explore_rmb(arm, N)                                  # running the RMB exploration strategy for N steps.

    figures.append(graphs.effects(history, show=False,
                                  title='motor babbling, {}-joint arm'.format(arm.dim)))

graphs.show([[figures[0], figures[1]],                                                           # displaying figures.
             [figures[2], figures[3]]])


# Each blue dot is the position reached by the tip of the arm during the execution of a motor command.
# 
# The random motor babbling strategy does rather well on the 2-joint arm. But as the number of joints increases, the distribution of the effects covers less and less of the approximate reachable area represented by the grey disk. Even after numerous samples of the 7, 20 and 100-joint arms, we don't have a good empirical estimation of the reachable space.

# ## Goal Babbling

# Motor babbling explores the possibilities offered by the arm by exploring the *motor space*: the choice of which motor command to execute is done by choosing a point in the motor space.
# 
# Another strategy is to explore not the motor space but the sensory space, i.e., the space of effects. *Goal babbling* does exactly this: choose a point in the effect space, consider it as a goal to be reached, and try to find a motor command to reach that goal. We still produce a motor command to execute, but the *choice* of that motor command is done in the sensory space.
# 
# Implementing the goal babbling strategy raises two problems:
# 1. How to find a motor command to reach a specific goal?
# 2. How goals are chosen?

# ### Inverse Model

# The answer to the first question is an *inverse model*. An inverse model is a process that transforms a goal, i.e., a point in the sensory space, in a motor command, i.e. a point in the motor space. The better an inverse model is, the less distance there is between the goal and the effect obtained when executing the motor command produced by the model.
# 
# There are many ways to create an inverse model. Some rely on the arm's schematics. Here, we can't do this, because of our agnostic constraint about not relying on specific knowledge of the arm. Another way is to learn the inverse model during the exploration: each execution of a motor command bring a motor command/effect pair; this data can be exploited to improve our ability to turn effects into motor commands. There are quite sophisticated ways to do that. Here, we choose one of the simplest ways.
# 
# When given a goal, our inverse model looks at our history of observations, and select the one that produced the effect closest to the goal. The motor command that produced this effect is retrieved, and we add a small, random perturbation to it. The resulting motor command is the one that is going to be executed.

# In[ ]:


D = 0.05                         # with a range of ±150°, creates a perturbation of ±15° (5% of 300° in each direction).


# In[ ]:


def dist(a, b):
    """Return the Euclidean distance between a and b"""
    return sum((a_i-b_i)**2 for a_i, b_i in zip(a, b))

def nearest_neighbor(goal, history):
    """Return the motor command of the nearest neighbor of the goal"""
    nn_command, nn_dist = None, float('inf')                                          # naive nearest neighbor search.
    for m_command, effect in history:
        if dist(effect, goal) < nn_dist:
            nn_command, nn_dist = m_command, dist(effect, goal)
    return nn_command

def inverse(arm, goal, history):
    """Transform a goal into a motor command"""
    nn_command = nearest_neighbor(goal, history)                              # find the nearest neighbor of the goal.

    new_command = []
    for m_i in nn_command:
        max_i = min( arm.limit, m_i + 2*D*arm.limit)
        min_i = max(-arm.limit, m_i - 2*D*arm.limit)
        new_command.append(random.uniform(min_i, max_i))       # create a random perturbation inside the legal values.

    return new_command


# #### Optional: Fast Nearest-Neighbors

# Now, the nearest neighbor implementation we have is fine, but it is too slow for some of the experiments we will do here. We replace it by a fast implementation from the [learners](https://github.com/benureau/learners) library. If you want to keep the slow but simple implementation, skip the next three code cells.

# In[ ]:


try: # if learners is not present, the change is not made.
    import learners

    _nn_set = learners.NNSet()
    def nearest_neighbor_fast(goal, history):
        """Same as nearest_neighbors, using the `learners` library."""
        global _nn_set
        if len(history) < len(_nn_set): # HACK
            _nn_set = learners.NNSet()
        for m_command, effect in history[len(_nn_set):]:
            _nn_set.add(m_command, y=effect)
        idx = _nn_set.nn_y(goal)[1][0]
        return history[idx][0]

    print('Using the fast implementation.')

except ImportError:
    nearest_neighbor_fast = nearest_neighbor
    print('Using the slow implementation.')


# Let's verify that the two nearest neighbor implementations do the same thing.

# In[ ]:


history_test = []
for i in range(1000):                                                  # comparing the results over 1000 random query.
    m_command = [random.random() for _ in range(7)]
    effect    = [random.random() for _ in range(2)]
    history_test.append((m_command, effect))

    goal = [random.random() for _ in range(2)]
    nn_a = nearest_neighbor(goal, history_test)
    nn_b = nearest_neighbor_fast(goal, history_test)
    assert nn_a == nn_b                                                              # the results should be the same.


# Okay, that checks out. Let's override the slow implementation.

# In[ ]:


nearest_neighbor = nearest_neighbor_fast


# ### Exploration strategy

# Once we have the inverse model, the remaining question is how to choose goals. There are many ways to cleverly select goals, and this is mostly explored in the context of *intrinsic motivations* (see [Oudeyer and Kaplan (2007)](#References) for instance). Here, we choose the simplest option: random choice. For each sample, we choose as a goal a random point in the square $[-1, 1]\times[-1,1]$.

# In[ ]:


def goal_babbling(arm, history):
    """Goal babbling strategy"""
    goal = [random.uniform(-1, 1), random.uniform(-1, 1)]                   # goal as random point in [-1, 1]x[-1, 1].
    return inverse(arm, goal, history)


# One detail is that the inverse model needs a non-empty history, because it works by creating perturbation of existing motor commands. To solve this problem we begin by doing 10 steps of random motor babbling, and then switch to random goal babbling for the remaining 4990 steps of our 5000-step exploration.

# In[ ]:


def explore_rgb(arm, n):
    """Explore the arm using random goal babbling (RGB) during n steps."""
    history = []

    for t in range(n):
        if t < 10:                                                     # random motor babbling for the first 10 steps.
            m_command = motor_babbling(arm)
        else:                                                                             # then random goal babbling.
            m_command = goal_babbling(arm, history)                                # goal babbling depends on history.

        effect = arm.execute(m_command)                        # executing the motor command and observing its effect.
        history.append((m_command, effect))                                                        # updating history.

    return history


# Let's look at how the random goal babbling strategy covers the reachable space for our four arms.

# In[ ]:


figures, histories_gb = [], []

for arm in [arm2, arm7, arm20, arm100]:
    history = explore_rgb(arm, N)
    histories_gb.append(history)                                   # we keep the histories for further analysis below.

    figures.append(graphs.effects(history,
                                  title='goal babbling, {}-joint arm'.format(arm.dim)))

graphs.show([[figures[0], figures[1]],
             [figures[2], figures[3]]])


# Compared with the figures produced by the motor babbling strategy, the exploration of the reachable space is significantly better when using random goal babbling. Now that we have made this observation, we need to address two things:
# 1. Why?
# 2. Why do the 20-joint and 100-joint still don't explore their reachable space fully with goal babbling?

# ## Effect Distribution and Sensorimotor Redundancy

# To understand why goal babbling is more efficient than goal babbling, we need to understand how effects are distributed in the sensorimotor space. The sensorimotor space is the space where the mapping between actions and effects can be expressed.
# 
# In a typical sensorimotor space, given a random action, all possible effects are not equal in probability. Some effects are more likely to be happening than others. This is due to two main reasons: the non-linear relationship between actions and effects, and *sensorimotor redundancy*.
# 
# The non-linear relationship means that small modifications of motor commands won't have a proportional effect over the whole sensorimotor space. In some parts of the sensorimotor space, a small motor command modification will have a significant impact on the effect produced, while in others it will have little or none at all.
# 
# We can demonstrate this with an example. In the code below, we take two motor commands (`m_a` and `m_b`), and look at the effects produced when the same 1000 random perturbations are applied to both. The effects stemming from the first motor command, `m_a`, are spread over a larger area than the effects stemming from the second. 

# In[ ]:


m_a = np.array([20, 20, 20, 20, 20, 20, 20])
m_b = np.array([50, 50, 50, 50, 50, 50, 50])

history_a, history_b = [], []
for _ in range(1000):
    perturbation = [random.uniform(-15, 15) for _ in range(7)]   # this is the same perturbation of our inverse model.
    m_a_p = m_a + perturbation
    m_b_p = m_b + perturbation
    history_a.append((m_a_p, arm7.execute(m_a_p)))
    history_b.append((m_b_p, arm7.execute(m_b_p)))

fig_a = graphs.effects(history_a, alpha=0.25, title='m_a')
graphs.postures(arm7, [m_a], fig=fig_a, show=False)

fig_b = graphs.effects(history_b, alpha=0.25, title='m_b')
graphs.postures(arm7, [m_b], fig=fig_b, show=False)

graphs.show([[fig_a, fig_b]])


# The second and by far the most significant reason for the uneven distribution of effect is *sensorimotor redundancy*.  A robotic arm is redundant if there is more than one arm posture that produce the same effect, i.e., the same position of the tip of the arm [[2]](#Footnotes).
# 
# The redundancy of an effect is the number of different motor commands that produce it, which can be infinite. The redundancy is heterogeneously distributed over the sensory space. For instance, there is only one way to reach the point (0, 1) (the motor commands with all zeros), but the point (0, 0) can be reached in an infinite number of ways on an arm with 3 joints or more (the angle of the first joint does not matter).
# 
# We can actually *show* the differences in redundancy on the 2-joint arm. We increase the number of steps to 50000 to get a better picture. This is where it takes a really long time if you have kept the slow implementation of the nearest neighbors algorithm.

# In[ ]:


history_rmb_50k = explore_rmb(arm2, 50000)                                        # random motor babbling exploration.
history_rgb_50k = explore_rgb(arm2, 50000)                                         # random goal babbling exploration.


# In[ ]:


rmb_fig = graphs.effects(history_rmb_50k, alpha=0.25,
                         title='2-joint arm, random motor babbling, 50000 steps.')
rgb_fig = graphs.effects(history_rgb_50k, alpha=0.25,
                         title='2-joint arm, random goal babbling, 50000 steps.')
graphs.show([[rmb_fig, rgb_fig]])


# In the random motor babbling distribution, there are two areas where the distribution of effect is roughly twice as less dense than the rest.
# 
# In those areas, only one solution exists to reach a position. In the rest of the reachable space, two solutions exist: one solution with a positive angle in the second joint, and one with a negative one. We can visualize them by separating the two types of solutions:

# In[ ]:


# separating effects with postures with a positive and negative second joint.
pos_history = [h for h in history_rmb_50k if h[0][1] > 0]                                 # h[0] is the motor command.
neg_history = [h for h in history_rmb_50k if h[0][1] < 0]                  # h[0][1] is the value of the second joint.

# creating some posture examples of each kind
pos_examples = [[-150+2*i,  i] for i in range(0, 151, 10)]
neg_examples = [[-150+2*i, -i] for i in range(0, 151, 10)]

# displaying the graphs
pos_fig = graphs.effects(pos_history, alpha=0.25,
                         title='postures with postive second joint')
graphs.postures(arm2, pos_examples, fig=pos_fig, show=False)

neg_fig = graphs.effects(neg_history, alpha=0.25,
                         title='postures with negative second joint')
graphs.postures(arm2, neg_examples, fig=neg_fig, show=False)

graphs.show([[pos_fig, neg_fig]])


# Random motor babbling is as an empirical estimator of the density of the effect distribution. And, because the sensorimotor redundancy has a much larger impact on the effect distribution than other factors, we can consider random motor babbling an approximate estimator of the sensorimotor redundancy.
# 
# Therefore, random motor babbling will produce effects preferentially in areas of high redundancy. When the differences in redundancy are high enough, the probability of random motor babbling producing effects in areas of low redundancy becomes too low for any practical purposes. This is what happens with the 20- and 100-joint arm: the center region has a redundancy orders of magnitude higher than the rest: random motor babbling only produces effect clustered in the center.
# 
# In contrast, the goal babbling strategy on the 2-joint arm explores different levels of redundancy equally [[3]](#Footnotes). This is because the exploration is directed uniformly in the sensory space, actively countering the redundancy. 

# ## Local Minima

# The goal babbling strategy does not explore the entire reachable space in the case of the 20 and 100-joint arm. To understand why, let's look at the posture producing the effect with the lowest x coordinate. 

# In[ ]:


def leftest_posture(history):
    """Return the posture producing the effect with minimum x"""
    left_m_command, left_effect = history[0]
    for m_command, effect in history:
        if effect[0] < left_effect[0]:
            left_m_command, left_effect = m_command, effect
    return left_m_command

fig = graphs.effects(histories_gb[2], alpha=0.25,
                     y_range=[-0.1, 0.2], x_range=[-1, 0.2],
                     width=900, height=300, show=False)
fig = graphs.postures(arm20, [leftest_posture(histories_gb[2])],
                      radius_factor=0.75, fig=fig)


# There is a loop in the arm starting at the 11th joint. Our inverse model is unable to disentangle that loop through successive perturbation: it tightens it instead. In this posture, the loop is almost completely tightened, as a matter of fact. This limits the span of the arm, and produces a distribution of effects that covers less than the total reachable area. This is a classic example of a local minimum. For another illustrated example of this phenomenon, see Figure 7 (p. 39) of [my Ph.D. thesis](http://fabien.benureau.com/docs/phd_benureau.pdf).
# 
# This limitation is created by the interactions of a simple inverse model and a simplistic robotic model. A temptation here is to improve the inverse model, possibly by choosing a more complex learning algorithm. In many cases, however, improving the robot is much cheaper computationally. For instance, real robots usually can't traverse themselves. Adding collision detection to our simulated robot would prevent the apparitions of such loops.

# ## Choosing Goals

# So far, we have seen the difference between motor babbling and goal babbling, and understood that the difference in efficiency between the two is related to how the redundancy is distributed in the motor space. And we have seen that, unsurprisingly, goal babbling is not immune to local minima.
# 
# But there's one detail we have overlooked so far. In the goal babbling strategy, goals are chosen in the $[-1, 1]\times[-1, 1]$ square. This is highly suspicious, as this square is almost the axis-aligned bounding box of the reachable space. This is clearly a breach of the agnostic constraint that we imposed at the beginning.
# 
# So what would happen if goals were chosen in an overestimated area, or and underestimated area? Let's find out.
# 
# First we define a goal babbling strategy who chooses goals in a specific area:

# In[ ]:


def goal_babbling_area(arm, history, area):
    """Goal babbling strategy, with a specific distribution of goals"""
    goal = [random.uniform(area[0][0], area[0][1]),
            random.uniform(area[1][0], area[1][1])]
    return inverse(arm, goal, history), goal

def explore_rgb_area(arm, n, area=([-1, 1], [-1 ,1])):
    """Explore the arm using random goal babbling over a specific area."""
    history, goals = [], []

    for t in range(n):
        if t < 10:                                                     # random motor babbling for the first 10 steps.
            m_command = motor_babbling(arm)
        else:                                                                             # then random goal babbling.
            m_command, goal = goal_babbling_area(arm, history, area)
            goals.append(goal)

        effect = arm.execute(m_command)                        # executing the motor command and observing its effect.
        history.append((m_command, effect))                                                        # updating history.

    return history, goals


# We run the strategy for four different cases: for the $[-1, 1]\times[-1, 1]$ square—as before, for the $[-2, 2]\times[-2, 2]$—four times the area of the unit square, for the $[-10, 10]\times[-10, 10]$ square—100 times the area, and $[-0.5, 0.5]\times[-0.5, 0.5]$—an area a quarter of the size of the unit square. We have two overestimated areas and one underestimated.

# In[ ]:


figures = []

for area in [([-0.5, 0.5], [-0.5, 0.5]),
             ([  -1,   1], [  -1,   1]),
             ([  -2,   2], [  -2,   2]),
             ([ -10,  10], [ -10,  10])]:
    random.seed(SEED)                                                       # same motor babbling phase for all cases.
    history, goals = explore_rgb_area(arm20, 5000, area=area)
    fig_e = graphs.effects(history, title='effect distribution')
    fig_g = graphs.goals(goals, title='goal distribution over {} x {}'.format(*area))
    figures.append([fig_g, fig_e])

graphs.show(figures)


# Here we see that the distribution of effects is strongly correlated with the distribution of goals. This is not surprising, but it illustrates one of the strengths of goal babbling: by manipulating the distribution of goals only, we can manipulate how exploration proceeds. This is exploited by the computational approaches of intrinsic motivations; see [Oudeyer and Kaplan (2007)](#References) and [Baldassarre and Mirolli (2013)](#References) for reviews.

# ### Agnostic Goal Distribution

# Since the distribution of goal matters, we need to show that we can devise algorithms that do not rely on spoon-fed goal areas. To do that, we have to compute the area where the goals are randomly drawn during the exploration. We choose an algorithm that draws goals into an area slightly bigger than the current axis-aligned bounding box of the observed effects:

# In[ ]:


def update_goal_area(history, extrema):
    """Update the goal area to be 1.4 times bigger than the current observations"""
    if extrema is None:                    # the first update of the area (based only on motor babbling observations).
        xs = [e[0] for _, e in history]
        ys = [e[1] for _, e in history]
        x_range = min(xs), max(xs)
        y_range = min(ys), max(ys)

    else:                                    # we only need to consider the last effect when doing subsequent updates.
        e = history[-1][1]
        x_range, y_range = extrema
        x_range = min(x_range[0], e[0]), max(x_range[1], e[0])
        y_range = min(y_range[0], e[1]), max(y_range[1], e[1])

    extrema = x_range, y_range
    width  = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    area = [[x_range[0] - 0.2*width , x_range[1] + 0.2*width ],
            [y_range[0] - 0.2*height, y_range[1] + 0.2*height]]
    return area, extrema

def explore_rgb_adaptative(arm, n):
    """Explore the arm using random goal babbling over a specific area."""
    history, goals, extrema = [], [], None      # extrema keeps tracks of the min and max values along each dimension.

    for t in range(n):
        if t < 10:                                                     # random motor babbling for the first 10 steps.
            m_command = motor_babbling(arm)
        else:                                                                             # then random goal babbling.
            area, extrema = update_goal_area(history, extrema)               # building the goal area from experience.
            m_command, goal = goal_babbling_area(arm, history, area)               # goal babbling depends on history.
            goals.append(goal)

        effect = arm.execute(m_command)                        # executing the motor command and observing its effect.
        history.append((m_command, effect))                                                        # updating history.

    return history, goals


# We test the behavior of this algorithm on the 20-joint arm.

# In[ ]:


random.seed(SEED)
history, goals = explore_rgb_adaptative(arm20, 5000)
fig_e = graphs.effects(history, title='effect distribution')
fig_g = graphs.goals(goals, title='adaptative goal distribution')
graphs.show([[fig_g, fig_e]])


# This goal babbling algorithm works fine. It still makes a problematic assumption: that the axis-aligned bounding box of the observation is a good approximation of the reachable space. More sophisticated algorithms, such as the Frontiers algorithm we introduced in [my Ph.D. thesis](http://fabien.benureau.com/docs/phd_benureau.pdf), or the direction-sampling algorithm of [Rolf (2013)](#References) are able to handle more general cases. [Baranes and Oudeyer (2013)](#References) has also shown that intrinsic motivations could cope with an overestimation of the goal space.

# ## Conclusion
# 
# We have shown how goal babbling works on a simple simulation of a 2D arm robot. We have shown that goal babbling, by choosing what to explore in the sensory space, and by relying on a learning algorithm, can explore much better than motor babbling. The next step is to go beyond a random choice of goal, and in particular, to make future goals depend on the history of observations: this is what intrinsic motivations do.

# ## Footnotes

# 1. The (0, -1) point is not reachable by the arm, due to the angle constraints: that would require every joint to be at 180 degrees. The unreachable area inside the unit circle decreases with the number of joints; only in the 2-joint arm the area is significant. Additionally, in the 2-joint arm case, the area centered around the base of the arm is not reachable either, for the same reasons.
# 2. We could add the orientation to the redundancy criterion. In that context, an arm is redundant only if it can produce the same position *and* orientation of the tip with different postures. Including the orientation criterion is important in most practical applications. Under that criterion, the 2-joint arm is not redundant.
# 3. Some areas near the inner and outer edges of the goal babbling distribution do display higher concentration of effects. This is an effect of our inverse model. For a precise explanation, see chapter 0 (page 38 and Figure 6) of [my Ph.D. thesis](http://fabien.benureau.com/docs/phd_benureau.pdf). 

# ## References

# * **Baldassarre, G. and Mirolli, M.** (eds.) (2013). Intrinsically Motivated Learning in Natural and Artificial Systems *(Springer Science + Business Media)*. [doi:10.1007/978-3-642-32375-1](http://doi.org/10.1007/978-3-642-32375-1)
# * **Baranes, A. and Oudeyer, P.-Y.** (2010). Intrinsically motivated goal exploration for active motor learning in robots: A case study. In *2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (IEEE)* [doi:10.1109/iros.2010.5651385](http://doi.org/10.1109/iros.2010.5651385)
# * **Baranes, A. and Oudeyer, P.-Y.** (2013). Active learning of inverse models with intrinsically motivated goal exploration in robots. *Robotics and Autonomous Systems* 61, 49–73. [doi:10.1016/j.robot.2012.05.008](http://doi.org/10.1016/j.robot.2012.05.008)
# * **Jamone, L., Natale, L., Hashimoto, K., Sandini, G., and Takanishi, A.** (2011). Learning task space control through goal directed exploration. In *2011 IEEE International Conference on Robotics and Biomimetics* [doi:10.1109/robio.2011.6181368](http://doi.org/10.1109/robio.2011.6181368)
# * **Moulin-Frier, C. and Oudeyer, P.-Y.** (2013). Exploration strategies in developmental robotics: A unified probabilistic framework. In *2013 IEEE Third Joint International Conference on Development and Learning and Epigenetic Robotics (ICDL-Epirob)*. [doi:10.1109/devlrn.2013.6652535](http://doi.org/10.1109/devlrn.2013.6652535)
# * **Oudeyer, P.-Y. and Kaplan, F.** (2007). What is intrinsic motivation? a typology of computational approaches. *Frontiers in neurorobotics* 1, 6. [doi:10.3389/neuro.12.006.2007](http://doi.org/10.3389/neuro.12.006.2007)
# * **Rolf, M., Steil, J., and Gienger, M.** (2011). Online goal babbling for rapid bootstrapping of inverse models in high dimensions. In *Proc. ICDL 2011.* vol. 2, 1–8. [doi:10.1109/devlrn.2011.6037368](http://doi.org/10.1109/devlrn.2011.6037368)
# * **Rolf, M.** (2013), "Goal babbling with unknown ranges: A direction-sampling approach," in *2013 IEEE Third Joint International Conference on Development and Learning and Epigenetic Robotics (ICDL-Epirob)*. [doi:10.1109/devlrn.2013.6652526](http://10.1109/devlrn.2013.6652526)

#   
