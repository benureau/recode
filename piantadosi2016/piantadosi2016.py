
# coding: utf-8

# <div align="right"><a href="https://beta.mybinder.org/v2/gh/benureau/recode/master?filepath=piantadosi2016/piantadosi2016.ipynb">run online</a> | <a href="http://fabien.benureau.com/recode/piantadosi2016/piantadosi2016.html">html</a> | <a href="https://github.com/humm/recode/tree/master/piantadosi2016">github</a> | <a href="https://dx.doi.org/10.6084/m9.figshare.3990486">10.6084/m9.figshare.3990486</a></div>

# # Recode: Extraordinary intelligence and the care of infants

# We recode the model of the article "Extraordinary intelligence and the care of infants" ([10.1073/pnas.1506752113](http://www.pnas.org/cgi/doi/10.1073/pnas.1506752113)) by [Steve Piantadosi](https://colala.bcs.rochester.edu/people/piantadosi/) and [Celeste Kidd](http://www.bcs.rochester.edu/people/ckidd/). The pdf is [available here](http://www.bcs.rochester.edu/people/ckidd/papers/PiantadosiKidd2016Extraord.pdf). Here, we only succinctly describe the model. You should consult the original article for details and for the rationale behind the model's choices.
# 
# The spirit of this notebook is to use simple code that is easy to understand and modify. This notebook requires no specific knowledge beyond a basic grasp of the Python language. We show *all the code* of the model, without relying on any library beyond [numpy](http://www.numpy.org/). Only the plotting routines, using the [bokeh](http://bokeh.pydata.org/) library, have been abstracted away in the [graphs.py](https://github.com/humm/recode/blob/master/piantadosi2016/graphs.py) file.
# 
# A citable version of this notebook is available at [figshare](https://dx.doi.org/10.6084/m9.figshare.3990486). You can contact me for questions or remarks at `fabien.benureau@gmail.com`.

# In[ ]:


import numpy as np
import graphs
np.random.seed(0)                                                                               # repeatable results.


# ## Model Equations

# Piantosi and Kidd's model ties together three main parameters:
# * $R$, the size of the adult brain.
# * $T$, the duration of the gestation period.
# * $I_p$, a quantification of the intelligence of the parents.
# 
# 

# The size of the child's head at age $t$, $g(t, R)$, follows a [Gompertz growth curve](https://en.wikipedia.org/wiki/Gompertz_function), with $b$ and $c$ free parameters, fitted to 37.1 and 0.42 respectively.

# $$g(t, R) = Re^{-be^{-ct}}$$

# In[ ]:


def g(t, R, b=37.1, c=0.42):
    """Size of the head at time t, given an adult size R."""
    return R * np.exp(-b * np.exp(-c * t)) # if you modify this function, you must modify the solve_MT function below.


# Because a large head means a more difficult and dangerous birth, the probability to survive childbirth decreases ([sigmoidally](https://en.wikipedia.org/wiki/Sigmoid_function)) when the head size at birth exceed a fixed parameter $V$, fitted to 5.48 cm. Here $T$ is the duration of the gestation period.

# $$P(\textrm{survive childbirth}\,|\,T,R) = \phi(V - g(T,R))$$

# In[ ]:


def phi(z):
    """Sigmoid function"""
    return 1/(1 + np.exp(-z))

def P_sb(T, R, V=5.48, g=g):
    """Probability to survive birth"""
    return phi(V - g(T, R))


# The probability to survive adulthood is tied to the time after birth to reach maturity $M$ ($M$ solves $g(M + T, R) = 0.99R$), and the intelligence of the parents $I_p$.

# $$P(\textrm{survive to adulthood}\,|\,M,I_p) = e^{-M(\gamma/I_p)}$$

# Here, the parameter $\gamma$ is fitted to 0.4.

# In[ ]:


def P_sc(M, I_p, gamma=0.4):
    """Probability to survive childhood"""
    return np.exp(-max(0, M) * gamma / I_p)


# The article assumes that $I_p$ is equal to $R$ [[1]](#Footnotes) for Figure 1 and 2A. To try other relationships, modify the following function:

# In[ ]:


def I(R):
    """Return the intelligence (of the parents) as a function of R"""
    return R


# ## Figure 1: Child Growth, Birth and Childhood Survival

# We reproduce Figure 1. The continuous line correspond to $R$ = 8.4 cm, and the dashed line to $R$ = 4.2 cm. You can modify the latter if you are running the jupyter notebook version (the slider will not appear in the html one) by using the slider bellow.

# In[ ]:


ts = np.linspace(0, 25, 251)                                     ## i.e., ts = [0.0, 0.1, 0.2, ..., 24.8, 24.9, 25.0] 
fig1_data = graphs.fig1(ts, g, P_sb, P_sc, I, R=8.4/2)


# In[ ]:


def update_fig1(R):
    graphs.update_fig1(fig1_data, g, P_sb, P_sc, R)
graphs.interact(update_fig1, R=graphs.FloatSlider(min=0.1,max=20.0,step=0.1,value=4.2))


# If the slider has no effect, rerun the last two code cells.

# ## Figure 2A: Fitness Landscape

# To reproduce Figure 2A, we need to compute $M$, which solves $g(M + T, R) = 0.99R$, the solution of which does not actually depends on the value of $R$. We employ a simple [dichotomy method](https://en.wikipedia.org/wiki/Bisection_method) regardless, rather than a precomputed answer, to allow for arbitrary modification of the $g$ function.

# In[ ]:


def solve_MT(R, g, b, c):
    """Return M+T, with M+T solving g(M+T, R) == 0.99*R."""
    return -np.log(-np.log(0.99)/b)/c           # closed-form solution. It may not hold if you modify the g function
                                                # above. In this case, comment this line, the code below is general.
    
    low, up = 1e-3, 25
    while up-low > 1e-3:                                                                # simple dichotomy algorithm.
        middle = 0.5*(up + low)
        if g(middle, R, b=b, c=c) < 0.99*R:
            low = middle
        else:
            up = middle
    return 0.5*(up+low)


# In[ ]:


K = 400                                                                              # K resolution of the landscape.
Ts = np.linspace(  0, 30, K+1)                                              # birth age, K+1 points between 0 and 30. 
Rs = np.linspace(0.1, 10, K)                                  # brain size (radius, cm), K points between 0.1 and 10.

def probability_matrix(Ts, Rs, gamma=0.4, V=5.48, b=37.1, c=0.42):
    """Return the matrix of the probabilities to survive until adulthood."""
    D = []
    for R in Rs:
        D.append([])
        MT = solve_MT(R, g, b, c) # MT = M + T
        for T in Ts:
            D[-1].append(P_sb(T, R, V=V) * P_sc(MT - T, I(R), gamma=gamma))
    return D


# In[ ]:


D = probability_matrix(Ts, Rs, gamma=0.4, V=5.48)
fig2a_data = graphs.fig2a(D)


# In[ ]:


def update_fig2a(gamma, V):
    D = probability_matrix(Ts, Rs, gamma=gamma, V=V, b=37.1, c=0.42)
    graphs.update_fig2a(fig2a_data, D, gamma, V)

graphs.interact(update_fig2a, continuous_update=False, 
                gamma=graphs.FloatSlider(min=0.1,max= 1.0,step=0.01,value=0.4, continuous_update=False),
                V    =graphs.FloatSlider(min=1.0,max=10.0,step=0.01,value=5.48, continuous_update=False))


# Using the sliders, you can also recreate the four figures S1.A-D of the [Supplementary Information](http://www.pnas.org/content/suppl/2016/05/18/1506752113.DCSupplemental/pnas.201506752SI.pdf). The new value for $\gamma$ and $V$ will be taken into account when you release the sliders, and the figure will take a few seconds to recompute. If the sliders do not work, reexecute the two previous code cells.

# ## Figure 2B: Evolutionary Model

# Figure 2B depicts the dynamics of an evolutionary model. One difference between the evolutionary model and the previous figures is that we don't assume that the intelligence $I$ is a direct function of the radius of the head $R$ anymore. Rather, variations, from the parents to the child, are now linearly correlated between $R$ and $I$ [[2]](#Footnotes).  
# 
# We use a small class for the population. Each agent of the population is represented by its $R$, $T$ and $I$ parameters.

# In[ ]:


class Population:
    
    def __init__(self, R_0, T_0, N=1000, g=g, P_sb=P_sb, P_sc=P_sc, gamma=0.4, V=5.48, b=37.1, c=0.42):
        I_0 = R_0                                                               # I_0 equal to R_0 at initialization.
        self.agents = [np.array((R_0 + np.random.normal(0,1),                     # N agents, with noisy but positive
                                 T_0 + np.random.normal(0,1),                     # R, T and I values around R_0, 
                                 I_0 + np.random.normal(0,1))) for i in range(N)] # T_0 and I_0 respectively.
        for agent in self.agents:
            agent[agent < 1e-3] = 1e-3                                                          # enforcing minimums.

        self.g, self.P_sb, self.P_sc = g, P_sb, P_sc
        self.gamma, self.V, self.b, self.c = gamma, V, b, c
                                
        self.trace = []    # the average value of R, T and I will be appended there every time self.mean() is called.
        self.mean()
        
    def survive(self, R, T, I):
        """Return True if the agent survives to adulthood."""
        if self.P_sb(T, R, g=self.g, V=self.V) > np.random.random():                                # survived birth.
            MT = solve_MT(R, g, b=self.b, c=self.c)
            if self.P_sc(MT - T, I, gamma=self.gamma) > np.random.random():                     # survived childhood.
                return True
        return False

    def mutate(self, R_p, T_p, I_p):
        """Return the child's R, T and I variables, derived from the one of his parents."""
        T = T_p + np.random.normal(0, 1)                                      # random variations for child's values. 
        noise_RI = np.random.normal(0, 1)                                     # correlated variation between R and I.
        R = R_p + noise_RI
        I = I_p + 0.9*noise_RI + (1.0 - 0.9**2)**0.5 * np.random.normal(0, 1)
        return np.array([max(R, 1e-3), max(T, 0), max(I, 1e-3)])                  # set hard minimums for R, T and I.
    
    def step(self):
        """Create a new agent from two random parents.
        
        If the child survives, it replaces a random agent in the population.
        """
        idx_a, idx_b = np.random.choice(len(self.agents), 2)
        p_a, p_b = self.agents[idx_a], self.agents[idx_b]            # 2 parents, chosen at random in the population.
        R_p, T_p, I_p = np.mean((p_a, p_b), axis=0)                                       # averaging parents values.

        R, T, I = self.mutate(R_p, T_p, I_p)
        
        if self.survive(R, T, I_p):
            idx = np.random.randint(0, len(self.agents))                                    # picking a random agent.
            self.agents[idx] = (R, T, I)                                               # replacing it with the child.
            
    def mean(self):
        """Compute the mean R, T of the population"""
        m = np.mean(self.agents, axis=0)
        self.trace.append(m)
        return m


# We create 100 different populations of 100 agents each. Each population has random starting values for $R$ and $T$ (with $0 < R_0 < 8$; $5 < T_0 < 25$; and $R_0 = I_0$), and each agent is given starting values around $R_0$, $T_0$ and $I_0$ with added noise (the noise between $R$ and $I$ is *not* correlated during initialization).
# 
# The evolutionary model will take a few minutes to compute. 

# In[ ]:


def evolve_population(starting_point, pop_class=Population):
    """Evolve a population"""
    R_0, T_0 = starting_point
    pop = pop_class(R_0, T_0, gamma=0.4, V=5.48, b=37.1, c=0.42)
    for t in range(100*len(pop.agents)):      # 100 generations, each generation last as long as the population size. 
        pop.step()
        if (t+1) % len(pop.agents) == 0:
            pop.mean()                                       # every generation, computing of the population average.
    return pop.trace                                             # returning the evolution of the population average.

import multiprocessing                                    # using multiple processes to run populations in parallels.
rts = [(np.random.uniform(0, 8),                                             # random values for R, with 0 <= R <= 8.
        np.random.uniform(5, 25)) for _ in range(100)]                      # random values for T, with 5 <= T <= 25.
traces = multiprocessing.Pool().map(evolve_population, rts)   # evolve populations based on the precomputed R_0, T_0.


# Figure 2B shows the average $R$ and $T$ values for each population with the starting average [[3]](#Footnotes) displayed as a dot, and the evolution of those values through generations displayed as a line starting from that dot.

# In[ ]:


graphs.fig2b(traces, D)


# ## Bonus: Intelligence and Brain Radius Relationship

# In the previous evolutionary model, the changes that occur between $R$ and $I$ are correlated from the parent to the child, but not $R$ and $I$ themselves. Over multiple generations, and thus multiple iteration of these correlated changes, this allows the intelligence and the brain radius to progressively drift away from each other. With more intelligence being better (because it increases children's survival rate to adulthood), and a smaller head radius being better (because it increases birth survival rate), we would expect that the populations would take advantage of this by keeping small heads and slowly increasing their intelligence.
# 
# This is what we observe when we plot the evolution of $R$ as a function of $I$ for each population. In the following figure, the end point of the evolution of each population is displayed as a square. Some populations, near the $(I=4, R=2)$ point even obtain almost twice as much intelligence as their brain size would allow under the assumption $R = I$. 

# In[ ]:


graphs.intelligence_radius_fig(traces)


# Does the model rely this phenomenum for its overall behavior? The answer is no. To show that, we consider a population with changes in $R$ and $I$ being correlated, but where the new value of $I$ derives from the parents value of $R_p$. This avoids the drift of the two values.

# In[ ]:


class PopulationNoDrift(Population):
    
    def mutate(self, R_p, T_p, I_p):
        """Return the child's R, T and I variables, function of his parents."""
        T = T_p + np.random.normal(0, 1)                                         
        noise_RI = np.random.normal(0, 1)                                       
        R = R_p + noise_RI
        I = R_p + 0.9*noise_RI + (1.0 - 0.9**2)**0.5 * np.random.normal(0, 1)          # I derives from R_p, not I_p.
        return np.array([max(R, 1e-3), max(T, 0), max(I, 1e-3)])
    
def evolve_nodrift_pop(starting_point):
    return evolve_population(starting_point, pop_class=PopulationNoDrift)


# In[ ]:


traces_nodrift = multiprocessing.Pool().map(evolve_nodrift_pop, rts)    


# In[ ]:


graphs.fig2b(traces_nodrift, D)


# In[ ]:


graphs.intelligence_radius_fig(traces_nodrift)


# ## Footnotes

# 1. From the article: "To create Fig. 2A, we have assumed that $I_p$, the parent’s intelligence, is equal to the child’s brain size at the limit of growth. This approximation holds in populations in which there are only small changes across generations." (page&nbsp;3).
# 2. From the article: "our model assumes that changes to intelligence are linearly correlated with changes to brain radius." ([Supplementary Information](http://www.pnas.org/content/suppl/2016/05/18/1506752113.DCSupplemental/pnas.201506752SI.pdf), page&nbsp;1)
# 3. The displayed starting point is different from $R_0$, $T_0$ because it is the averaged value across the population. The effect is particularly important near zero, because noisy variations have all been clipped to be positive: the average value of $R$ for the population is therefore never close to zero.

#  
