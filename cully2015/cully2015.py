
# coding: utf-8

# In[1]:

get_ipython().run_cell_magic(u'javascript', u'', u'IPython.OutputArea.auto_scroll_threshold = 9999; // avoid collapsing output.')


# # Recode: Robots that can adapt like animals

# We recode the arm experiment of the article "Robot that can adapt like animals" ([DOI](http://dx.doi.org/10.1038/nature14422)) by Antoine Cully, Jeff Clune, Danesh Tarapore and Jean-Baptiste Mouret. The article is available [on the Nature website](http://www.nature.com/nature/journal/v521/n7553/full/nature14422.html), and a preprint [is available here](http://www.isir.upmc.fr/files/2015ACLI3468.pdf). The authors have made the [C++ code used for the experiments](http://pages.isir.upmc.fr/~mouret/code/ite_source_code.tar.gz) in the article available, but it was necessary to consult it to code this Python implementation. The [supplementary information](http://www.nature.com/nature/journal/v521/n7553/extref/nature14422-s1.pdf) document, however, was instrumental to it. This code is available on the [recode github repository](https://github.com/humm/recode), and is published under the [OpenScience License](http://fabien.benureau.com/openscience.html). 
# 
# We won't attempt to summarize or re-explain the aims behind the experiments; we assume the reader is familiar with the article. Moreover, we only implement the arm experiment here, not the hexapod one. The main differences between this code and the one presented in the article are:
# 0. We employ a kinematic, planar simulation for the robotic arm, in place of both the paper's simulation and the real robot.
# 0. We do not filter self-collisions of the arm.
# 0. We do restrict the working area of the arm to the carema field of view.
# 
# The code is divided in two parts: one implementing the MAP-Elites algorithm and another implementing the M-BOA optimization algorithm. The code depends on the [numpy](http://www.numpy.org/), and the [bokeh]() library for the figures. The comments from the paper's pseudocode (Supplementary Figure 1) have been inserted into the code when appropriate. They are prefixed with a double "`##`" sign.
# 
# The code is optimized for comprehension, not efficiency. Obvious optimization can be made, but they would reduce clarity.

# In[2]:

import random, math
import numpy as np
random.seed(0) # same results every time.


# ## MAP-Elites

# ### The robotic arm

# The arm has 8 joints, each with a range of $\pm \pi/2$, and a length of 62 cm. The `arm2d()` function computes the position of the end effector (in meters) given a set of angles, expressed as normalized values between 0 and 1. All the angles values manipulated outside of this function are between 0 and 1.

# In[3]:

ARM_DIM = 8

def arm2d_rad(angles):
    """Return the position of the end effector. Accepts angles in radians."""
    u, v, sum_a, length = 0, 0, 0, 0.62/len(angles)
    for a in angles:
        sum_a += a
        u, v = u + length*math.sin(sum_a), v + length*math.cos(sum_a) # zero pose is at x=0,y=1
    return u, v

def arm2d(angles):
    """Return the position of the end effector. Accept angles in [0, 1]"""
    angles = [math.pi*(a-0.5) for a in angles]
    return arm2d_rad(angles)


# The performance measure for MAP-Elites is the opposite of the variance between joints. Given $p_0, ..., p_8$ the values of the angles, the performance is:
# $$\textrm{performance(angles)} = -\frac{1}{8}\sum_0^8 (p_i - m)^2 \,\,\,\,\textrm{ with } m \textrm{ the average value of the angles: }\,\,\,\, m = \sum_0^8 p_i$$
# Let's remark that the performance in general depends on the value of the angles and on the value of the behavior (returned by `arm2d(angles)`), but it does not here.
# 

# In[4]:

def performance(angles):
    """Performance based on the variance of the angles"""
    m = sum(a for a in angles)/len(angles)
    return -sum((a - m)**2 for a in angles)/len(angles)


# In the rest of the code, a set of 8 angles values will be called a *controller*.

# ### Populating the performance map

# The performance map keeps tracks of the best controllers (here, angles values) for all different observed behaviors. It discretizes the behavioral space into a grid, and for each cell of the grid, keeps only the topmost performing controller. In this implementation we keep three separate python dictionaries for the controllers, behaviour and performance.

# In[5]:

ctrl_map = {}
behv_map = {}
perf_map = {}
all_coos = [] # keeping track of all the non-empty cells to quickly choose a random one


# To populate the performance matrix, $I$ simulations are done. First, a number (variable `B`) of random controllers is tried. And then, for the remaining of the $I$ simulations, mutations of those controllers are tried.

# In[6]:

def map_elites(I=200000, B=400):
    """Populate the performance map"""
    for i in range(I):
        if i < B:                                                      
            c = [random.random() for _ in range(ARM_DIM)]     ## the first 400 controllers are generated randomly.
        else:                                                 ## the next controllers are generated using the map.
            rand_coo = random.choice(all_coos)
            c_prime = ctrl_map[rand_coo]                             ## Randomly select a controller c in the map.
            c = perturb(c_prime)                                          ## Create a randomly modified copy of c.
        behavior = arm2d(c)                       ## Simulate the controller and record its behavioral descriptor.
        p = performance(c)                                                              ## Record its performance.
        add_mat(c, behavior, p)                                                      # Update the performance map.
    
RES = 200 # number of row and columns in the behavioral grid
    
def add_mat(ctrl, behavior, perf):
    """Update the perf map if necessary"""
    x, y = behavior    
    coo = (int((x+0.7)/0.7*RES/2), int((y+0.7)/0.7*RES/2))      # coo is the discretized coordinate of a behavior. 
    perf_old = perf_map.get(coo, float('-inf'))
    if perf_old < perf:          ## If the cell is empty or if perf is better than the current stored performance.
        if not coo in ctrl_map:
            all_coos.append(coo)
        ctrl_map[coo] = ctrl                           ## Associate the controller with its behavioral descriptor.
        perf_map[coo] = perf              ## Store the performance of c′ in the behavior-performance map according
        behv_map[coo] = behavior                                                  ## to its behavioral descriptor.


# So far, what is missing is the `perturb()` function (line 9). The random modification of an existing controller is done using a polynomial mutation operator (see *Multi-Objective Optimization Using Evolutionary Algorithms* by K. Deb 
# (2001), p. 120). For a value $c_i$, whose extremum values are 0 and 1, and given  $r_i$, a random value in [0, 1], the mutation goes:
# $$c_i' = c_i + \delta_i \textrm{ with } \delta_i = \begin{cases} (2r_i)^{1/(\eta_m+1)} + 1 &\mbox{if } r < 0.5 \\ 
# 1 - (2(1-r_i))^{1/(\eta_m+1)} & \mbox{if } r_i \geq 0.5. \end{cases}$$ 
# The value of $\eta_m$ is fixed to 10. 

# In[7]:

ETA_M = 10.0

def mutate(c_i):
    """Polynomial mutation operator (see Deb (2001) p. 120)"""
    r_i = random.random()
    if r_i < 0.5:
        delta_i = (2*r_i)**(1/(ETA_M + 1)) - 1
    else:
        delta_i = 1 - (2*(1 - r_i))**(1/(ETA_M + 1))
    return min(1.0, max(0.0, c_i + delta_i))


# When creating a random perturbation of a controller (i.e. a vector of 8 values in [0,1]), each value has a 12.5% chance to mutate. Therefore:

# In[8]:

MUTATION_RATE = 0.125

def perturb(c):
    """Return a random perturbation of the controller"""
    return [mutate(c_i) if random.random() < MUTATION_RATE else c_i for c_i in c]


# We can now run the MAP-Elites algorithm. Using this implementation (in 2015), 2 millions simulations will take of the order of one minute, depending on your hardware. The original article does 20 million simulations. 

# In[9]:

I = 2000000 # number of simulation
B = 400     # bootstrapping

map_elites(I=I, B=B)


# ### Visualizating the map

# We use [bokeh](http://bokeh.pydata.org/en/latest/) for visualizating the performance map. The plotting code is in the `graph.py` file. The color are displayed on a logarithmic scale.

# In[10]:

import graphs
graphs.variance_map(perf_map, RES)


# Compared to the results of the article (Extended Data Figure 7.c), this performance map differs in the center because we do not prevent self-collisions; they make high-performing postures impossible in the center.

# ## M-BOA Optimization 

# ### A broken arm
# Fourteen differents damage conditions are explored in the article (Extended Data Figure 7.b). Joints can either be stuck at 45°, or have an permanent offset of 45°. For the latter, we assume the offset does not change the range of angles the controller accepts. 

# In[11]:

DMG_COND = 3 # change this for different damage condition

# for each case, the format is (<dict of stuck joint>, <dict of offset joints>)
# the values of the stuck or offset angles can be changed.
damages = [({5:45}, {}), ({4:45}, {}), ({3:45}, {}), ({2:45}, {}), # stuck joint only
           ({}, {5:45}), ({}, {4:45}), ({}, {3:45}), ({}, {2:45}), # offset joint only
           ({2:45}, {5:45}), ({2:45}, {4:45}), ({2:45}, {3:45}),   # stuck then offset joints
           ({5:45}, {2:45}), ({4:45}, {2:45}), ({3:45}, {2:45})]   # offset then stuck joints

def arm2d_broken(angles):
    angles = [math.pi*(a-0.5) for a in angles]
    stuck, offset = damages[DMG_COND]
    for i, a in stuck.items():
        angles[i]  = math.radians(a) # joint stuck at 45 degrees
    for j, a in offset.items():
        angles[j] += math.radians(a) # permanent joint offset of 45 degrees
    return arm2d_rad(angles)


# With the broken arm, the performance function changes. It is now the distance of the end-effector to a fixed target.

# In[12]:

TARGET = 0.0, 0.62 # in m. Change this for different targets.

def performance2(behavior):
    """Performance on broken arm"""
    return -dist(TARGET, behavior)


# ### A kernel function

# The kernel function serves to compute to covariance matrix of the Gaussian process: it quantifies how behaviors are related to one another, and how a performance measure on one behavior affect the estimation of the performance of neighboring behaviors. The article uses the Matérn kernel function with $\nu = 5/2$: 
# $$\textrm{matern}(\mathbf{x}, \mathbf{y}) = \left(1 + \frac{\sqrt{5}{\lVert\mathbf{x}-\mathbf{y}\rVert}}{\rho} + \frac{5{\lVert\mathbf{x}-\mathbf{y}\rVert}^2}{3\rho^2}\right)\textrm{exp}\left(-\frac{\sqrt{5}{\lVert\mathbf{x}-\mathbf{y}\rVert}}{\rho}\right)$$

# In[13]:

RHO = 0.1 # the higher the value, the greater the portion of the performance map will be affected.
          # see section 1.6 of the supplementary information for explanation about this value.

def dist(x, y):
    return math.sqrt(sum((x_i - y_i)**2 for x_i, y_i in zip(x, y)))

def matern(x, y):
    """Return the Matern kernel function (with nu = 5/2)"""
    d = dist(x, y)
    return (1 + 5**0.5*d/RHO + 5*d*d/(3*RHO*RHO))*math.exp(-5**0.5*d/RHO)


# ### M-BOA initialization

# We initialize the performance probility distribution of the broken arm with the performance of the simulations on the new performance metric. 

# In[14]:

P_f = {} # performance probability distribution
perf_simu = {} # performance of the intact arm on the distance performance function

for coo in perf_map.keys():
    behavior = behv_map[coo]
    mu       = performance2(behavior)                                       ## Initialize the mean prior from the map.
    sigma2   = matern(behavior, behavior)           ## Initialize the variance prior (in the common case k(x, x) = 1).
    P_f[coo] = (mu, sigma2)                                                     ## Definition of the Gaussian Process.
    perf_simu[coo] = mu                                                         


# In[15]:

import graphs
graphs.distance_map(perf_simu, RES)
print('color scale range: [{:.3g}, {:.3g}]'.format(min(perf_simu.values()), max(perf_simu.values())))


# ### M-BOA adaptation

# The Map-Based Bayesian Optimization Algorithm (M-BOA) initializes the distribution of performance with the results of the simulation, and updates distribution each time the robot is executed on the broken robot. The loop stops when the broken robot is within 5 cm of the target, or when 20 updates have been made.

# In[16]:

SIGMA2_NOISE = 0.03 # see section 1.6 of the supplementary information for explanation about these values.
KAPPA        = 0.3

tried_coo  = [] # the cells of the map whose controller has been executed on the broken robot. 
tried_behv = [] # the behaviors ...
tried_perf = [] # the performances ...

def stopping_criterion():
    return len(tried_perf) > 0 and max(tried_perf) > -0.05

def select_test():
    """Select the controller to try as the argmax of (mu + KAPPA*sigma2)"""
    max_p, coo_t = float('-inf'), None
    for coo, (m_x, sigma2_x) in P_f.items():
        p = m_x + KAPPA*sigma2_x
        if p > max_p:
            max_p, coo_t = p, coo
    
    return coo_t, ctrl_map[coo_t]

def update_gaussian_process():
    """Update the distribution of the performance"""
    P_diff = np.array([perf_i - perf_simu[coo_i] for perf_i, coo_i in zip(tried_perf, tried_coo)])

    K = np.array([[matern(x, y) for x in tried_behv] for y in tried_behv]) + SIGMA2_NOISE*np.eye(len(tried_behv))
    K_inv = np.linalg.pinv(K)                                         ## Compute the observations' correlation matrix.
    
    for coo in P_f.keys():
        behavior = behv_map[coo]
        k = np.array([matern(behavior, xi_i) for xi_i in tried_behv])          ## Compute the behavior vs. observation 
                                                                                               ## correlation  vector.
        mu = perf_simu[coo] + np.dot(k.T, np.dot(K_inv, P_diff))                                   ## Update the mean.
        sigma2 = matern(behavior, behavior) - np.dot(np.dot(k.T, K_inv), k)                    ## Update the variance.
        P_f[coo] = (mu, sigma2)                                                        ## Update the Gaussian Process.

def adaptation_step():
    coo_t, ctrl_t = select_test()                                ## Select next test (argmax of acquisition function).
    perf_t = performance2(arm2d_broken(ctrl_t))                           ## Evaluation of ctrl_t on the broken robot.

    tried_coo.append(coo_t)
    tried_behv.append(behv_map[coo_t])
    tried_perf.append(perf_t)
    
    update_gaussian_process()                                                          ## Update the Gaussian Process.
    return coo_t
    

perf_maps = [] # storing maps for graphs

while len(tried_behv) < 20 and not stopping_criterion():                                             ## Iteration loop.
    coo = adaptation_step()
    
    tryout = ctrl_map[coo], behv_map[coo], tried_perf[-1]
    perf_maps.append(({coo: e[0] + KAPPA*e[1] for coo, e in P_f.items()}, tryout))
    print('{}. {:4.1f} cm to target'.format(len(tried_behv), -100*tried_perf[-1]))


# The process behind M-BOA can be visualized. The graphs below show the distribution of the acquistion function $\mu + \kappa\sigma^2$ after each update. The black arm is the intact robot, while the red arm is the broken one. The target area is the large red disk. When the broken arm reaches it, the update loop stops.
# 
# At each update, the controller with the highest acquisition score is selected (illustrated by the black, intact arm), and executed on the red, broken arm, and the distribution of the acquisition function is updated. The maximum of the updated map is the white dot. It corresponds to the next controller to be executed on the broken robot.

# In[17]:

import graphs
graphs.plot_maps(perf_maps, RES, damages[DMG_COND], TARGET)


# In order to compare the result with the ground truth, we can compute the entire performance map of the broken arm.

# In[18]:

perf_broken = {coo: performance2(arm2d_broken(c)) for coo, c in ctrl_map.items()}
p_min = max(perf_broken.values())
graphs.distance_map(perf_broken, RES, title='minimum distance to target: {:5.1f} cm'.format(-100*p_min))


# ## Experiment Babbling

# The complete code on this page runs under one minute on most hardware. This allows to quickly modify the code to see how the algorithm reacts. Any capitalized variable can be modified. Different damage conditions, different damage angles can be tried. 
# 
# For any comment or question, contact me at fabien.benureau@gmail.com.  

#  
