
# coding: utf-8

# In[2]:


get_ipython().system(' pip install pyknow ')


# In[10]:


from random import choice
from pyknow import *
from enum import Enum
from textwrap import wrap, dedent
from itertools import chain



class StartupInfo(Fact):
    """Info about startup features."""
    pass

class MarketFit(Enum):
    no = 0
    yes = 1
    
class TeamComposition(Enum):
    weak = 0
    strong = 1
    
class RevenuModel(Enum):
    no = 0
    yes = 1
    
class Take_Feedback_Seriously(Enum):
    no = 0
    yes = 1
    
class Start_With_Small_Test_Markets(Enum):
    no = 0
    yes = 1

class Startup_Focus(Enum):
    no = 0
    yes = 1
class Build_Engaged_Communities(Enum):
    no = 0
    yes = 1
class Innovative_solution(Enum):
    no = 0
    yes = 1
class Good_user_experience(Enum):
    no = 0
    yes = 1
class Planning_effectively(Enum):
    no = 0
    yes = 1
class Learning_all_the_time(Enum):
    no = 0
    yes = 1

    
    

class StartupRuleBase(KnowledgeEngine):
    
    @DefFacts()
    def startup(self):
        print(dedent(""" There are some starup results """))

        for x in chain(MarketFit, TeamComposition, RevenuModel,Take_Feedback_Seriously, Start_With_Small_Test_Markets, Startup_Focus, Build_Engaged_Communities, Innovative_solution, Good_user_experience, Planning_effectively, Learning_all_the_time):
            print(x)
            yield StartupInfo(x)

            
    @Rule(AS.f << StartupInfo(MATCH.v))
    def generate_combinations(self, f, v):
        self.retract(f)
        self.declare(*[Fact(v, x) for x in range(0,2 )])

    @Rule(
        Fact(MarketFit.yes, MATCH.m1),
        Fact(TeamComposition.strong, MATCH.t1),
        Fact(RevenuModel.yes, MATCH.r1 & MATCH.m1 & MATCH.t1 ))
    def success(self):
        print("success")

    @Rule(
        Fact(Take_Feedback_Seriously.yes, MATCH.f1),
        Fact(Build_Engaged_Communities.yes, MATCH.e1),
        Fact(Planning_effectively.yes, MATCH.p1 & MATCH.f1 & MATCH.e1 ))
    def success(self):
        print("success")
        
    @Rule(
        Fact(Learning_all_the_time.yes, MATCH.l1),
        Fact(Innovative_solution.yes, MATCH.i1),
        Fact(Good_user_experience.yes, MATCH.u1 & MATCH.l1 & MATCH.i1 ))
    def success(self):
        print("success")

        
        



engine = StartupRuleBase()
engine.reset()
# engine.declare(Light(color=choice(['green', 'yellow', 'blinking-yellow', 'red'])))
engine.run()

