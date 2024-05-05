import pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
from enum import IntEnum
import mesa
import networkx as nx

class Type(IntEnum):
    POSITIVE = 0
    NEGATIVE = 1
    NEUTRAL = 2
    
class Urgency(IntEnum):
    NO_EXPECTATION = 0
    REASONABLE_TIMEFRAME = 1
    URGENT = 2
    
class AfterHours(IntEnum):
    TOLERABLE = 0
    STRESSED = 1
  
class Hours(IntEnum):
    DURING = 0
    AFTER = 1

class Person(IntEnum):
    EMPLOYEE = 0
    SUPERVISOR = 1
    BOSS = 2
    
def number_state(model, state):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)

class Message():
    def __init__(self, message_type, urgency, sender, recipient, at_step, work_hours):
        self.message_type = message_type
        self.urgency = urgency
        self.sender = sender
        self.recipient = recipient
        self.at_step = at_step
        self.work_hours = work_hours
    
    def recipient_stress(self):
        message_type_multiplier = 0 if self.message_type == Type.POSITIVE else \
                                  0.01 if self.message_type == Type.NEGATIVE else \
                                  0.001 if self.message_type == Type.NEUTRAL else 0

        urgency_multiplier = 0 if self.urgency == Urgency.NO_EXPECTATION else \
                             0.005 if self.urgency == Urgency.REASONABLE_TIMEFRAME else \
                             0.1 if self.urgency == Urgency.URGENT else 0

        if self.recipient == Person.BOSS:
            sender_multiplier = 0
        elif self.recipient == Person.SUPERVISOR:
            sender_multiplier = 0.001 if self.sender == Person.EMPLOYEE else \
                                0.005 if self.sender == Person.SUPERVISOR else \
                                0.05 if self.sender == Person.BOSS else 0
        else: #recipient == EMPLOYEE
            sender_multiplier = 0.005 if self.sender == Person.EMPLOYEE else \
                                0.05 if self.sender == Person.SUPERVISOR else \
                                0.1 if self.sender == Person.BOSS else 0

        if self.recipient.afterhours == AfterHours.TOLERABLE:
            work_hours_multiplier = 0.001
        else:
            work_hours_multiplier = 0.001 if self.work_hours == Hours.DURING else \
                                    0.25 if self.work_hours == Hours.AFTER else 0
 
        return (message_type_multiplier, urgency_multiplier, sender_multiplier, work_hours_multiplier)

class Worker(mesa.Agent):
    """
    Individual Agent definition and its properties/interaction methods
    """

    def __init__(self, unique_id, model, type, after_hours):
        super().__init__(unique_id, model)
        self.type = type
        self.afterhours = after_hours
        
        self.received_messages = []
        self.sent_messages = []
        self.stress = 0
    
    def step(self):
        logistic_factor = 1 - self.stress / 10
        if not self.random.random() < 0.03: #holidays
            if self.random.random() < self.model.org_comm_frequency * logistic_factor:
                self.new_message()
            if self.random.random() < self.model.org_comm_frequency * logistic_factor:
                self.reply_to_message()
        
    def calculate_stress(self):
        stress = 0
        logistic_factor = 1 - self.stress / 10
        recent_messages = 0
        for message in reversed(self.received_messages): 
            if message.at_step >= self.model.schedule.steps - self.model.message_stress_duration:
                recent_messages += 1
                (message_type_multiplier,
                urgency_multiplier, 
                sender_multiplier, 
                work_hours_multiplier) = message.recipient_stress() 
                stress += message_type_multiplier * logistic_factor
                stress += urgency_multiplier * logistic_factor
                stress += sender_multiplier * logistic_factor
                stress += work_hours_multiplier * logistic_factor
            else:
                break
        stress += (recent_messages * 0.005) * logistic_factor     
        
        logistic_factor = self.stress / 10
        for message in reversed(self.received_messages):
            if self.model.hours == Hours.DURING:
                message.at_step = (self.model.schedule.steps - self.model.message_stress_duration) + 1
            else: 
                if self.model.schedule.steps - message.at_step >= self.model.message_stress_duration:
                    if message.work_hours is Hours.AFTER and self.afterhours is AfterHours.STRESSED:
                        continue
                    (message_type_multiplier,
                    urgency_multiplier, 
                    sender_multiplier, 
                    work_hours_multiplier) = message.recipient_stress() 
                    reduce_stress = 0
                    reduce_stress += message_type_multiplier * logistic_factor
                    reduce_stress += urgency_multiplier * logistic_factor
                    reduce_stress += sender_multiplier * logistic_factor
                    reduce_stress += work_hours_multiplier * logistic_factor
                    stress += -(reduce_stress)
                if self.model.schedule.steps - message.at_step < self.model.message_stress_duration:
                    continue
                else:
                    break
        if self.model.work_hours == Hours.AFTER:
            if recent_messages < 1:
                reduce_stress = 0.05 * logistic_factor
                stress += -(reduce_stress)
            elif recent_messages < 2:
                reduce_stress = 0.01 * logistic_factor
                stress += -(reduce_stress)
        self.stress += stress
        return self.stress
    
    def worker_type(self):
        if self.type == Person.BOSS:
            return 'Boss'
        elif self.type == Person.SUPERVISOR:
            return 'Supervisor'
        else:
            return 'Employee'
    def afterhours_stressed(self):
        if self.afterhours == AfterHours.STRESSED:
            return 'Stressed'
        else:
            if self.type == Person.BOSS or self.type == Person.SUPERVISOR:
                return None
            else:
                return 'Tolerant'
            
    def is_supervisor(self):
        if self.worker_type() in ['Boss', 'Employee']:
            return None
        else:
            return self.worker_type()
        
    def is_boss(self):
        if self.worker_type() in ['Supervisor', 'Employee']:
            return None
        else:
            return self.worker_type()
 
    def num_received_messages(self):
        return len(self.received_messages)
    
    def num_sent_messages(self):
        return len(self.sent_messages)
        
    def new_message(self):
        if self.type == Person.BOSS or self.type == Person.SUPERVISOR:
            if self.model.positive_leadership == True:
                message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.8, 0.05, 0.15])[0]
            else: #bad leadership
                message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.2, 0.6, 0.2])[0]
        else: #EMPLOYEE
            if self.model.positive_leadership == True:
                message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.6, 0.1, 0.3])[0]
            else: #bad leadership
                message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.2, 0.4, 0.4])[0]
                
        if self.model.work_hours == Hours.AFTER:
            urgency = False
            if self.random.random() < self.model.urgent_message_chance:
                urgency = True
                co_workers = self.model.get_neighbors(self.pos, self.random.choices([1,2,3], weights=[0.8, 0.15, 0.05])[0])
            else:
                co_workers = self.model.get_neighbors(self.pos)
            if self.model.messaging_after_hours == True and self.random.random() < (self.model.org_comm_frequency * self.model.after_hours_comm_frequency):
                for recipient in co_workers:
                    if self.random.random() < (self.model.org_comm_frequency * self.model.after_hours_comm_frequency):
                        if urgency:
                            self.sent_messages.append(
                                Message(
                                    message_type, 
                                    Urgency.URGENT, 
                                    self, 
                                    recipient, 
                                    self.model.schedule.steps, 
                                    self.model.work_hours)
                            )
                            recipient.received_messages.append(
                                Message(
                                    message_type, 
                                    Urgency.URGENT, 
                                    self, 
                                    recipient, 
                                    self.model.schedule.steps, 
                                    self.model.work_hours)
                            )
                        else: 
                            self.sent_messages.append(
                                Message(
                                    message_type, 
                                    Urgency.NO_EXPECTATION, 
                                    self, 
                                    recipient, 
                                    self.model.schedule.steps, 
                                    self.model.work_hours)
                            )
                            recipient.received_messages.append(
                                Message(
                                    message_type, 
                                    Urgency.NO_EXPECTATION, 
                                    self, 
                                    recipient, 
                                    self.model.schedule.steps, 
                                    self.model.work_hours)
                            )
                        break;
        else:
            urgency = False
            if self.random.random() < self.model.urgent_message_chance:
                urgency = True
                co_workers = self.model.get_neighbors(self.pos, radius=self.random.choices([1,2,3], weights=[0.5, 0.35, 0.15])[0])
            else:
                co_workers = self.model.get_neighbors(self.pos)
            for recipient in co_workers:
                if self.random.random() < self.model.org_comm_frequency:
                    if urgency:
                        self.sent_messages.append(
                            Message(
                                message_type, 
                                Urgency.URGENT, 
                                self, 
                                recipient, 
                                self.model.schedule.steps, 
                                self.model.work_hours)
                        )
                        recipient.received_messages.append(
                            Message(
                                message_type, 
                                Urgency.URGENT, 
                                self, 
                                recipient, 
                                self.model.schedule.steps, 
                                self.model.work_hours)
                        )
                    else: 
                        self.sent_messages.append(
                            Message(
                                message_type, 
                                Urgency.REASONABLE_TIMEFRAME, 
                                self, 
                                recipient, 
                                self.model.schedule.steps, 
                                self.model.work_hours)
                        )
                        recipient.received_messages.append(
                            Message(
                                message_type, 
                                Urgency.REASONABLE_TIMEFRAME, 
                                self, 
                                recipient, 
                                self.model.schedule.steps, 
                                self.model.work_hours)
                        )
                        
    def reply_to_message(self):
        for message in reversed(self.received_messages):
            if message.at_step == self.model.schedule.steps - 1 and message.urgency != Urgency.NO_EXPECTATION:
                if self.type == Person.BOSS or self.type == Person.SUPERVISOR:
                    if message.message_type == Type.POSITIVE:
                        if self.model.positive_leadership == True:
                            message_type = Type.POSITIVE
                        else: #bad leadership
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.3, 0.5, 0.2])[0]
                    elif message.message_type == Type.NEGATIVE:
                        if self.model.positive_leadership == True:
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.8, 0.05, 0.15])[0]
                        else: #bad leadership
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.2, 0.6, 0.2])[0]
                    else: #message_type == NEUTRAL
                        if self.model.positive_leadership == True:
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.75, 0.05, 0.2])[0]
                        else: #bad leadership
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.2, 0.6, 0.2])[0]
                else: #EMPLOYEE
                    if message.message_type == Type.POSITIVE:
                        if self.model.positive_leadership == True:
                            message_type = self.random.choices([Type.POSITIVE, Type.NEUTRAL], weights=[0.9, 0.1])[0]
                        else: #bad leadership
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.6, 0.1, 0.3])[0]
                    elif message.message_type == Type.NEGATIVE:
                        if self.model.positive_leadership == True:
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.6, 0.1, 0.3])[0]
                        else: #bad leadership
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.1, 0.6, 0.3])[0] 
                    else: #message_type == NEUTRAL
                        if self.model.positive_leadership == True:
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.4, 0.3, 0.3])[0]
                        else: #bad leadership
                            message_type = self.random.choices([Type.POSITIVE, Type.NEGATIVE, Type.NEUTRAL], weights=[0.3, 0.4, 0.3])[0]  
                            
                urgency_reply_frequency = self.model.org_comm_frequency * 1.25 if (self.model.org_comm_frequency * 1.25) - 1 < 0 else 0.99
                if self.random.random() < urgency_reply_frequency:
                    if message.urgency == Urgency.URGENT:
                        self.sent_messages.append(
                           Message(
                               message_type, 
                               message.urgency, 
                               self, 
                               message.sender, 
                               self.model.schedule.steps, 
                               self.model.work_hours)
                        )
                        message.sender.received_messages.append(
                           Message(
                               message_type, 
                               message.urgency, 
                               self, 
                               message.sender, 
                               self.model.schedule.steps, 
                               self.model.work_hours)
                        ) 
                else:   
                    if self.random.random() < self.model.org_comm_frequency:
                        if self.model.work_hours == Hours.DURING:
                            urgency = self.random.choices([Urgency.NO_EXPECTATION, Urgency.REASONABLE_TIMEFRAME], weights=[0.5, 0.5])[0]
                            self.sent_messages.append(
                                Message(message_type, 
                                        urgency,
                                        self, 
                                        message.sender, 
                                        self.model.schedule.steps,
                                        self.model.work_hours)
                            )
                            message.sender.received_messages.append(
                                 Message(message_type, 
                                         urgency,
                                         self, 
                                         message.sender, 
                                         self.model.schedule.steps,
                                         self.model.work_hours)
                            )
                        else:
                            if self.model.messaging_after_hours == True and self.random.random() < self.model.after_hours_comm_frequency:
                                self.sent_messages.append(
                                    Message(
                                        message_type, 
                                        Urgency.NO_EXPECTATION, 
                                        self, 
                                        message.sender, 
                                        self.model.schedule.steps, 
                                        self.model.work_hours)
                                )
                                message.sender.received_messages.append(
                                    Message(
                                        message_type, 
                                        Urgency.NO_EXPECTATION, 
                                        self, 
                                        message.sender, 
                                        self.model.schedule.steps, 
                                        self.model.work_hours)
                                ) 
            else:
                break
                        
class OrgCommunication(mesa.Model):
    """
    A virus model with some number of agents
    """

    def __init__(
        self,
        num_nodes=25,
        num_bosses=1,
        supervisor_ratio=0.2,
        avg_worker_recipients=3,
        org_comm_frequency=0.6,
        after_hours_comm_frequency=0.8,
        stressed_after_hours_ratio=0.4,
        urgent_message_chance=0.1,
        message_stress_duration=8,
        positive_leadership=True,
        messaging_after_hours=True,
        graph=None
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_bosses = num_bosses
        self.supervisor_size = (
            int(supervisor_ratio * num_nodes) if supervisor_ratio <= 0.99 else num_nodes
        )
        #prob = avg_worker_recipients / self.num_nodes
        #self.G = nx.stochastic_block_model(sizes=[self.num_bosses, self.supervisor_size, (self.num_nodes - (self.num_bosses + self.supervisor_size))],
        #                                   p=[[1, 1, 0],
        #                                      [1, 0.1, 0.3],
        #                                      [0, 0.3, prob]])
        if graph:
            self.G = graph
        else:
            self.G = nx.barabasi_albert_graph(n=self.num_nodes, m=avg_worker_recipients)
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)
        
        self.org_comm_frequency = org_comm_frequency
        self.after_hours_comm_frequency = after_hours_comm_frequency
        self.stressed_after_hours_ratio = stressed_after_hours_ratio
        self.urgent_message_chance = urgent_message_chance
        self.message_stress_duration = message_stress_duration 
        self.positive_leadership = positive_leadership 
        self.messaging_after_hours = messaging_after_hours
        
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "After Hours Stressed?": Worker.afterhours_stressed,
                "Worker Type" : Worker.worker_type,
                "Supervisor?" : Worker.is_supervisor,
                "Boss?": Worker.is_boss,
                "Messages Sent": Worker.num_sent_messages,
                "Messages Received": Worker.num_received_messages,
                "Stress": Worker.calculate_stress
            }
        )
        
        self.work_hours = Hours.DURING
        self.hours = 1
        
        # Create agents
        for i, node in enumerate(reversed(list(self.G.nodes()))):
            a = Worker(
                self.next_id(),
                self,
                Person.EMPLOYEE,
                AfterHours.TOLERABLE
            )
            self.schedule.add(a)
            self.grid.place_agent(a, node)
           
        available_to_stress = []
        for i in range(self.num_bosses + self.supervisor_size, len(list(self.G))):
            available_to_stress.append(list(self.G)[i])
        stressed_nodes = self.random.sample(available_to_stress, int(num_nodes * self.stressed_after_hours_ratio))
        for a in self.grid.get_cell_list_contents(stressed_nodes):
            a.afterhours = AfterHours.STRESSED
            
        boss_nodes = []
        for i in range(self.num_bosses):
            boss_nodes.append(list(self.G)[i])
        for a in self.grid.get_cell_list_contents(boss_nodes):
            if a.afterhours == AfterHours.STRESSED:
                a.afterhours = AfterHours.TOLERABLE
            a.type = Person.BOSS
          
        supervisor_nodes = []  
        for i in range(self.num_bosses, self.supervisor_size):
            supervisor_nodes.append(list(self.G)[i]) 
        for a in self.grid.get_cell_list_contents(supervisor_nodes):
            a.type = Person.SUPERVISOR
        
        self.running = True
        self.datacollector.collect(self)
        
    def get_neighbors(
        self, node_id: int, include_center: bool = False, radius: int = 1
    ) -> list[mesa.Agent]:
        """Get all agents in adjacent nodes (within a certain radius)."""
        neighborhood = self.grid.get_neighborhood(node_id, include_center, radius)
        return self.grid.get_cell_list_contents(neighborhood) 
    
    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
        
        #fig,ax=plt.subplots(1,1,figsize=(8,8))
        #title_hours = 'After Hours' if self.work_hours == Hours.AFTER else 'During Hours'
        #self.plot_grid(fig, layout='kamada-kawai', title=f'{title_hours}: {self.hours}')
        
        self.hours = self.calculate_workday()
        
    def calculate_workday(self):
        if self.work_hours == Hours.DURING:
            if self.hours >= 8:
                self.hours = 1
                self.work_hours = Hours.AFTER
            else:
                self.hours += 1
        else:
            if self.hours >= 16:
                self.hours = 1
                self.work_hours = Hours.DURING
            else:
                self.hours += 1
        return self.hours
            
    def plot_grid(self,fig,layout='spring',title='', close=True):
      cmap = ListedColormap(["orange", "lime", "skyblue"])
      
      graph = self.G
      if layout == 'kamada-kawai':      
          pos = nx.kamada_kawai_layout(graph)  
      elif layout == 'circular':
          pos = nx.circular_layout(graph)
      else:
          pos = nx.spring_layout(graph, iterations=5, seed=8)  
      plt.clf()
      ax=fig.add_subplot()
      people = [int(i.type) for i in self.grid.get_all_cell_contents()]
      colors = [cmap(i) for i in people]
      

      nx.draw(graph, pos, node_size=[300 if graph.nodes[node]['agent'][0].afterhours == AfterHours.TOLERABLE else 700 for node in graph.nodes],
              edge_color='gray', node_color=colors, with_labels=True,
              alpha=0.9,font_size=14,ax=ax)
      ax.set_title(title)
      legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), markersize=15, label='Employee'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), markersize=15, label='Supervisor'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(2), markersize=15, label='Boss')]
      plt.legend(bbox_to_anchor=(0.9,1), handles=legend_handles, loc='upper left')
      plt.pause(.5)
      plt.savefig("plots/organization/organization.png")
      if close:
          plt.show(block=False)
          plt.close()
      else:
          plt.show()
      
    def avg_stress(self):
        all_stress = 0
        for a in self.schedule.agents:
            all_stress += a.stress
        
        return all_stress / len(self.schedule.agents)

    def run_model(self, n):
        for i in range(n):
            self.step()
            
def model_running(num_nodes, hours_to_run, after_hours_messaging, graph=None):
    if after_hours_messaging == 0:
        model = OrgCommunication(num_nodes=num_nodes, messaging_after_hours=False, graph=graph)
    else:
        model = OrgCommunication(num_nodes=num_nodes, after_hours_comm_frequency=after_hours_messaging, graph=graph)
    model.run_model(hours_to_run) # 8760 hours in a year

    print(f"The average stress from this run is {model.avg_stress():.2f}.")

    fig,ax=plt.subplots(1,1,figsize=(8,8))
    model.plot_grid(fig, layout='kamada-kawai', title='Organizational ICT Messaging', close=False)
    plt.close()

    data = model.datacollector.get_agent_vars_dataframe()
    fig, ax = plt.subplots(figsize=(10,8))
    plt.clf()
    ax=fig.add_subplot()
    data.groupby('Step').Stress.mean().plot(linewidth=3, color="purple", ax=ax)
    lines = ax.get_lines()
    annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(line, idx):
        posx, posy = [line.get_xdata()[idx], line.get_ydata()[idx]]
        annot.xy = (posx, posy)
        text = f'{line.get_label()}: Hour:{posx}, {posy:.2f}'
        annot.set_text(text)
        # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for line in lines:
                cont, ind = line.contains(event)
                if cont:
                    update_annot(line, ind['ind'][0])
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)  
    plt.title("Average Stress")
    plt.xlabel("Hours (8760 = 1 year)")
    plt.ylabel("Stress (1-10 scale)")
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
    plt.savefig("plots/organization/average_stress.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12,8))
    plt.clf()
    ax=fig.add_subplot()
    df = (data.pivot_table(index=['Step'], columns=['AgentID', 'Worker Type', 'After Hours Stressed?'], values='Stress', aggfunc='sum')
        .reset_index()
        .set_index('Step'))
    df.columns = [ f'{i}:{j}-{k}' for i,j,k in df.columns ]
    df.plot(linewidth=3, ax=ax)
    lines = ax.get_lines()
    annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)   
    plt.title("Stress per Employee")
    plt.xlabel("Hours (8760 = 1 year)")
    plt.ylabel("Stress (1-10 scale)")
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
    legend = plt.legend(bbox_to_anchor=(1.04, 1), shadow=True, loc="upper left")
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.subplots_adjust(right=0.75)
    plt.savefig("plots/organization/stress_per_employee.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12,8))
    plt.clf()
    ax=fig.add_subplot()
    df = (data.pivot_table(index=['Step'], columns=['AgentID', 'Worker Type', 'Supervisor?'], values='Stress', aggfunc='sum')
        .reset_index()
        .set_index('Step'))
    df.columns = [ f'{i}:{j}' for i,j,k in df.columns ]
    df.plot(linewidth=3, ax=ax)
    lines = ax.get_lines()
    annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)   
    plt.title("Stress per Supervisor")
    plt.xlabel("Hours (8760 = 1 year)")
    plt.ylabel("Stress (1-10 scale)")
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
    legend = plt.legend(bbox_to_anchor=(1.04, 1), shadow=True, loc="upper left")
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.subplots_adjust(right=0.75)
    plt.savefig("plots/organization/stress_per_supervisor.png")
    plt.close()



    fig, ax = plt.subplots(figsize=(10,8))
    plt.clf()
    ax=fig.add_subplot()
    (data.pivot_table(index=['Step'], columns='Worker Type', values='Stress', aggfunc='mean')
        .reset_index()
        .set_index('Step')).plot(linewidth=3, ax=ax)
    lines = ax.get_lines()
    annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.title("Average Stress by Worker Type")
    plt.xlabel("Hours (8760 = 1 year)")
    plt.ylabel("Stress (1-10 scale)")
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
    legend = plt.legend()
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.savefig("plots/organization/average_stress_by_type.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10,8))
    plt.clf()
    ax=fig.add_subplot()
    (data.pivot_table(index=['Step'], columns='After Hours Stressed?', values='Stress', aggfunc='mean')
        .reset_index()
        .set_index('Step')).plot(linewidth=3, ax=ax)
    lines = ax.get_lines()
    annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.title("Average Stress by After Hours Stress Type")
    plt.xlabel("Hours (8760 = 1 year)")
    plt.ylabel("Stress (1-10 scale)")
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10])
    legend = plt.legend()
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.savefig("plots/organization/average_stress_by_stress_type.png")
    plt.close()
    
    return model.G
    
if __name__ == "__main__":
    num_nodes = int(input("Enter amount of workers to run communication model on: "))
    hours_to_run = int(input("Enter amount of hours to run communication model on: "))
    graph = model_running(num_nodes=num_nodes, hours_to_run=hours_to_run, after_hours_messaging=0)
    input("Enter any key when you are ready to run light after hours messaging: ")
    model_running(num_nodes=num_nodes, hours_to_run=hours_to_run, after_hours_messaging=0.2, graph=graph)
    input("Enter any key when you are ready to run moderate after hours messaging: ")
    model_running(num_nodes=num_nodes, hours_to_run=hours_to_run, after_hours_messaging=0.5, graph=graph)
    input("Enter any key when you are ready to run high after hours messaging: ")
    model_running(num_nodes=num_nodes, hours_to_run=hours_to_run, after_hours_messaging=0.8, graph=graph)
    

