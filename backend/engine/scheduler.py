# backend/engine/scheduler.py

from typing import List


class Scheduler:
    def __init__(self, agents: List):
        self.agents = agents

    def get_agents(self) -> List:
        """
        Return agents in fixed order
        """
        return self.agents

    def get_agents_by_role(self, role: str) -> List:
        """
        Filter agents by role (name)
        """
        return [agent for agent in self.agents if agent.name == role]

    def reorder_agents(self, new_order: List[str]):
        """
        Reorder agents based on list of names
        """
        name_to_agent = {agent.name: agent for agent in self.agents}

        reordered = []
        for name in new_order:
            if name in name_to_agent:
                reordered.append(name_to_agent[name])

        self.agents = reordered

    def add_agent(self, agent):
        """
        Add new agent dynamically
        """
        self.agents.append(agent)

    def remove_agent(self, agent_name: str):
        """
        Remove agent by name
        """
        self.agents = [a for a in self.agents if a.name != agent_name]