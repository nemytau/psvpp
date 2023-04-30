from alns.Beans.node import Installation
from typing import List


def daily_visits_from_departure_scenarios(installations: List[Installation], period_length=7):
    """

    :param installations: list of installations
    :param period_length: length of the cycled period
    :return: list of installations' indices that have visit planned on the voyage starting that day
    :rtype: list[list[Installation]]
    """
    visits = [[] for n in range(0, period_length)]
    for installation in installations:
        scenario = installation.random_departure_scenario()
        for day in scenario:
            visits[day].append(installation)
    return visits
