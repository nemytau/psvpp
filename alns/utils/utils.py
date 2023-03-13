
def build_weekly_departure_scnarios(installations):
    visits = [[] for n in range(0,7)]
    for i, installation in enumerate(installations):
        scenario = installation.random_departure_scenario()
        for day in scenario:
            visits[day].append(i)
    return visits

