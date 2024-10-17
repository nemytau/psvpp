import alns.utils.io as io
from alns.data_generator import *
from alns.Beans.schedule import Schedule
import time
from alns.utils.utils import *
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from alns.alns.destroy_operator import worst_removal, random_removal
from alns.alns.repair_operator import deep_greedy_insertion, k_regret_insertion
from alns.alns.improve_operator import *
from dash import dcc
from dash import html
import dash


def schedule_figure(sch, title=None):
    start_date = pd.to_datetime('04-10-2022')
    df = sch.to_df_for_visualization()
    df['Start_datetime'] = start_date + pd.to_timedelta(df.Start, unit='h')
    df['End_datetime'] = start_date + pd.to_timedelta(df.End, unit='h')
    nvessels = df['Vessel'].nunique()
    nvisits = sum([len(voyage.route) for voyages in sch.schedule.values() for voyage in voyages])
    suffix = f' with {nvessels} vessels and {nvisits} visits, total cost: {sch.total_cost}'
    if title is None:
        title = f'Schedule {suffix}'
    else:
        title = f'{title} {suffix}'
    fig = px.timeline(df,
                      x_start='Start_datetime',
                      x_end='End_datetime',
                      y='Vessel',
                      color='Route',
                      width=1600,
                      height=200 + nvessels * 100,
                      title=title,
                      )
    fig.update_xaxes(range=[pd.to_datetime('04-10-2022'), pd.to_datetime('04-17-2022')])
    return fig


def draw_figures(figures: list[go.Figure]):
    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig) for fig in figures
    ])
    app.run_server(debug=True, use_reloader=False)


def find_solution(insts, vessels, base, sch=None):
    if sch is None:
        sch = Schedule(vessels, insts, base)
    removed_visits = worst_removal(sch, 5)
    sch.insert_idle_vessel_and_add_empty_voyages()
    deep_greedy_insertion(removed_visits, sch)
    sch.drop_empty_voyages()
    sch.update()
    if not sch.feasible:
        print('Not feasible')
        return sch
    sch = number_of_voyages_reduction(sch)
    sch = fleet_size_reduction(sch)
    return sch

def main():
    gen_param_name = 'SMALL_3'
    dataset_name = 'test1'
    sol_idx = 1

    insts, vessels, base = load_data(gen_param_name=gen_param_name, dataset_name=dataset_name,
                                     source=io.IOSource.SAMPLE)
    # insts, vessels, base = generate_data(gen_param_name=gen_param_name, dataset_name=dataset_name,
    #                                      source=io.IOSource.SAMPLE, save=True)
    # sch = Schedule(vessels, insts, base)
    # dump_solution(sch, gen_param_name, dataset_name, sol_idx=sol_idx, source=io.IOSource.SAMPLE)

    sch = load_solution(gen_param_name, dataset_name, sol_idx=sol_idx, source=io.IOSource.SAMPLE)
    general_solution = find_solution(insts, vessels, base, sch)
    insts_op1 = insts[:14]
    insts_op2 = insts[14:18]
    insts_op3 = insts[18:]
    insts_op12 = insts_op1 + insts_op2
    insts_op13 = insts_op1 + insts_op3
    insts_op23 = insts_op2 + insts_op3
    insts_op123 = insts
    insts_ops = {
        'Coalition [1]': insts_op1,
        'Coalition [2]': insts_op2,
        'Coalition [3]': insts_op3,
        'Coalition [1,2]': insts_op12,
        'Coalition [1,3]': insts_op13,
        'Coalition [2,3]': insts_op23,
        'Coalition [1,2,3]': insts_op123
    }
    solutions = {}
    for coalition, insts_op in insts_ops.items():
        print(coalition)
        if coalition == 'Coalition [1,2,3]':
            solutions[coalition] = general_solution
        else:
            solutions[coalition] = find_solution(insts_op, vessels, base)
    for coalition, solution in solutions.items():
        print(f'{coalition}: {solution.total_cost}')
    figures = [schedule_figure(sch, title=f'Schedule for {coalition}') for coalition, sch in solutions.items()]
    draw_figures(figures)


if __name__ == '__main__':
    main()
