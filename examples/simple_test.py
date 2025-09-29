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


def schedule_figure(sch):
    start_date = pd.to_datetime('04-10-2022')
    df = sch.to_df_for_visualization()
    df['Start_datetime'] = start_date + pd.to_timedelta(df.Start, unit='h')
    df['End_datetime'] = start_date + pd.to_timedelta(df.End, unit='h')

    fig = px.timeline(df,
                      x_start='Start_datetime',
                      x_end='End_datetime',
                      y='Vessel',
                      color='Route',
                      text='Load',
                      # width=800
                      )
    fig.update_xaxes(range=[pd.to_datetime('04-10-2022'), pd.to_datetime('04-17-2022')])
    # Add total cost to chart title
    fig.update_layout(title_text=f'Schedule with total cost: {sch.total_cost}')
    return fig


def explicit_schedule_figure(sch):
    """

    :param sch:
    :type sch: Schedule
    :return:
    """
    start_date = pd.to_datetime('04-10-2022')
    df = pd.concat([voyage.get_route_stages_df() for voyage in sch.flattened_voyages()])
    df['Start_datetime'] = start_date + pd.to_timedelta(df.start_time, unit='h')
    df['End_datetime'] = start_date + pd.to_timedelta(df.end_time, unit='h')
    df.to_excel('test.xlsx')
    cm = {
        'Service': px.colors.qualitative.D3[1],
        'Sailing': px.colors.qualitative.Dark2[1],
        'Waiting': px.colors.qualitative.Light24[0],
        'Waiting at base': px.colors.qualitative.Pastel2[2],
        'Service at base': px.colors.qualitative.Pastel2[7]
    }
    fig = px.timeline(df,
                      x_start='Start_datetime',
                      x_end='End_datetime',
                      y='Vessel',
                      color='action',
                      text='description',
                      color_discrete_map=cm,
                      # width=800
                      )
    fig.update_xaxes(range=[pd.to_datetime('04-10-2022'), pd.to_datetime('04-17-2022')])
    return fig


def draw_figures(figures: list[go.Figure]):
    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig) for fig in figures
    ])
    app.run_server(debug=True, use_reloader=False)


import time


def main():
    gen_param_name = 'SMALL_2'
    dataset_name = 'test1'

    insts, vessels, base = load_data(gen_param_name=gen_param_name, dataset_name=dataset_name,
                                     source=io.IOSource.SAMPLE)

    # insts, vessels, base = generate_data(gen_param_name=gen_param_name, dataset_name=dataset_name,
    #                                  source=io.IOSource.SAMPLE, save=True)
    # sch = Schedule(insts, vessels, base)
    # dump_solution(sch, gen_param_name, dataset_name, sol_idx=2, source=io.IOSource.SAMPLE)

    sch = load_solution(gen_param_name, dataset_name, sol_idx=2, source=io.IOSource.SAMPLE)

    for inst in sch.installations:
        inst.adjTW = inst.adjust_time_window_for_service(inst.time_window)
    sch.base.adjTW = (8, 8)

    for i, v in enumerate(sch.vessels):
        v.cost = 1000 - i * 50
    print([(v, v.cost) for v in sch.vessels])

    figure1 = schedule_figure(sch)
    xfigure1 = explicit_schedule_figure(sch)

    start_time = time.time()
    removed_visits = worst_removal(sch)
    sch.insert_idle_vessel_and_add_empty_voyages()
    deep_greedy_insertion(removed_visits, sch)
    # k_regret_insertion(removed_visits, sch, 2)
    sch.drop_empty_voyages()
    sch.update()
    destroy_repair_time = time.time()
    # sch = number_of_voyages_reduction(sch)

    figure2 = schedule_figure(sch)
    xfigure2 = explicit_schedule_figure(sch)

    start_improve_time = time.time()
    #sch = fleet_size_reduction(sch)
    #sch = cost_reduction(sch)
    end_improve_time = time.time()

    figure3 = schedule_figure(sch)
    xfigure3 = explicit_schedule_figure(sch)

    start_operator_time = time.time()
    sch = deep_greedy_swap(sch)
    end_operator_time = time.time()

    figure4 = schedule_figure(sch)
    xfigure4 = explicit_schedule_figure(sch)

    print(f'Destroy-repair time: {destroy_repair_time - start_time}')
    print(f'Improvement time: {end_improve_time - start_improve_time}')
    print(f'Operator time: {end_operator_time - start_operator_time}')

    # draw_figures([figure3, figure4])
    draw_figures([figure1, xfigure1,
                  figure2, xfigure2,
                  figure3, xfigure3,
                  figure4, xfigure4])



if __name__ == '__main__':
    main()
