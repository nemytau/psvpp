import alns.utils.io as io
from alns.data_generator import *
from alns.Beans.schedule import Schedule
from alns.alns.alns import ALNS
import time
from alns.utils.utils import *
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
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


def main():
    gen_param_name = 'SMALL_2'
    dataset_name = 'test1'

    insts, vessels, base = load_data(gen_param_name=gen_param_name, dataset_name=dataset_name,
                                     source=io.IOSource.SAMPLE)
    for inst in insts:
        inst.adjTW = inst.adjust_time_window_for_service(inst.time_window)
    base.adjTW = (8, 8)
    for i, v in enumerate(vessels):
        v.cost = 1000 - i * 50


    alns = ALNS(insts, base, vessels)
    start = time.time()
    sch = alns.run()
    end = time.time()
    print(f'Execution time: {end - start}')
    print(sch.total_cost)
    figure = schedule_figure(sch)
    xfigure = explicit_schedule_figure(sch)
    draw_figures([figure, xfigure])

if __name__ == '__main__':
    main()
