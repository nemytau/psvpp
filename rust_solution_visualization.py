import json
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

def load_schedule_df(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['voyages'])

def load_explicit_schedule_df(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['stages'])

def schedule_figure(df):
    start_date = pd.to_datetime('04-10-2022')
    df['Start_datetime'] = start_date + pd.to_timedelta(df.Start, unit='h')
    df['End_datetime'] = start_date + pd.to_timedelta(df.End, unit='h')

    fig = px.timeline(df,
                      x_start='Start_datetime',
                      x_end='End_datetime',
                      y='Vessel',
                      color='Route',
                      text='Load')
    fig.update_xaxes(range=[pd.to_datetime('04-10-2022'), pd.to_datetime('04-17-2022')])
    fig.update_layout(title_text='Schedule from Rust output')
    return fig


# New function: explicit_schedule_figure
def explicit_schedule_figure(df: pd.DataFrame):
    start_date = pd.to_datetime('04-10-2022')
    df['Start_datetime'] = start_date + pd.to_timedelta(df.Start, unit='h')
    df['End_datetime'] = start_date + pd.to_timedelta(df.End, unit='h')

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
                      color='Action',
                      text='Description',
                      color_discrete_map=cm)
    fig.update_xaxes(range=[pd.to_datetime('04-10-2022'), pd.to_datetime('04-17-2022')])
    fig.update_layout(title_text='Explicit Voyage Stages')
    return fig

def draw_figures(figures: list):
    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig) for fig in figures
    ])
    app.run_server(debug=True, use_reloader=False)

def main():
    print("Starting visualization...")
    df = load_schedule_df('output/solution_vis.json')
    fig = schedule_figure(df)
    df_explicit = load_explicit_schedule_df("output/explicit_schedule.json")
    fig_explicit = explicit_schedule_figure(df_explicit)

    draw_figures([fig, fig_explicit])


if __name__ == '__main__':
    main()