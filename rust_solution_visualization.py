import json
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

def load_schedule_df(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['voyages']), data.get('cost')

def load_explicit_schedule_df(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['stages']), data.get('cost')

def schedule_figure(df, cost=None):
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
    title = 'Schedule from Rust output'
    if cost is not None:
        title += f' (Cost: {cost})'
    fig.update_layout(title_text=title)
    return fig


# New function: explicit_schedule_figure
def explicit_schedule_figure(df: pd.DataFrame, cost=None):
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
    title = 'Explicit Voyage Stages'
    if cost is not None:
        title += f' (Cost: {cost})'
    fig.update_layout(title_text=title)
    return fig

def draw_figures(figures: list):
    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig) for fig in figures
    ])
    app.run_server(debug=True, use_reloader=False)


def testing_intermediate_implementation():
    print("Starting visualization...")
    # Regular schedule views
    df_init, cost_init = load_schedule_df('output/solution_vis.json')
    fig_init = schedule_figure(df_init, cost=cost_init)
    df_destroy, cost_destroy = load_schedule_df('output/solution_vis_after_destroy.json')
    fig_destroy = schedule_figure(df_destroy, cost=cost_destroy)
    df_repair, cost_repair = load_schedule_df('output/solution_vis_after_repair.json')
    fig_repair = schedule_figure(df_repair, cost=cost_repair)
    # NEW: Manually added voyage schedule view
    df_manual, cost_manual = load_schedule_df('output/solution_vis_after_manual_addition.json')
    fig_manual = schedule_figure(df_manual, cost=cost_manual)

    # Explicit schedule views
    df_explicit_init, cost_explicit_init = load_explicit_schedule_df('output/explicit_schedule.json')
    fig_explicit_init = explicit_schedule_figure(df_explicit_init, cost=cost_explicit_init)
    df_explicit_destroy, cost_explicit_destroy = load_explicit_schedule_df('output/explicit_schedule_after_destroy.json')
    fig_explicit_destroy = explicit_schedule_figure(df_explicit_destroy, cost=cost_explicit_destroy)
    df_explicit_repair, cost_explicit_repair = load_explicit_schedule_df('output/explicit_schedule_after_repair.json')
    fig_explicit_repair = explicit_schedule_figure(df_explicit_repair, cost=cost_explicit_repair)

    draw_figures([
        fig_init, fig_destroy, fig_repair,  # regular
        fig_explicit_init, fig_explicit_destroy, fig_explicit_repair,   # explicit
    ])

def testing_improvement_operators():
    print("Testing improvement operators...")
    # Load initial schedule
    df_init, cost_init = load_schedule_df('output/voyage_reduction_init_seed42.json')
    fig_init = schedule_figure(df_init, cost=cost_init)
    
    # Load improved schedule
    df_impr, cost_impr = load_schedule_df('output/voyage_reduction_improved_seed42.json')
    fig_impr = schedule_figure(df_impr, cost=cost_impr)
    
    # Draw both figures for comparison
    draw_figures([fig_init, fig_impr])
     
def main():
    # You can uncomment the function you want to test
    # testing_intermediate_implementation()
    testing_improvement_operators()


if __name__ == '__main__':
    main()