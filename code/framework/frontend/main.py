from nicegui import ui, app
import requests
import random
from io import BytesIO
import base64

"""
This module defines the main application for the Manufacturing ML Dashboard using the NiceGUI framework.
The application provides several pages including a dashboard, model management, analytics, and configuration.
It interacts with an API to fetch model data and metrics, and displays various statistics and charts.
Note: This application does not include user authentication.
Functions:
- fetch_models(): Fetches the list of models from the API.
- fetch_model_metrics(model_id): Fetches the metrics for a specific model from the API.
- show_model_details(model): Displays a dialog with detailed information about a model.
- create_header(): Creates the header for the application with navigation links.
- dashboard(): Defines the dashboard page with system overview statistics.
- model_management(): Defines the model management page with a list of models and their details.
- analytics(): Defines the analytics page with charts for various performance metrics.
- config_page(): Defines the configuration page (currently empty).
- health_check(): Defines a health check endpoint for the application.
The application runs on host '0.0.0.0' and port 5002.
"""

API_URL = 'http://featherml_api:5001/api'

# Create plotly figure to display metrics in application
import plotly.express as px
import pandas as pd

# Generate some random data for the plot
df = pd.DataFrame({
    'epoch': range(1, 11),
    'accuracy': [random.uniform(90, 95) for _ in range(10)],
    'loss': [random.uniform(0.1, 0.3) for _ in range(10)]
})

fig = px.line(df, x='epoch', y=['accuracy', 'loss'], title='Model Performance Metrics')

# Echart sample 
echart = ui.echart({
    'xAxis': {'type': 'value'},
    'yAxis': {'type': 'category', 'data': ['A', 'B'], 'inverse': True},
    'legend': {'textStyle': {'color': 'gray'}},
    'series': [
        {'type': 'bar', 'name': 'Alpha', 'data': [0.1, 0.2]},
        {'type': 'bar', 'name': 'Beta', 'data': [0.3, 0.4]},
    ],
})

def fetch_models():
    try:
        response = requests.get(f'{API_URL}/models/list_models')
        response.raise_for_status()
        return response.json()['models']
    except requests.RequestException as e:
        ui.notify(f"Error fetching models: {e}")
        return []

def fetch_model_metrics(model_id):
    try:
        response = requests.get(f'{API_URL}/models/{model_id}/metrics')
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        ui.notify(f"Error fetching metrics: {e}")
        return {}

def show_model_details(model):
    with ui.dialog() as dialog:
        with ui.card():
            ui.label(f"Details for {model['model_name']}").classes('text-h5')
            ui.markdown(f"""
            **Framework:** {model['framework_type']}
            **Version:** {model['version']}
            **Description:** {model['description']}
            **Model Path:** {model['model_path']}
            """)

            # Safely access performance_metrics, defaulting to an empty dict if not present
            metrics = model.get('performance_metrics', {})
            if metrics:
                ui.label('Performance Metrics').classes('text-h6')
                ui.markdown(f"""
                {', '.join([f'{metric}: {value}' for metric, value in metrics.items()])}
                """)
            else:
                ui.markdown("No performance metrics available.")

            ui.button('Close', on_click=dialog.close)

    dialog.open()


def update_model():
    try:
        response = requests.post(f'{API_URL}/models/update_model')
        response.raise_for_status()
        ui.notify('Model updated successfully!')
    except requests.RequestException as e:
        ui.notify(f"Error updating model: {e}")
        
def create_header():
    with ui.header().classes('justify-between'):
        ui.label('Manufacturing ML Dashboard').classes('text-h5')
        with ui.row():
            ui.link('Dashboard', '/').classes('text-white text-decoration-none')
            ui.link('Models', '/models').classes('text-white text-decoration-none')
            ui.link('Analytics', '/analytics').classes('text-white text-decoration-none')
            ui.link('Configuration', '/config').classes('text-white text-decoration-none')

@ui.page('/')
def dashboard():
    create_header()
    with ui.column():
        ui.label('System Overview').classes('text-h4 text-center')
        with ui.row().classes('justify-center'):
            with ui.card().classes('p-4 m-2'):
                ui.label('Total Production').classes('text-h6')
                ui.label(f'{random.randint(1000, 5000)} units').classes('text-h4')
            with ui.card().classes('p-4 m-2'):
                ui.label('Active Machines').classes('text-h6')
                ui.label(f'{random.randint(10, 50)}').classes('text-h4')
            with ui.card().classes('p-4 m-2'):
                ui.label('Defective Products').classes('text-h6')
                ui.label(f'{random.randint(0, 100)} units').classes('text-h4')
        with ui.row().classes('justify-center'):
            ui.button('Refresh Data', on_click=lambda: ui.notify('Data refreshed!'))

@ui.page('/models')
def model_management():
    create_header()
    with ui.column().classes('w-full'):
        ui.label('Model Management').classes('text-h4 text-center')

        # Fetch models from the API
        models = fetch_models()
        
        # Overall model statistics and chart
        with ui.card().classes('p-4 m-2 w-full'):
            ui.label('Overall Model Stats').classes('text-h6')
            ui.label(f'Total Models: {len(models)}').classes('text-body1')
            active_models = sum(1 for model in models if random.choice([True, False]))  # Replace with actual logic
            ui.label(f'Active Models: {active_models}').classes('text-body1')
            
        # Add a chart to visualize some metrics across models (this is an example of using echart)
        with ui.card().classes('w-full'):
            ui.label('Performance Metrics').classes('text-h6')

            # Prepare data for chart
            model_names = [model['model_name'] for model in models]
            accuracy_data = [model.get('performance_metrics', {}).get('accuracy', 0) for model in models]
            rmse_data = [model.get('performance_metrics', {}).get('rmse', 0) for model in models]

            chart_data = {
                'tooltip': {
                    'trigger': 'axis',
                    'axisPointer': {'type': 'shadow'}  # Shows tooltip on hover
                },
                'xAxis': {
                    'type': 'category',
                    'data': model_names,
                    'name': 'Model Name',
                    'nameLocation': 'middle',
                    'nameGap': 25  # Distance from the label to the axis
                },
                'yAxis': [
                    {
                        'type': 'value',
                        'name': 'Accuracy',
                        'min': 0,
                        'max': 1,
                        'position': 'left',
                        'axisLabel': {'formatter': '{value} %'}  # Adds a percentage sign
                    },
                    {
                        'type': 'value',
                        'name': 'RMSE',
                        'min': 0,
                        'position': 'right',
                        'axisLabel': {'formatter': '{value}'}
                    }
                ],
                'series': [
                    {
                        'name': 'Accuracy',
                        'type': 'bar',
                        'data': accuracy_data,
                        'yAxisIndex': 0,  # Links to the first yAxis (Accuracy)
                        'itemStyle': {'color': '#1f77b4'},  # Blue color
                        'label': {
                            'show': True,
                            'position': 'insideTop',
                            'formatter': '{c}',  # Shows values on top of the bars
                            'color': 'white'
                        }
                    },
                    {
                        'name': 'RMSE',
                        'type': 'bar',
                        'data': rmse_data,
                        'yAxisIndex': 1,  # Links to the second yAxis (RMSE)
                        'itemStyle': {'color': '#ff7f0e'},  # Orange color
                        'label': {
                            'show': True,
                            'position': 'insideTop',
                            'formatter': '{c}',  # Shows values on top of the bars
                            'color': 'white'
                        }
                    }
                ]
            }
            
            ui.echart(chart_data).classes('w-full h-64')

        # Create a grid layout for the cards
        with ui.row().classes('flex-wrap justify-around'):  # Ensures that items wrap and spread evenly across rows
            for model in models:
                with ui.card().classes('p-2 m-2 w-1/4'):  # Adjusts card size to be smaller (1/4 of the row width)
                    # Display model name, framework, and version
                    ui.label(model['model_name']).classes('text-h6')
                    ui.label(f"Framework: {model['framework_type']}").classes('text-body1')
                    ui.label(f"Version: {model['version']}").classes('text-body1')
                    ui.label(f"Description: {model['description']}").classes('text-body2')
                    
                    # Display performance metrics (handle different types of metrics dynamically)
                    metrics = model.get('performance_metrics', {})
                    if metrics:
                        ui.label('Performance Metrics:').classes('text-h6')
                        with ui.column():
                            for metric, value in metrics.items():
                                ui.label(f"{metric.capitalize()}: {value}").classes('text-body2')

                    # Display additional parameters
                    additional_params = model.get('additional_params', {})
                    if additional_params:
                        ui.label('Additional Parameters:').classes('text-h6')
                        with ui.column():
                            for param, value in additional_params.items():
                                ui.label(f"{param}: {value}").classes('text-body2')

                    # Action buttons (Trigger, Configure, View Details)
                    with ui.row().classes('justify-between'):
                        ui.button('Trigger', on_click=lambda m=model: ui.notify(f'{m["model_name"]} triggered!'))
                        ui.button('Configure', on_click=lambda m=model: ui.notify(f'Configure {m["model_name"]}'))
                        ui.button('View', on_click=lambda m=model: show_model_details(m))

        


@ui.page('/analytics')
def analytics():
    create_header()  # Assuming this function creates your header
    
    with ui.column():
        ui.label('Analytics Dashboard').classes('text-h4 text-center')

        # Example chart data using ui.echart for charting
        chart_data = {
            'xAxis': {'type': 'category', 'data': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']},
            'yAxis': {'type': 'value'},
            'series': [
                {'name': 'Accuracy', 'type': 'line', 'data': [90, 92, 91, 93, 94, 95]},
                {'name': 'Precision', 'type': 'line', 'data': [85, 87, 86, 88, 89, 90]},
                {'name': 'Recall', 'type': 'line', 'data': [80, 82, 81, 83, 84, 85]},
            ]
        }
        with ui.card().classes('p-4 m-2'):
            ui.label('Model Performance Metrics').classes('text-h6')
            # Display chart using ui.echart instead of ui.chart
            ui.echart(options=chart_data).classes('w-full h-64')
            
        

        # Example table data
        table_data = [
            ['Model A', '1.0', '95%', '90%', '85%'],
            ['Model B', '1.1', '94%', '89%', '84%'],
            ['Model C', '2.0', '93%', '88%', '83%'],
            ['Model D', '2.1', '92%', '87%', '82%'],
            ['Model E', '3.0', '91%', '86%', '81%']
        ]
        with ui.table(rows=table_data).classes('w-full'):
            ui.table_head(['Model Name', 'Version', 'Accuracy', 'Precision', 'Recall'])
        # Example of fetching and displaying metrics for 'assembly_line_optimization'
    models = fetch_models()
    with ui.row().classes('justify-center'):
        ui.label('Select Model:').classes('text-h6')
        model_select = ui.select(options=[(model['id'], model['name']) for model in models], on_change=lambda e: show_model_details(e.value))
    ui.button('Show Selected Model Metrics', on_click=lambda: show_model_details(model_select.value))

@ui.page('/config')
def config_page():
    create_header()

@ui.page('/health')
def health_check():
    with ui.card().classes('absolute-center'):
        ui.label('Health Check Status').classes('text-h4')
        ui.label('System is Healthy').classes('text-h6 text-success')
        ui.icon('check_circle', color='green').classes('text-4xl')
        ui.button('Back to Dashboard', on_click=lambda: ui.navigate('/')).classes('mt-4')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host='0.0.0.0', port=5002)