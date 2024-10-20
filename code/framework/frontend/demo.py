from nicegui import ui
import requests
import random
from io import BytesIO
import base64

API_URL = 'http://localhost/api'  # Ensure that this URL is reachable from the app

# Global authentication state (for simplicity)
user_authenticated = False

def login(username, password):
    global user_authenticated
    # Replace with actual authentication logic
    if username == 'admin' and password == 'admin':
        user_authenticated = True
        ui.notify('Login successful!')
        ui.navigate.to('/')
    else:
        ui.notify('Invalid credentials')

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
            ui.label(f"Details for {model['name']}").classes('text-h5')
            ui.markdown(f"""
            **Version:** {model['version']}
            **Size:** {model['size_mb']} MB
            **Created On:** {model['created_on']}
            """)
            metrics = fetch_model_metrics(model['id'])
            if metrics:
                ui.label('Performance Metrics').classes('text-h6')
                ui.markdown(f"""
                **Accuracy:** {metrics['accuracy']}%
                **Precision:** {metrics['precision']}%
                **Recall:** {metrics['recall']}%
                **Inference Time:** {metrics['inference_time']} ms
                """)
                ui.label('Metrics Over Time').classes('text-h6')
                # Assuming metrics['history'] is a list of dicts with 'epoch', 'accuracy', etc.
                chart_data = {
                    'xAxis': {'type': 'category', 'data': [h['epoch'] for h in metrics['history']]},
                    'yAxis': {'type': 'value'},
                    'series': [
                        {'name': 'Accuracy', 'type': 'line', 'data': [h['accuracy'] for h in metrics['history']]},
                        {'name': 'Loss', 'type': 'line', 'data': [h['loss'] for h in metrics['history']]},
                    ]
                }
                ui.chart(chart_data).classes('w-full h-64')
            ui.button('Close', on_click=dialog.close)
    dialog.open()

@ui.page('/login')
def login_page():
    ui.label('Please Log In').classes('text-h4 text-center')
    username = ui.input('Username')
    password = ui.input('Password', password=True)
    ui.button('Login', on_click=lambda: login(username.value, password.value))

def require_authentication(page_function):
    def wrapper():
        if not user_authenticated:
            ui.navigate.to('/login')
        else:
            page_function()
    return wrapper

def create_header():
    with ui.header().classes('justify-between'):
        ui.label('Manufacturing ML Dashboard').classes('text-h5')
        with ui.row():
            ui.link('Dashboard', '/').classes('text-white text-decoration-none')
            ui.link('Models', '/models').classes('text-white text-decoration-none')
            ui.link('Analytics', '/analytics').classes('text-white text-decoration-none')
            ui.link('Configuration', '/config').classes('text-white text-decoration-none')

@ui.page('/')
@require_authentication
def dashboard():
    create_header()
    with ui.column():
        ui.label('System Overview').classes('text-h4 text-center')
        # Add dashboard content here
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
@require_authentication
def model_management():
    create_header()
    with ui.column():
        ui.label('Model Management').classes('text-h4 text-center')
        
        # List of models from API
        models = fetch_models()
        
        # Display models in cards
        for model in models:
            with ui.card().classes('p-4 m-2'):
                ui.label(model['name']).classes('text-h6')
                ui.label(f"Version: {model['version']}").classes('text-body1')
                ui.label(f"Size: {model['size_mb']} MB").classes('text-body1')
                ui.label(f"Accuracy: {model['accuracy']}%").classes('text-body1')
                with ui.row().classes('justify-between'):
                    ui.button('Trigger', on_click=lambda m=model: ui.notify(f'{m["name"]} triggered!'))
                    ui.button('Configure', on_click=lambda m=model: ui.notify(f'Configure {m["name"]}'))
                    ui.button('View', on_click=lambda m=model: show_model_details(m))
        
        # Additional card with stats
        with ui.card().classes('p-4 m-2'):
            ui.label('Overall Model Stats').classes('text-h6')
            ui.label(f'Total Models: {len(models)}').classes('text-body1')
            active_models = sum(1 for model in models if random.choice([True, False]))
            ui.label(f'Active Models: {active_models}').classes('text-body1')
            average_accuracy = sum(model['accuracy'] for model in models) / len(models) if models else 0
            ui.label(f'Average Accuracy: {average_accuracy:.2f}%').classes('text-body1')

@ui.page('/analytics')
@require_authentication
def analytics():
    create_header()
    # Analytics content here
    import matplotlib.pyplot as plt

    def plot_metric(metric_name, values):
        fig, ax = plt.subplots()
        ax.plot(values, marker='o')
        ax.set_title(metric_name)
        ax.set_xlabel('Time')
        ax.set_ylabel(metric_name)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    with ui.column():
        ui.label('Analytics').classes('text-h4 text-center')
        metrics = {
            'Accuracy': [random.uniform(0.7, 0.99) for _ in range(10)],
            'Precision': [random.uniform(0.7, 0.99) for _ in range(10)],
            'Recall': [random.uniform(0.7, 0.99) for _ in range(10)],
        }
        for metric_name, values in metrics.items():
            with ui.card().classes('p-4 m-2'):
                ui.label(metric_name).classes('text-h6')
                img_data = plot_metric(metric_name, values)
                ui.image(f'data:image/png;base64,{img_data}')

@ui.page('/config')
@require_authentication
def config_page():
    create_header()
    # Add configuration management content here

# Start the NiceGUI app
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host='0.0.0.0', port=5002)