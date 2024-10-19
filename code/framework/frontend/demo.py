from nicegui import ui
import requests

API_URL = 'http://localhost/api/'  # Ensure that this URL is reachable from the app

def get_status():
    try:
        response = requests.get(f'{API_URL}/status')
        response.raise_for_status()  # Raises an exception for 4xx/5xx responses
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def update_config(new_config):
    try:
        response = requests.post(f'{API_URL}/config', json=new_config)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

@ui.page('/')
def main_page():
    ui.label('Monitoring App')
    
    with ui.row():
        ui.button('Get Status', on_click=lambda: ui.notify(get_status()))
        ui.button('Update Config', on_click=lambda: ui.notify(update_config({'key': 'value'})))

# Start the NiceGUI app
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host='0.0.0.0', port=5001)