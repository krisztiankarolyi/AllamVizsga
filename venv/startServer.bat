@echo off
cd "C:\Users\Kiki\OneDrive\Documents\Sapi 3.1\projektek\AllamVizsga\venv"
rem Activate virtual environment
call \Scripts\activate

rem Start Django development server
cd "\AllamVizsga"
python manage.py runserver
pause