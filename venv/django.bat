start "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" "http://127.0.0.1:8000/"
start "" "C:\AllamVizsga\Szakdolgozat.docx"
cmd /k "cd /d C:\AllamVizsga\venv\Scripts & activate & python C:\AllamVizsga\venv\AllamVizsga\manage.py runserver"
echo "django has started"