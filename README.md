There are 3 steps:

If running via the zip file refer the below steps. If cloning via git, then scroll below

1. Kaggle json. Paste the kaggle credentials in the root of this project under .kaggle/ For now, my credentials are in place for the evaluation.
2. OLLAMA_API_KEY. These keys will be disabled after the execution of this project. If my key exceeds limit, kindly replace it with your custom credential under app/core/config.py line 25
3. From the root of the project, run docker compose up --build
    a. Make sure Docker is up and running.
    b. Check if the docker is configured correctly: 
        brew install docker-credential-helper
        nano ~/.docker/config.json
        OR
        vi ~/.docker/config.json

        Change this line: 
            "credsStore": "desktop"
        to
            "credsStore": "osxkeychain"

        Save and close

        Restart docker and try again

4. Upon successful start, the page should be up and running on http://localhost:3000/
5. Paste a kaggle URL selecting either the Team Flow / Baseline Flow (The other flows visible are for future extensibility) 
    Tip: Lookup in ~/data/benchmark/kaggle/download_manifest.json There are links present for the easy execution
6. Enter a query. Ex: Which was the most sold product. (And wait for results)

Thanks!


For the github run:

Navigate to ~/.kaggle and paste your kaggle credentials in the kaggle.json file

1. From the root of the project, run docker compose up --build
    a. Make sure Docker is up and running.
    b. Check if the docker is configured correctly: 
        brew install docker-credential-helper
        nano ~/.docker/config.json
        OR
        vi ~/.docker/config.json

        Change this line: 
            "credsStore": "desktop"
        to
            "credsStore": "osxkeychain"

        Save and close

        Restart docker and try again

2. Upon successful start, the page should be up and running on http://localhost:3000/
3. Paste a kaggle URL selecting either the Team Flow / Baseline Flow (The other flows visible are for future extensibility) 
    Tip: Lookup in ~/data/benchmark/kaggle/download_manifest.json There are links present for the easy execution
4. Enter a query. Ex: Which was the most sold product. (And wait for results)
