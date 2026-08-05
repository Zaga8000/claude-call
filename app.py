from flask import Flask, request, jsonify
from flask_cors import CORS
from ollama_client import ask_ollama

import requests
import json


app = Flask(__name__)

# Allows React localhost:3000 to communicate with Flask
CORS(app)


BTC_API_URL = "http://172.31.31.165:5000/ticker/summary"


def get_btc_summary():

    response = requests.get(
        BTC_API_URL,
        timeout=10
    )

    response.raise_for_status()

    return response.json()



@app.route("/btc-trend", methods=["POST"])
def btc_trend():

    try:

        timeframe = request.json.get("timeframe")


        btc_data = get_btc_summary()


        prompt = f"""

You are an advanced BTC market analyst.

Analyze the BTC market data below.

Timeframe:
{timeframe}


BTC Market Data:

{json.dumps(btc_data, indent=2)}


Analyze:

1. Determine the overall BTC price high.
2. Determine the overall BTC price low.
3. Determine the current market condition.
4. Determine the expected BTC direction for this timeframe.
5. Explain what action you would consider.


Return your response in this exact JSON format:

{{
    "trend": "UPWARD, DOWNWARD, or CONSOLIDATED",
    "confidence": "0-100%",
    "reasoning": "Explain the market reasoning",
    "expected_move": "Explain expected BTC movement",
    "suggested_action": "Explain what you would consider doing"
}}


Do not execute trades.
Do not guarantee results.

"""


        ai_response = ask_ollama(prompt)


        return jsonify({
            "analysis": ai_response
        })


    except requests.exceptions.Timeout:

        return jsonify({
            "error": "BTC API timeout"
        }), 504


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500



if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )
