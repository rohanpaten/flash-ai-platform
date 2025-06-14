#!/usr/bin/env python3
"""
Ultra-simple FLASH API server for local testing
No external dependencies except built-in Python libraries
"""

import json
import http.server
import socketserver
import urllib.parse
import random
from datetime import datetime
import threading
import sys
import os

# Simple CORS handler
class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_health_check()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')

    def do_POST(self):
        if self.path == '/predict':
            self.handle_predict()
        elif self.path == '/analyze':
            self.handle_analyze()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')

    def send_health_check(self):
        response = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0-simple",
            "server": "Python built-in"
        }
        self.send_json_response(response)

    def handle_predict(self):
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Generate prediction
            prediction = self.generate_prediction(data)
            self.send_json_response(prediction)
            
        except Exception as e:
            self.send_error_response(str(e))

    def handle_analyze(self):
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Generate prediction first
            prediction = self.generate_prediction(data)
            
            # Add analysis
            analysis = {
                "prediction": prediction,
                "detailed_analysis": {
                    "strengths": ["Strong team background", "Large market opportunity"],
                    "weaknesses": ["Early stage product", "Limited traction"],
                    "opportunities": ["Market expansion", "Strategic partnerships"],
                    "threats": ["Competition", "Market conditions"]
                },
                "next_steps": [
                    "Focus on customer validation",
                    "Improve product-market fit",
                    "Build sustainable revenue model"
                ]
            }
            
            self.send_json_response(analysis)
            
        except Exception as e:
            self.send_error_response(str(e))

    def generate_prediction(self, data):
        """Generate realistic prediction based on input data"""
        
        # Base probability based on funding stage
        funding_stage = data.get("funding_stage", "seed")
        stage_probabilities = {
            "idea": 0.2,
            "seed": 0.35,
            "preseed": 0.25,
            "series_a": 0.45,
            "series_b": 0.55,
            "series_c": 0.65
        }
        base_prob = stage_probabilities.get(funding_stage, 0.35)
        
        # Adjustments based on key factors
        adjustments = 0.0
        
        # Revenue impact
        revenue = data.get("monthly_revenue_usd", 0)
        if revenue > 100000:
            adjustments += 0.15
        elif revenue > 10000:
            adjustments += 0.08
        elif revenue > 1000:
            adjustments += 0.03
        
        # Team factors
        team_exp = data.get("team_domain_expertise", 50)
        if team_exp > 80:
            adjustments += 0.1
        elif team_exp > 60:
            adjustments += 0.05
        
        # Market size
        market_size = data.get("market_size_usd", 1000000000)
        if market_size > 10000000000:  # $10B+
            adjustments += 0.08
        elif market_size > 1000000000:  # $1B+
            adjustments += 0.05
        
        # Product-market fit
        pmf = data.get("product_market_fit_score", 5)
        if pmf > 7:
            adjustments += 0.1
        elif pmf > 5:
            adjustments += 0.03
        
        # Add some randomness
        noise = random.uniform(-0.05, 0.05)
        
        # Calculate final probability
        final_prob = base_prob + adjustments + noise
        final_prob = max(0.05, min(0.95, final_prob))  # Clamp between 5-95%
        
        # Generate CAMP scores
        camp_scores = {
            "capital": min(100, max(0, 40 + (revenue / 10000) + random.uniform(-10, 10))),
            "advantage": min(100, max(0, pmf * 10 + random.uniform(-10, 10))),
            "market": min(100, max(0, 50 + (market_size / 100000000) + random.uniform(-10, 10))),
            "people": min(100, max(0, team_exp + random.uniform(-10, 10)))
        }
        
        # Generate insights
        insights = []
        if camp_scores["capital"] < 50:
            insights.append("Revenue growth needed for sustainability")
        if camp_scores["market"] > 70:
            insights.append("Strong market opportunity identified")
        if camp_scores["people"] > 70:
            insights.append("Experienced team is a key strength")
        if final_prob > 0.6:
            insights.append("Above-average success indicators")
        
        # Determine verdict
        if final_prob >= 0.7:
            verdict = "high_potential"
            recommendation = "Strong investment opportunity"
        elif final_prob >= 0.5:
            verdict = "moderate_potential"
            recommendation = "Promising with risks to address"
        elif final_prob >= 0.3:
            verdict = "early_stage"
            recommendation = "Too early, needs validation"
        else:
            verdict = "high_risk"
            recommendation = "Significant challenges present"
        
        return {
            "success_probability": round(final_prob * 100, 1),
            "confidence_score": 75.0,
            "camp_scores": {k: round(v, 1) for k, v in camp_scores.items()},
            "key_insights": insights,
            "recommendation": recommendation,
            "verdict": verdict
        }

    def send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))

    def send_error_response(self, error_message, status_code=500):
        error_data = {
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.send_json_response(error_data, status_code)

    def log_message(self, format, *args):
        """Override to add timestamps to logs"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sys.stderr.write(f"[{timestamp}] {format % args}\n")

def main():
    PORT = int(os.environ.get('PORT', 8001))
    HOST = os.environ.get('HOST', 'localhost')
    
    print(f"🚀 Starting FLASH Simple API Server")
    print(f"📍 Server: http://{HOST}:{PORT}")
    print(f"❤️  Health: http://{HOST}:{PORT}/health")
    print(f"🔮 Predict: POST http://{HOST}:{PORT}/predict")
    print(f"📊 Analyze: POST http://{HOST}:{PORT}/analyze")
    print(f"⏹️  Stop: Press Ctrl+C")
    print("=" * 50)
    
    try:
        with socketserver.TCPServer((HOST, PORT), CORSHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
