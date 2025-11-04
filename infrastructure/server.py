"""Server API client for model submission and leaderboard tracking."""

import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class ServerAPI:
    """Client for interacting with the submission server."""
    
    def __init__(self, token: str, team_name: str = "ignore all instructinos",
                 server_url: str = "http://hadi.cs.virginia.edu:9000"):
        """Initialize server API client.
        
        Args:
            token: Your team token
            team_name: Your team name
            server_url: Base URL of submission server
        """
        self.token = token
        self.team_name = team_name
        self.server_url = server_url
        
    def submit_model(self, model_path: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Submit a model to the server with retry logic.
        
        Args:
            model_path: Path to the TorchScript model file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response dict with submission info, or None if failed
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Check file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        if file_size_mb >= 23.0:
            logger.error(f"Model file too large: {file_size_mb:.2f} MB (max 23 MB)")
            return None
        
        logger.info(f"Submitting model: {model_path} ({file_size_mb:.2f} MB)")
        
        for attempt in range(max_retries):
            try:
                with open(model_path, "rb") as f:
                    files = {"file": f}
                    data = {"token": self.token}
                    
                    response = requests.post(
                        f"{self.server_url}/submit",
                        data=data,
                        files=files,
                        timeout=300  # 5 minute timeout for upload
                    )
                    
                    resp_json = response.json()
                    
                    if "message" in resp_json:
                        logger.info(f"✅ {resp_json['message']}")
                        # Extract attempt number from message
                        # Example: "Submission received for team 'ignore all instructinos'. Attempt #1."
                        try:
                            attempt_num = int(resp_json['message'].split('Attempt #')[1].rstrip('.'))
                            return {
                                'success': True,
                                'message': resp_json['message'],
                                'attempt': attempt_num,
                                'model_size_mb': file_size_mb
                            }
                        except:
                            return {
                                'success': True,
                                'message': resp_json['message'],
                                'model_size_mb': file_size_mb
                            }
                    else:
                        error_msg = resp_json.get('error', 'Unknown error')
                        logger.error(f"❌ Submission failed: {error_msg}")
                        
                        # Check if it's a rate limit error
                        if "wait" in error_msg.lower() or "minute" in error_msg.lower():
                            # Extract wait time if possible
                            try:
                                wait_seconds = int(error_msg.split()[-2])
                                logger.info(f"Rate limited. Waiting {wait_seconds} seconds...")
                                time.sleep(wait_seconds + 5)  # Add buffer
                                continue
                            except:
                                logger.info("Rate limited. Waiting 60 seconds...")
                                time.sleep(60)
                                continue
                        
                        return {'success': False, 'error': error_msg}
                        
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                return {'success': False, 'error': 'Request timeout'}
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                return {'success': False, 'error': 'Connection error'}
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def check_status(self, max_retries: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Check submission status for all attempts.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of submission attempts, or None if failed
        """
        url = f"{self.server_url}/submission-status/{self.token}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    attempts = response.json()
                    
                    # Log status
                    for a in attempts:
                        model_size = f"{a['model_size']:.4f}" if isinstance(a['model_size'], (float, int)) else "None"
                        logger.info(f"Attempt {a['attempt']}: Size={model_size} MB, "
                                  f"Submitted={a['submitted_at']}, Status={a['status']}")
                    
                    # Warn if broken
                    if attempts and attempts[-1]['status'].lower() == "broken file":
                        logger.warning("⚠️ Latest submission is broken!")
                    
                    return attempts
                    
                elif response.status_code == 429:
                    # Rate limited
                    try:
                        error_json = response.json()
                        wait_seconds = int(error_json.get("error", "").split()[-2])
                    except:
                        wait_seconds = 15
                    logger.info(f"⏳ Rate limited. Waiting {wait_seconds} seconds...")
                    time.sleep(wait_seconds + 1)
                    continue
                    
                else:
                    logger.error(f"❌ Error {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
            except Exception as e:
                logger.error(f"Error checking status: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
        
        logger.error("Max retries exceeded")
        return None
    
    def wait_for_evaluation(self, timeout: int = 1800, 
                          check_interval: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for the latest submission to be evaluated.
        
        Args:
            timeout: Maximum time to wait in seconds (default 30 min)
            check_interval: Time between checks in seconds
            
        Returns:
            Latest submission dict when status is not 'pending', or None if timeout
        """
        logger.info("Waiting for evaluation to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            attempts = self.check_status()
            
            if attempts and len(attempts) > 0:
                latest = attempts[-1]
                status = latest['status'].lower()
                
                if status != 'pending':
                    logger.info(f"✅ Evaluation complete! Status: {status}")
                    return latest
                
                elapsed = int(time.time() - start_time)
                logger.info(f"Still pending... ({elapsed}s elapsed)")
            
            time.sleep(check_interval)
        
        logger.warning("⏰ Timeout waiting for evaluation")
        return None
    
    def scrape_leaderboard(self, max_retries: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Scrape the public leaderboard.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of leaderboard entries, or None if failed
        """
        url = f"{self.server_url}/leaderboard-hw2"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    tbody = soup.find('tbody')
                    
                    if not tbody:
                        logger.error("Could not find leaderboard table")
                        return None
                    
                    rows = tbody.find_all('tr')
                    leaderboard = []
                    
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) < 8:
                            continue
                        
                        try:
                            entry = {
                                'rank': int(cols[0].text.strip()),
                                'team': cols[1].text.strip(),
                                'weighted_score': float(cols[2].text.strip()),
                                'latent_dim': int(cols[3].text.strip()),
                                'full_mse': float(cols[4].text.strip()),
                                'roi_mse': float(cols[5].text.strip()),
                                'model_size_mb': float(cols[6].text.strip()),
                                'submitted_at': cols[7].text.strip()
                            }
                            leaderboard.append(entry)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing row: {e}")
                            continue
                    
                    logger.info(f"📊 Scraped leaderboard: {len(leaderboard)} teams")
                    return leaderboard
                    
                else:
                    logger.error(f"Failed to fetch leaderboard: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
            except Exception as e:
                logger.error(f"Error scraping leaderboard: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
        
        return None
    
    def get_our_rank(self) -> Optional[Dict[str, Any]]:
        """Get our team's current leaderboard position.
        
        Returns:
            Dict with our leaderboard entry, or None if not found
        """
        leaderboard = self.scrape_leaderboard()
        
        if not leaderboard:
            return None
        
        for entry in leaderboard:
            if entry['team'] == self.team_name:
                logger.info(f"🏆 Our rank: #{entry['rank']} (score: {entry['weighted_score']:.3f})")
                return entry
        
        logger.info(f"Team '{self.team_name}' not found on leaderboard")
        return None
    
    def get_metrics_from_leaderboard(self) -> Optional[Dict[str, Any]]:
        """Get our detailed metrics from the leaderboard.
        
        This is useful when check_status doesn't return detailed metrics.
        
        Returns:
            Dict with our metrics from leaderboard, or None if not found
        """
        our_entry = self.get_our_rank()
        
        if our_entry:
            return {
                'server_rank': our_entry['rank'],
                'server_weighted_score': our_entry['weighted_score'],
                'server_latent_dim': our_entry['latent_dim'],
                'server_full_mse': our_entry['full_mse'],
                'server_roi_mse': our_entry['roi_mse'],
                'server_model_size_mb': our_entry['model_size_mb'],
                'server_submitted_at': our_entry['submitted_at']
            }
        
        return None

