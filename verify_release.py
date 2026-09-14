"""Find winning routes for the shipped maps and replay them through the real HTTP API."""
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import app
from build_sokoban_levels import verify


def check():
    reports=[]
    with app.app.test_client() as client:
        for path in sorted((app.ROOT/'levels').glob('*.json')):
            record=json.loads(path.read_text())
            print('Finding route for '+path.stem,flush=True)
            report=verify(record)
            number=path.stem.split('_')[1]
            start=client.post('/api/play',json={'operation':'start','level':number}).get_json()
            token=start['token'];game=start['game']
            other=client.post('/api/play',json={'operation':'start','level':number}).get_json()
            for action in report['steps']:
                result=client.post('/api/play',json={'operation':'step','token':token,'action':action,'revision':game['revision']})
                assert result.status_code==200,result.get_json()
                game=result.get_json()['game']
                assert game['outcome']!='lose'
            assert game['outcome']=='win',game
            assert game['state']==report['final_state']
            terminal=client.post('/api/play',json={'operation':'step','token':token,'action':1,'revision':game['revision']}).get_json()['game']
            assert terminal==game
            assert client.post('/api/play',json={'operation':'resume','token':other['token']}).get_json()['game']['state']==other['game']['state']
            restarted=client.post('/api/play',json={'operation':'restart','token':token}).get_json()['game']
            assert restarted['state']==start['game']['state'] and restarted['moves']==0 and restarted['outcome'] is None
            print(f"Level {number}: WON in {game['moves']} actions; isolated sessions and restart verified",flush=True)
            reports.append(report)
        # A failed physiology session is terminal; actions cannot revive it.
        start=client.post('/api/play',json={'operation':'start','level':'11'}).get_json()
        token=start['token'];game=start['game']
        for i in range(15):
            game=client.post('/api/play',json={'operation':'step','token':token,'action':3 if i%2==0 else 5,'revision':game['revision']}).get_json()['game']
        assert game['outcome']=='lose'
        after=client.post('/api/play',json={'operation':'step','token':token,'action':0,'revision':game['revision']}).get_json()['game']
        assert after==game
        assert client.post('/api/play',json={'operation':'step','token':token,'action':7,'revision':game['revision']}).status_code==400
        assert client.post('/api/dbn',json={'operation':'solve'}).status_code==404
        assert client.post('/api/levels/save',json={}).status_code==404
        assert client.post('/api/play',json={'operation':'resume','token':'unknown'}).status_code==410
        assert client.post('/api/play',json={'operation':'step','token':token,'action':0,'revision':-1}).status_code==409
    Path('/home/dev/okbe-release-verification.json').write_text(json.dumps(reports,indent=2))
    print('PASS: all four levels win through DBN-backed play; terminal outcomes, restart, isolation, invalid requests and removed editor endpoints checked.',flush=True)

if __name__=='__main__':check()
