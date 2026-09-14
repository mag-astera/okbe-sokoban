import json
from unittest.mock import patch
import pytest
from test_dbn_composer import setup
import local_h_preview as h

class Response:
    def __init__(self, expr): self.data=json.dumps({'message':{'content':json.dumps(expr)}}).encode()
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,*args): return self.data

def test_validated_proposal_is_read_only():
    expr={'or':[{'var':'hunger','in':[0]},{'var':'hydration','in':[0]}]}
    body={'text':'Either hunger or hydration is zero','dims':{'hunger':30,'hydration':30}}
    original=json.dumps(body)
    with patch.object(h.urllib.request,'urlopen',return_value=Response(expr)):
        result=h.propose(body)
    assert result['condition']==expr
    assert [e['matches'] for e in result['examples']]==[True,False]
    assert json.dumps(body)==original

@pytest.mark.parametrize('expr',[{'var':'invented','in':[0]},{'var':'hunger','in':[30]},{'var':'hunger'},{'or':[]}])
def test_rejects_invalid_model_conditions(expr):
    with patch.object(h.urllib.request,'urlopen',return_value=Response(expr)), pytest.raises(ValueError):
        h.propose({'text':'test','dims':{'hunger':30}})
