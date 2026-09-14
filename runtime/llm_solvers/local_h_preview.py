"""Local language-to-condition proposals; never applies or executes model output."""
import json
import time
import urllib.error
import urllib.request
from dbn_composer import gate

MODEL = 'qwen3.5:9b'


def propose(body):
    text = body.get('text', '').strip()
    dims = body.get('dims', {})
    if not text or len(text) > 4000:
        raise ValueError('Describe a condition in 1–4000 characters.')
    if not isinstance(dims, dict) or not dims or len(dims) > 100:
        raise ValueError('Load the world spaces first.')
    if any(not isinstance(k, str) or not k or type(v) is not int or not 1 <= v <= 2500000 for k,v in dims.items()):
        raise ValueError('Invalid state-space dimensions.')
    leaf = {'type':'object','properties':{'var':{'type':'string','enum':list(dims)},'in':{'type':'array','items':{'type':'integer'}}},'required':['var','in'],'additionalProperties':False}
    bounds = {'type':'object','properties':{'var':{'type':'string','enum':list(dims)}, **{k:{'type':'number'} for k in ('gt','gte','lt','lte')}},'required':['var'],'additionalProperties':False}
    ref = {'$ref':'#/$defs/condition'}
    branches = [leaf, bounds]
    for op in ('and','or'):
        branches.append({'type':'object','properties':{op:{'type':'array','items':ref,'minItems':1}},'required':[op],'additionalProperties':False})
    branches.append({'type':'object','properties':{'not':ref},'required':['not'],'additionalProperties':False})
    schema = {'$defs':{'condition':{'anyOf':branches}},'anyOf':branches}
    prompt = ('Translate a condition to JSON for H-psi. Preserve EVERY clause. '
              'Use and/or arrays for conjunction/disjunction, not for negation, '
              'var and in for membership, and gt/gte/lt/lte for strict/inclusive bounds. '
              'Either/or means at least one; both/and means all. '
              'Only use the available variable names. State indices run from 0 to size minus 1. '
              'action: 0 interact, 1 wait, 2 down, 3 right, 4 up, 5 left. '
              'Conditions read CURRENT state; do not invent next-state or arrival events. '
              'Return only the trigger condition; the user separately selects the effect. '
              'Available sizes: '+json.dumps(dims)+'. Schema: '+json.dumps(schema))
    payload = {'model':MODEL,'stream':False,'think':False,'format':schema,
               'options':{'temperature':0,'num_ctx':8192,'num_predict':1200},
               'messages':[{'role':'system','content':prompt},{'role':'user','content':text}]}
    start = time.monotonic()
    req = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError) as e:
        raise ValueError('Local model unavailable. Start Ollama with `ollama serve` and ensure qwen3.5:9b is downloaded.') from e
    if result.get('done_reason') == 'length':
        raise ValueError('The model ran out of output space. Try a shorter condition.')
    expr = json.loads(result['message']['content'])
    predicate = gate(expr, dims, {})
    examples = []
    for label, state in [('All states zero', {n:0 for n in dims}), ('All states maximum', {n:s-1 for n,s in dims.items()})]:
        examples.append({'label':label,'matches':bool(predicate.eval(state,state.get('action',0)))})
    return {'condition':expr,'examples':examples,'seconds':round(time.monotonic()-start,2),'model':MODEL}
