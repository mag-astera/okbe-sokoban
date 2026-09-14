"""Play-only release: immutable levels, compiled DBN dynamics, isolated sessions."""
from copy import deepcopy
from functools import lru_cache
import numpy as np
from pathlib import Path
import json
import hashlib
import os
import secrets
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'runtime/llm_solvers')]
from flask import Flask, jsonify, request
from player_metrics import MetricsStore, COMPETITION_DEADLINE
from datetime import datetime, timezone
import tutorial_server as tutorial
import tutorial_affordances
import tutorial_items
import dbn_composer

tutorial.warm_solver()
if not tutorial.SOLVER['loaded']:
    raise RuntimeError(tutorial.SOLVER['error'])

LEVELS = {}
for path in sorted((ROOT / 'levels').glob('*.json')):
    record = json.loads(path.read_text())
    world = tutorial.normalize_world(record['world'])
    spec = tutorial_affordances.prepare(world, record['tutorial']['spec'])
    spaces = tutorial.build_inspect_spaces(world)
    ordered, affs, _ = dbn_composer.compile_model(spaces, spec)
    number = path.stem.split('_')[1]
    colors = {}
    for aff in spec['affordances']:
        door = next((d for d in world['doors'] if aff['target'] == 'door_'+str(d['id'])), None)
        if not door: continue
        for rule in aff.get('rules', []):
            if rule['alpha'] == 1:
                for part in rule.get('when', {}).get('and', []):
                    if part.get('var', '').startswith('key') and part.get('in') == [1]:
                        colors[part['var']] = door['color']
    LEVELS[number] = dict(world=world, spec=spec, level_hash=hashlib.sha256(path.read_bytes()).hexdigest(), model=dict(spaces=ordered, affs=affs),
        initial=tutorial.inspect_x0(world, spaces), colors=colors,
        title={'10':'Boxes only', '11':'Survival', '12':'Keys & doors', '13':'Keys, doors & survival'}[number])

app = Flask(__name__, static_folder=str(ROOT/'static'), static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 4096
sessions = {}
lock = threading.RLock()
SESSION_TTL = 12 * 3600
MAX_SESSIONS = 500
REWIND_BUDGET = 20
metrics = MetricsStore(os.environ.get("METRICS_DIR", "/tmp/okbe-play-metrics"))


def new_session(level, player):
    now = time.monotonic()
    for token in list(sessions):
        if now - sessions[token]['last'] > SESSION_TTL: del sessions[token]
    if len(sessions) >= MAX_SESSIONS: raise ValueError('Game is busy; please try again later.')
    token = secrets.token_urlsafe(32)
    session = dict(player=player, began=now, timing_started=False, elapsed_final=None, level=level,
        state=dict(LEVELS[level]['initial']), initial_state=dict(LEVELS[level]['initial']),
        history=[], rewinds=0, rewind_remaining=REWIND_BUDGET if level == '13' else 0,
        moves=0, revision=0, last=now)
    sessions[token] = session
    metrics.record(player, level, "start")
    session['attempt'] = metrics.begin_trajectory(player, level, session['state'], LEVELS[level]['level_hash'])
    metrics.true_trajectory(player, session['attempt'], [])
    return token, session


class StochasticTransition(Exception):
    pass


class DeterministicChoice:
    """Only preview an action when every DBN conditional has one outcome."""
    def choice(self, values, p):
        support = np.flatnonzero(np.asarray(p) > 0)
        if len(support) != 1:
            raise StochasticTransition()
        index = int(support[0])
        return index if isinstance(values, (int, np.integer)) else values[index]


def play_transition(level, state, action, rng=None):
    result = dbn_composer.step_model(level['model'], state, action, 'factorized', rng=rng)
    # A blocked directional input does not submit an environment time step.
    counted = not (action in (2, 3, 4, 5) and result['state']['X'] == state['X'])
    if not counted:
        result['state'] = dict(state)
    return result, counted


@lru_cache(maxsize=8192)
def next_moves(level_id, state_items):
    level = LEVELS[level_id]
    state = dict(state_items)
    status = tutorial.play_task_status(level['world'], state)
    if status['violations'] or status['complete']:
        return {}
    moves = {}
    for action in (0, 2, 3, 4, 5):
        try:
            result, counted = play_transition(level, state, action, rng=DeterministicChoice())
        except StochasticTransition:
            continue
        after = tutorial.play_task_status(level['world'], result['state'])
        moves[action] = dict(state=result['state'], counted=counted,
            outcome='lose' if after['violations'] else 'win' if after['complete'] else None)
    return moves


def preview_graph(level_id, state, depth=3):
    """Small, deduplicated DBN lookahead; never advances sessions or records data."""
    graph = {}
    frontier = {tuple(sorted(state.items()))}
    for _ in range(depth):
        following = set()
        for items in frontier:
            key = json.dumps(items, separators=(',', ':'))
            if key in graph:
                continue
            moves = next_moves(level_id, items)
            graph[key] = moves
            following.update(tuple(sorted(move['state'].items())) for move in moves.values()
                             if not move['outcome'])
        frontier = following
    return graph


def snapshot(session):
    level = LEVELS[session['level']]
    state = session['state']
    world = level['world']
    status = tutorial.play_task_status(world, state)
    # UI receives only the map, state and objective, never an editable H specification.
    return dict(level=session['level'], title=level['title'], state=state, moves=session['moves'],
        elapsed_seconds=(session['elapsed_final'] if session['elapsed_final'] is not None else
            round(time.monotonic()-session['began'], 3) if session['timing_started'] else 0),
        timer_running=session['timing_started'] and session['elapsed_final'] is None,
        display_name=metrics.players[session['player']].get('display_name', ''),
        pseudonym=metrics.players[session['player']].get('pseudonym', ''),
        needs_prize_name=bool(not metrics.players[session['player']].get('display_name')
            and not metrics.players[session['player']].get('leaderboard_excluded')
            and any(v.get('wins', 0) for v in metrics.players[session['player']]['levels'].values())),
        leaderboard_excluded=bool(metrics.players[session['player']].get('leaderboard_excluded')),
        name_choice_made=bool(metrics.players[session['player']].get('name_choice_made') or metrics.players[session['player']].get('display_name')),
        transitions=next_moves(session['level'], tuple(sorted(state.items()))),
        preview_graph=preview_graph(session['level'], state),
        metrics=metrics.counts(session['player'],session['level']),
        rewind_available=session['level'] == '13' and bool(session['history']) and session.get('rewind_remaining', 0) > 0,
        rewind_count=session.get('rewinds', 0),
        rewind_remaining=session.get('rewind_remaining', 0),
        revision=session['revision'], outcome='lose' if status['violations'] else 'win' if status['complete'] else None,
        map=dict(rows=world['rows'], cols=world['cols'], walls=world['walls'],
            sources=world['internal_goalstates_2_type_ind'], boxes=world['boxes'], doors=world['doors'],
            items=tutorial_items.item_specs(world), targets=[24,56,72], colors=level['colors'],
            physiology=[dict(name=n, max=s.ns-1) for n,s in [(s.name,s) for s in level['model']['spaces']] if n in ('hunger','hydration')]),
        objective='Push all three boxes onto the tan targets. Any box can fill any target.' +
            (' Keep hunger and hydration above zero.' if 'hunger' in state else ''),
        conditions=tutorial.task_panel_data(world, world['task_clauses'])['human'])


@app.after_request
def headers(response):
    response.headers['Cache-Control'] = 'no-store' if request.path.startswith('/api/') else 'no-cache'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
    return response


@app.get('/')
def index(): return app.send_static_file('index.html')


@app.get('/healthz')
def health(): return jsonify(ok=True, levels=list(LEVELS))


@app.post('/api/play')
def play():
    if not request.is_json: return jsonify(error='JSON required'), 415
    body = request.get_json(silent=True)
    if not isinstance(body, dict): return jsonify(error='Invalid request'), 400
    with lock:
        try:
            operation = body.get('operation')
            if operation == 'start':
                level = str(body.get('level'))
                if level not in LEVELS: raise ValueError('Choose one of the four challenges.')
                player = metrics.player(request.cookies.get('okbe-player'))
                token, session = new_session(level, player)
            else:
                token = body.get('token')
                session = sessions.get(token) if isinstance(token, str) else None
                if not session or time.monotonic()-session['last'] > SESSION_TTL:
                    return jsonify(error='Session expired. Choose a level to start again.'), 410
                level = LEVELS[session['level']]
                if operation == 'restart':
                    metrics.trajectory(session['player'], session['attempt'], dict(event='restart',
                        step=session['moves'], state=dict(session['state']), seconds=round(time.monotonic()-session['began'], 3)))
                    metrics.record(session['player'], session['level'], 'restart', session['moves'], time.monotonic()-session['began'])
                    session['began'] = time.monotonic()
                    session.update(state=dict(level['initial']), initial_state=dict(level['initial']), history=[],
                        rewinds=0, rewind_remaining=REWIND_BUDGET if session['level'] == '13' else 0,
                        moves=0, timing_started=False, elapsed_final=None, revision=session['revision']+1)
                    session['attempt'] = metrics.begin_trajectory(session['player'], session['level'], session['state'], level['level_hash'])
                    metrics.true_trajectory(session['player'], session['attempt'], [])
                elif operation == 'rewind':
                    if session['level'] != '13':
                        raise ValueError('Rewind is available in the final Keys, doors & survival level only.')
                    if body.get('revision') != session['revision']:
                        return jsonify(error='State changed; retry.', game=snapshot(session)), 409
                    if session.get('rewind_remaining', 0) <= 0:
                        raise ValueError('No rewind steps remain in this attempt.')
                    if session['history']:
                        removed = session['history'].pop()
                        restored = dict(session['initial_state']) if not session['history'] else dict(session['history'][-1]['next_state'])
                        restored_elapsed = removed['elapsed_before']
                        session.update(state=restored, moves=len(session['history']), rewinds=session.get('rewinds', 0)+1,
                            rewind_remaining=session.get('rewind_remaining', 0)-1,
                            timing_started=restored_elapsed > 0, elapsed_final=None,
                            began=time.monotonic()-restored_elapsed, revision=session['revision']+1)
                        metrics.trajectory(session['player'], session['attempt'], dict(event='rewind',
                            step=removed['step'], action=removed['action'], state=dict(removed['state']),
                            next_state=dict(removed['next_state']), restored_state=restored,
                            discarded_transition=removed, true_step=len(session['history']),
                            counted=False, seconds=restored_elapsed))
                        metrics.true_trajectory(session['player'], session['attempt'], session['history'])
                elif operation == 'step':
                    action = body.get('action')
                    if type(action) is not int or action not in range(6): raise ValueError('Invalid action.')
                    if body.get('revision') != session['revision']:
                        return jsonify(error='State changed; retry.', game=snapshot(session)), 409
                    status = tutorial.play_task_status(level['world'], session['state'])
                    if not status['complete'] and not status['violations']:
                        result, counted = play_transition(level, session['state'], action)
                        if not counted:
                            metrics.trajectory(session['player'], session['attempt'], dict(event='blocked_input',
                                step=session['moves'], state=dict(session['state']), action=action,
                                next_state=dict(session['state']), counted=False,
                                seconds=round(time.monotonic()-session['began'], 3) if session['timing_started'] else 0))
                        else:
                            elapsed_before = round(time.monotonic()-session['began'], 3) if session['timing_started'] else 0
                            if not session['timing_started']:
                                session['began'] = time.monotonic()
                                session['timing_started'] = True
                            after = tutorial.play_task_status(level['world'], result['state'])
                            elapsed = round(time.monotonic()-session['began'], 3)
                            if after['violations'] or after['complete']:
                                session['elapsed_final'] = elapsed
                            transition = dict(state=dict(session['state']), action=action,
                                next_state=dict(result['state']), alphas=result['alphas'],
                                elapsed_before=elapsed_before, elapsed_after=elapsed,
                                step=session['moves']+1)
                            metrics.trajectory(session['player'], session['attempt'], dict(event='transition',
                                step=session['moves']+1, state=dict(session['state']), action=action,
                                next_state=result['state'], alphas=result['alphas'],
                                seconds=elapsed,
                                outcome='lose' if after['violations'] else 'win' if after['complete'] else None,
                                reasons=after['violations']))
                            session['history'].append(transition)
                            session.update(state=result['state'], moves=session['moves']+1, revision=session['revision']+1)
                            metrics.true_trajectory(session['player'], session['attempt'], session['history'])
                            metrics.record(session['player'], session['level'], 'action')
                            after = tutorial.play_task_status(level['world'], session['state'])
                            if after['violations'] or after['complete']:
                                metrics.record(session['player'], session['level'], 'lose' if after['violations'] else 'win',
                                    session['moves'], session['elapsed_final'], after['violations'])
                elif operation != 'resume': raise ValueError('Unknown operation.')
                session['last'] = time.monotonic()
            response = jsonify(token=token, game=snapshot(session))
            response.set_cookie('okbe-player', session['player'], max_age=365*24*3600,
                httponly=True, secure=True, samesite='Lax')
            return response
        except ValueError as error:
            return jsonify(error=str(error)), 400


@app.post('/api/player-name')
def player_name():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error='JSON required'), 400
    with lock:
        try:
            # Only the browser cookie selects the player; never accept a supplied ID.
            player = metrics.player(request.cookies.get('okbe-player'))
            name = metrics.set_name(player, body.get('name'))
            response = jsonify(name=name, pseudonym=metrics.players[player].get('pseudonym', ''))
            response.set_cookie('okbe-player', player, max_age=365*24*3600,
                httponly=True, secure=True, samesite='Lax')
            return response
        except ValueError as error:
            return jsonify(error=str(error)), 400


@app.get('/api/leaderboard')
def leaderboard():
    with lock:
        now = datetime.now(timezone.utc)
        return jsonify(levels=metrics.leaderboard(), competition=dict(
            deadline=COMPETITION_DEADLINE.isoformat(), server_now=now.isoformat(),
            closed=now >= COMPETITION_DEADLINE))


@app.post('/api/leaderboard/reset')
def reset_leaderboard():
    with lock:
        cookie = request.cookies.get('okbe-player')
        if not cookie:
            return jsonify(error='No player selected'), 400
        player = metrics.player(cookie)
        metrics.reset_leaderboard(player)
        return jsonify(ok=True)


@app.post('/api/leaderboard/exclude')
def exclude_from_leaderboard():
    with lock:
        cookie = request.cookies.get('okbe-player')
        if not cookie:
            return jsonify(error='No player selected'), 400
        player = metrics.player(cookie)
        metrics.exclude_from_leaderboard(player)
        return jsonify(ok=True)


if __name__ == '__main__':
    from waitress import serve
    # Pay first-use kernel compilation before accepting players' first inputs.
    for level_id, level in LEVELS.items():
        preview_graph(level_id, level['initial'])
    port = int(os.environ.get('PORT', '8080'))
    print(f'OKBE Sokoban ready on :{port}', flush=True)
    serve(app, host='0.0.0.0', port=port, threads=8)
