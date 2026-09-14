"""Anonymous per-browser player totals, with durable JSON and readable Markdown."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import secrets
import html

TITLES = {'12':'Keys & doors', '10':'Boxes only', '11':'Survival', '13':'Keys, doors & survival'}
COMPETITION_DEADLINE = datetime(2026, 9, 19, 6, 59, tzinfo=timezone.utc)

# Enough pairs that an unused one is almost always free, so a player who declines
# to be named still gets a clean two-word label with nothing appended.
PSEUDONYM_FIRST = ('Cedar', 'Willow', 'Mossy', 'Amber', 'Quiet', 'Silver', 'Maple', 'Fern',
                   'Copper', 'Hazel', 'Dusky', 'Bramble', 'Thistle', 'Juniper', 'Slate', 'Russet')
PSEUDONYM_SECOND = ('Badger', 'Otter', 'Fox', 'Owl', 'Lynx', 'Robin', 'Heron', 'Marten', 'Hare', 'Wren')


class MetricsStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.players = {}

    def player(self, cookie):
        if isinstance(cookie, str) and re.fullmatch(r'[0-9a-f]{32}', cookie):
            path = self.directory / ('player-'+cookie+'.json')
            if cookie in self.players:
                return cookie
            if path.exists():
                self.players[cookie] = json.loads(path.read_text())
                for level in TITLES:
                    self.players[cookie]['levels'].setdefault(level, dict(attempts=0, actions=0, wins=0, failures=0, restarts=0, best_win_actions=None))
                return cookie
        player = secrets.token_hex(16)
        self.players[player] = dict(player=player, created=self.now(), updated=self.now(),
            levels={n:dict(attempts=0, actions=0, wins=0, failures=0, restarts=0,
                           best_win_actions=None) for n in TITLES}, events=[])
        self.save(player)
        return player

    @staticmethod
    def now():
        # Preserve ordering for solves that happen within the same second.
        return datetime.now(timezone.utc).isoformat(timespec='microseconds')

    def record(self, player, level, event, moves=0, seconds=0, reasons=()):
        data = self.players[player]
        counts = data['levels'][level]
        if event == 'start': counts['attempts'] += 1
        elif event == 'restart':
            counts['restarts'] += 1
            counts['attempts'] += 1
        elif event == 'action': counts['actions'] += 1
        elif event == 'win':
            counts['wins'] += 1
            best = counts['best_win_actions']
            previous_time = self.winning_time(data, level)
            if best is None or moves < best or (moves == best and (previous_time is None or seconds < previous_time)):
                counts['best_win_seconds'] = round(seconds, 3)
            counts['best_win_actions'] = moves if best is None else min(best, moves)
        elif event == 'lose': counts['failures'] += 1
        else: raise ValueError('Unknown metric event')
        if event != 'action':
            data['events'].append(dict(time=self.now(), level=level, event=event,
                moves=moves, seconds=round(seconds, 3), reasons=list(reasons)))
        data['updated'] = self.now()
        self.save(player)

    def counts(self, player, level):
        levels = self.players[player]['levels']
        return dict(failures=sum(v['failures'] for v in levels.values()),
            level_failures=levels[level]['failures'])

    def new_pseudonym(self):
        # self.players only holds players this process has touched, so read the
        # folder too; two players sharing a label would be unresolvable at payout.
        taken = {data.get('pseudonym') for data in self.players.values()}
        for path in self.directory.glob('player-*.json'):
            try:
                taken.add(json.loads(path.read_text()).get('pseudonym'))
            except (OSError, ValueError):
                continue
        pairs = [first+' '+second for first in PSEUDONYM_FIRST for second in PSEUDONYM_SECOND]
        free = [pair for pair in pairs if pair not in taken]
        if free:
            return secrets.choice(free)
        base = secrets.choice(pairs)
        suffix = 2
        while '%s %d' % (base, suffix) in taken:
            suffix += 1
        return '%s %d' % (base, suffix)

    def set_name(self, player, name):
        if not isinstance(name, str) or len(name) > 80:
            raise ValueError('Use a name of at most 80 characters.')
        name = ' '.join(name.split())
        if name and self.players[player].get('leaderboard_excluded'):
            raise ValueError('This player is excluded from the prize leaderboard.')
        if any(ord(c) < 32 for c in name):
            raise ValueError('Please use a plain-text name.')
        self.players[player]['display_name'] = name
        if not name and not self.players[player].get('leaderboard_excluded') and not self.players[player].get('pseudonym'):
            self.players[player]['pseudonym'] = self.new_pseudonym()
        self.players[player]['name_choice_made'] = True
        self.players[player]['updated'] = self.now()
        self.save(player)
        return name

    def leaderboard(self):
        boards = {level: [] for level in ('12', '11', '13')}
        for path in self.directory.glob('player-*.json'):
            data = json.loads(path.read_text())
            name = data.get('display_name') or data.get('pseudonym', '')
            if not name or data.get('leaderboard_excluded'):
                continue
            for level in boards:
                cutoff = data.get('leaderboard_since')
                wins = [e for e in data.get('events', []) if e['event']=='win' and e['level']==level
                    and datetime.fromisoformat(e['time']) < COMPETITION_DEADLINE
                    and (not cutoff or datetime.fromisoformat(e['time']) > datetime.fromisoformat(cutoff))]
                # Action count is primary. Equal-action scores are ordered by
                # when the player first solved that challenge, rather than by
                # how long the run took.
                best = min(wins, key=lambda e:(e['moves'],e['time']), default=None)
                if best is not None:
                    boards[level].append(dict(name=name, steps=best['moves'], seconds=best['seconds'],
                        _achieved_at=best['time']))
        for level, rows in boards.items():
            rows.sort(key=lambda row: (row['steps'], row['_achieved_at'], row['name'].casefold()))
            previous = None
            for index, row in enumerate(rows, 1):
                score = (row['steps'], row['_achieved_at'])
                if score != previous:
                    rank = index
                row['rank'] = rank
                previous = score
            boards[level] = [{k:v for k,v in row.items() if not k.startswith('_')} for row in rows[:20]]
        return boards

    def reset_leaderboard(self, player):
        self.players[player]['leaderboard_since'] = self.now()
        self.save(player)

    def exclude_from_leaderboard(self, player):
        self.players[player]['leaderboard_excluded'] = True
        self.set_name(player, '')

    @staticmethod
    def winning_time(data, level):
        counts = data.get('levels', {}).get(level, {})
        if counts.get('best_win_seconds') is not None:
            return counts['best_win_seconds']
        times = [event['seconds'] for event in data.get('events', [])
            if event['event'] == 'win' and event['level'] == level
            and event['moves'] == counts.get('best_win_actions') and 'seconds' in event]
        return min(times) if times else None

    def begin_trajectory(self, player, level, state, level_hash):
        attempt = secrets.token_hex(16)
        self.trajectory(player, attempt, dict(event='start', level=level,
            level_sha256=level_hash, initial_state=dict(state),
            action_names={'0':'interact', '1':'wait', '2':'down', '3':'right', '4':'up', '5':'left'}))
        return attempt

    def trajectory(self, player, attempt, entry):
        # Append one record, rather than rewriting a growing trajectory on every move.
        folder = self.directory / ('player-'+player+'-attempts')
        folder.mkdir(exist_ok=True)
        row = dict(schema_version=1, attempt=attempt, time=self.now(), **entry)
        with (folder / (attempt+'.jsonl')).open('a') as stream:
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')

    def true_trajectory(self, player, attempt, transitions):
        """Atomically publish the currently active path after rewinds.

        The append-only JSONL attempt log remains the complete audit trail. This
        sidecar is the branch with discarded suffixes spliced out, which is the
        trajectory used for the final move count and replay analysis.
        """
        folder = self.directory / ('player-'+player+'-attempts')
        folder.mkdir(exist_ok=True)
        payload = dict(schema_version=1, attempt=attempt, time=self.now(),
                       transitions=transitions)
        self.atomic(folder / (attempt+'.true.json'), json.dumps(payload, separators=(',', ':'))+'\n')

    def save(self, player):
        data = self.players[player]
        prefix = self.directory / ('player-'+player)
        self.atomic(prefix.with_suffix('.json'), json.dumps(data, indent=2)+'\n')
        name = html.escape(data.get('display_name', '') or 'Not provided')
        name = re.sub(r'([\\`*_{}\[\]()#+.!|>~-])', r'\\\1', name)
        lines = ['# Player '+player[:8], '', 'Optional player name: '+name, '',
            'Random browser identifier; the name is volunteered by the player. No email collected.', '',
            'Created: '+data['created'], 'Updated: '+data['updated'], '',
            '**Total failures: '+str(sum(v['failures'] for v in data['levels'].values()))+'**', '',
            'Full state–action trajectories: [attempt files](player-'+player+'-attempts/).',
            'Each JSONL file records an initial joint state and accepted state/action/next-state transitions. Older attempts may predate trajectory logging.', '',
            '| Challenge | Attempts | Actions | Wins | Failures | Restarts | Best winning actions | Time of best run (s) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for level, title in TITLES.items():
            v = data['levels'][level]
            values = [str(v[k]) for k in ('attempts','actions','wins','failures','restarts')]
            elapsed = self.winning_time(data, level)
            lines.append('| '+title+' | '+' | '.join(values)+' | '+str(v['best_win_actions'] or '—')+' | '+(str(elapsed) if elapsed is not None else '—')+' |')
        lines += ['', 'A failure is counted once on YOU LOSE. A voluntary restart is recorded separately.',
            'Actions include movement, interaction, and waiting. Elapsed time includes pauses during an attempt.',
            '', '## Attempt events', '', '| UTC | Challenge | Event | Actions in attempt | Elapsed seconds | Reason |',
            '| --- | --- | --- | ---: | ---: | --- |']
        for e in data['events']:
            reason = '; '.join(str(r) for r in e['reasons']).replace('|','/').replace('\n',' ')
            lines.append(f"| {e['time']} | {TITLES[e['level']]} | {e['event']} | {e['moves']} | {e['seconds']} | {reason} |")
        self.atomic(prefix.with_suffix('.md'), '\n'.join(lines)+'\n')

    @staticmethod
    def atomic(path, content):
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(content)
        temporary.replace(path)
