'use strict';
const $=id=>document.getElementById(id), canvas=$('board'), ctx=canvas.getContext('2d');
let game=null, token=sessionStorage.getItem('okbe-play-token'), busy=false;
// Deliberately page-local: a reload or new tab asks unnamed players again.
let welcomeDismissed=false;
function rememberNameChoice(){welcomeDismissed=true;}
// Challenge progression
const LEVEL_ORDER = ['12','11','13'];
let advanceTimer = null;
function cancelAdvance(){if(advanceTimer!==null)clearTimeout(advanceTimer);advanceTimer=null;}
function replayLevel(){cancelAdvance();stopMoving();request('restart');}
function scheduleNextLevel(){
 cancelAdvance();
 if(game.outcome!=='win')return;
 stopMoving();
 const next = LEVEL_ORDER[LEVEL_ORDER.indexOf(game.level)+1];
 if(!next){$('interaction-hint').textContent='All three challenges complete!';return;}
 $('interaction-hint').textContent='Next level in 5 seconds — or click Replay this level.';
 const wonToken=token;
 advanceTimer=setTimeout(()=>{
  advanceTimer=null;
  if(token===wonToken && game.outcome==='win')request('start',{level:next});
 },5000);
}
// Loss countdown is attached to one confirmed terminal state.
let lossTimer=null,lossKey=null,lossRevealTimer=null,lossVisible=false;
function cancelLossRestart(){
 if(lossTimer!==null)clearInterval(lossTimer);
 if(lossRevealTimer!==null)clearTimeout(lossRevealTimer);
 lossRevealTimer=null;lossVisible=false;lossTimer=null;lossKey=null;
 $('loss-countdown').hidden=true;$('loss-caption').hidden=true;
}
function scheduleLossRestart(){
 if(game.outcome!=='lose')return;
 const key=token+':'+game.revision;if(key===lossKey)return;
 cancelLossRestart();stopMoving();pendingInputs.length=0;lossKey=key;
 lossRevealTimer=setTimeout(()=>{
  lossRevealTimer=null;
  if(token+':'+game.revision!==key||game.outcome!=='lose'){cancelLossRestart();return;}
  lossVisible=true;$('outcome').hidden=false;
  let seconds=3;
  $('loss-countdown').hidden=false;$('loss-caption').hidden=false;$('loss-caption').innerHTML='Restarting…<br>Press R to restart now';
  $('loss-seconds').textContent='3';
  lossTimer=setInterval(()=>{
   if(token+':'+game.revision!==key||game.outcome!=='lose'){cancelLossRestart();return;}
   seconds--;
   if(seconds===0){cancelLossRestart();request('restart',{automatic:true});return;}
   $('loss-seconds').textContent=String(seconds);
  },1000);
 },300);
}
// Drawing
const images={};
function image(name){if(!images[name]){const im=new Image();im.src='/static/img/'+name;im.onload=draw;images[name]=im;}return images[name];}
function sprite(name,x,y,size){const im=image(name);if(!im.complete||!im.naturalWidth)return;const scale=size/Math.max(im.naturalWidth,im.naturalHeight);ctx.drawImage(im,x+(size-im.naturalWidth*scale)/2,y+(size-im.naturalHeight*scale)/2,im.naturalWidth*scale,im.naturalHeight*scale);}
function draw(){
 ctx.clearRect(0,0,720,720);if(!game)return;
 const m=game.map,s=game.state,c=720/m.cols;
 const wall=new Set(m.walls),targets=new Set(m.targets);
 for(let cell=0;cell<m.rows*m.cols;cell++){
  const x=(cell%m.cols)*c,y=Math.floor(cell/m.cols)*c;
  ctx.fillStyle='#c4ceb6';ctx.fillRect(x,y,c,c);ctx.strokeStyle='#829079';ctx.lineWidth=.7;ctx.strokeRect(x,y,c,c);
  if(targets.has(cell)){ctx.fillStyle='#d7b77e';ctx.strokeStyle='#927346';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x+c/2,y+c/2,c*.23,0,Math.PI*2);ctx.fill();ctx.stroke();}
  if(wall.has(cell))sprite('mountain.png',x+c*.05,y+c*.05,c*.9);
  if(m.sources[cell]!==undefined)sprite(m.sources[cell]===0?'tree.png':'lake.png',x+c*.05,y+c*.05,c*.9);
  for(const item of m.items)if(item.cell===cell&&!s[item.name])sprite(m.colors[item.name]?'key_'+m.colors[item.name]+'.png':'key.png',x+c*.2,y+c*.2,c*.6);
  for(const d of m.doors)if(d.cell===cell)sprite(s['door_'+d.id]?'mountain_w_door_open.png':'mountain_w_door_closed_'+d.color+'.png',x,y,c);
  for(const b of m.boxes)if(s['box_'+b.id]===cell)sprite('box.png',x+c*.1,y+c*.1,c*.8);
  if(s.X===cell){
   const meters=m.physiology;
   sprite('Stoffel_pic_right.png',x+c*(meters.length?.17:.11),y+c*(meters.length?.32:.11),c*(meters.length?.66:.78));
   if(game.outcome==='lose')sprite('skull.png',x+c*.17,y+c*(meters.length?.32:.17),c*.66);
   meters.forEach((p,i)=>{
    const bx=x+c*.27,by=y+c*(.08+i*.125),w=c*.56,h=c*.065;
    const color=p.name==='hunger'?'#a2d07e':'#6bbcf5';
    ctx.fillStyle='#27313b';ctx.fillRect(bx-1,by-1,w+2,h+2);
    ctx.fillStyle=color;ctx.fillRect(bx,by,w*Math.max(0,Math.min(1,s[p.name]/p.max)),h);
    ctx.save();ctx.font=`700 ${c*.135}px system-ui`;ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillStyle='#000';
    const text=String(s[p.name]),tx=x+c*.23,ty=by+h/2;
    ctx.fillText(text,tx,ty,c*.21);ctx.restore();
   });
  }
 }
}
// Load before the first loss so the overlay appears without an image-fetch delay.
image('skull.png');
function render(confirmed=true){
 syncRunClock(confirmed);
 $('player-profile').hidden=!!game.leaderboard_excluded;
 $('prize-name-reminder').hidden=!game.needs_prize_name;
 if(game.needs_prize_name)$('prize-name-reminder').textContent='You won! Add your name above so we can identify you if you win a prize. Your saved score will stay attached to you.';
 if(!$('player-name').dataset.dirty)$('player-name').value=game.display_name||'';
 $('title').textContent=game.title;$('objective').textContent=game.objective.replace('Keep hunger and hydration above zero.', 'Keep hunger and hydration above zero. You will die at zero even if you reach the lake or tree on that turn.');$('conditions').textContent=typeof game.conditions==='string'?game.conditions:JSON.stringify(game.conditions);
 $('failures').textContent=`Failures: ${game.metrics?.failures ?? 0} · this challenge: ${game.metrics?.level_failures ?? 0}`;
 $('moves').textContent=String(game.moves);document.querySelectorAll('[data-level]').forEach(b=>b.classList.toggle('active',b.dataset.level===game.level));
 if(!$('rewind-hint')){const hint=document.createElement('div');hint.id='rewind-hint';document.querySelector('.board-wrap').prepend(hint);}
 $('rewind-hint').hidden=game.level!=='13';
 if(game.level==='13')$('rewind-hint').innerHTML=`New feature: Press <span class="rewind-key">Shift</span> to rewind. <span class="rewind-budget">Backups remaining: ${game.rewind_remaining ?? 0}</span>`;
 const out=$('outcome');out.hidden=!game.outcome||(game.outcome==='lose'&&!lossVisible);$('outcome-title').textContent=game.outcome==='win'?'YOU WIN!':'YOU LOSE!';out.classList.toggle('lose',game.outcome==='lose');
 $('replay-level').hidden=game.outcome!=='win';
 $('win-caption').hidden=game.outcome!=='win';
 $('win-caption').textContent=LEVEL_ORDER.indexOf(game.level)<LEVEL_ORDER.length-1?'Next level in 5 seconds':'All levels complete';
 const effects=[], m=game.map, state=game.state;
 for(const it of m.items)if(it.cell===state.X&&!state[it.name])effects.push('pick up the '+(m.colors[it.name]||'')+' key');
 if(m.sources[state.X]===0)effects.push('eat from the tree');
 if(m.sources[state.X]===1)effects.push('drink from the lake');
 for(const door of m.doors){
  const distance=Math.abs(door.cell%m.cols-state.X%m.cols)+Math.abs(Math.floor(door.cell/m.cols)-Math.floor(state.X/m.cols));
  const hasKey=Object.entries(m.colors).some(([name,color])=>color===door.color&&state[name]);
  if(distance===1&&!state['door_'+door.id]&&hasKey)effects.push('open the '+door.color+' door');
 }
 $('interaction-hint').textContent=game.outcome?'Press R to restart.':effects.length?'Press Space to '+effects.join(' and ')+'.':'';
 $('interaction-hint').classList.toggle('ready',!!effects.length&&!game.outcome);
 $('inventory').replaceChildren();
 function card(name,label,icon,value,max){const el=document.createElement('div');el.className='card'+(!value?' empty':'');const img=document.createElement('img');img.src='/static/img/'+icon;img.alt='';const title=document.createElement('strong');title.textContent=label;const count=document.createElement('span');count.className='value';count.textContent=max?`${value} / ${max}`:value?'Held':'Empty';el.append(img,title,count);if(max){const bar=document.createElement('div');bar.className='meter';const fill=document.createElement('i');fill.style.width=`${100*value/max}%`;fill.style.background=name==='hunger'?'#a2d07e':'#6bbcf5';bar.append(fill);el.append(bar);}$('inventory').append(el);}
 for(const p of game.map.physiology)card(p.name,p.name,p.name==='hunger'?'apple.png':'water_drops.png',game.state[p.name],p.max);
 for(const it of game.map.items){const color=game.map.colors[it.name];card(it.name,color?color[0].toUpperCase()+color.slice(1)+' key':it.name,color?'key_'+color+'.png':'key.png',game.state[it.name]);}
 draw();
 if(game.outcome!=='lose')cancelLossRestart();
 if(confirmed){scheduleNextLevel();scheduleLossRestart();refreshLeaderboard();showNameWelcome();}
}
function controls(){document.querySelectorAll('button').forEach(b=>b.disabled=b.hasAttribute('data-action')&&(!game||!!game.outcome));$('restart').disabled=!game;canvas.setAttribute('aria-busy',String(busy));}
// One request in flight; previews may advance over several unconfirmed inputs.
const pendingInputs=[];
let confirmedGame=null, inFlight=null;
const stateKey=state=>JSON.stringify(Object.keys(state).sort().map(k=>[k,state[k]]));
function prediction(base,action){return base?.preview_graph?.[stateKey(base.state)]?.[action]||base?.transitions?.[action];}
function predict(base,action){
 const p=prediction(base,action);if(!p||base.outcome)return base;
 return {...base,...p,moves:base.moves+(p.counted===false?0:1),transitions:{}};
}
function displayPending(){
 game=confirmedGame;
 if(!game)return;
 for(const input of [inFlight,...pendingInputs]){
  if(!input)continue;
  if(input.operation!=='step')break;
  const next=predict(game,input.extra.action);
  if(next===game)break;
  game=next;
 }
 render(false);
}
async function request(operation,extra={}){
 if(operation==='start'||operation==='restart'||operation==='rewind'){cancelLossRestart();cancelAdvance();}
 if(operation==='step'&&(!game||(busy?game.outcome:(confirmedGame||game).outcome)))return;
 if(busy){
  if(operation==='start'||operation==='restart'||operation==='rewind'){pendingInputs.length=0;pendingInputs.push({operation,extra});}
  else if(operation==='step'&&pendingInputs.length<3&&!pendingInputs.some(p=>p.operation!=='step')){
   pendingInputs.push({operation,extra});displayPending();
  }
  else if(operation==='step')$('status').textContent='Waiting for the server — this input was not queued.';
  return;
 }
 if(operation==='start'||operation==='restart'||operation==='rewind'){cancelAdvance();pendingInputs.length=0;}
 confirmedGame=confirmedGame||game;
 const before=confirmedGame, revision=before?.revision;
 let expired=false,received=false,failed=false;
 busy=true;inFlight={operation,extra};$('status').textContent='';
 displayPending();controls();
 try{
  const response=await fetch('/api/play',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation,token,revision,...extra})});
  const data=await response.json();expired=response.status===410;
  if(!response.ok){pendingInputs.length=0;throw Error(data.error||'Unable to take that action.');}
  if(data.game){
   received=true;confirmedGame=data.game;inFlight=null;game=confirmedGame;
   if(operation==='restart'&&!game.display_name)welcomeDismissed=!!extra.automatic;
   if(operation==='resume'&&!LEVEL_ORDER.includes(game.level))pendingInputs.push({operation:'start',extra:{level:LEVEL_ORDER[0]}});
   if(data.token){token=data.token;sessionStorage.setItem('okbe-play-token',token);}
   // Sync the real clock, then draw only the reconciled position (no backward flash).
   syncRunClock(true);
   if(game.outcome){const control=pendingInputs.find(p=>p.operation!=='step');pendingInputs.length=0;if(control)pendingInputs.push(control);}
   displayPending();
   if(!pendingInputs.length){game=confirmedGame;render();}
  }
 }catch(error){
  failed=true;pendingInputs.length=0;inFlight=null;
  // A lost response may follow a saved action. Resume instead of resending it.
  if(operation==='step'&&!expired){
   try{
    const response=await fetch('/api/play',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation:'resume',token})});
    const data=await response.json();if(response.ok&&data.game){confirmedGame=data.game;received=true;}
   }catch(_ignored){}
  }
  game=confirmedGame||before;if(game)render(received);
  $('status').textContent=error.message+' — pending inputs cleared; refresh if disconnected.';
 }finally{
  inFlight=null;busy=false;controls();
  if(expired){confirmedGame=null;token=null;sessionStorage.removeItem('okbe-play-token');request('start',{level:LEVEL_ORDER[0]});}
  else if(!failed){const next=pendingInputs.shift();if(next)request(next.operation,next.extra);}
 }
}
// Fixed cadence: ignore OS key-repeat events (which start slowly, then accelerate).
const MOVE_INTERVAL_MS = 250;
const arrowActions = {ArrowUp:4, ArrowDown:2, ArrowLeft:5, ArrowRight:3};
let heldArrow = null, moveTimer = null;
function stopMoving(){heldArrow=null;if(moveTimer!==null)clearInterval(moveTimer);moveTimer=null;}
function heldStep(){
 if(!game || game.outcome){stopMoving();return;}
 if(heldArrow && (!busy || (pendingInputs.length<2 && prediction(game,arrowActions[heldArrow]))))request('step',{action:arrowActions[heldArrow]});
}
document.querySelectorAll('[data-level]').forEach(b=>b.onclick=()=>{stopMoving();request('start',{level:b.dataset.level});});
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>{stopMoving();request('step',{action:Number(b.dataset.action)});});
$('restart').onclick=()=>{stopMoving();request('restart');};
if(!$('shift-key')){const key=document.createElement('span');key.id='shift-key';key.className='keyboard-key';key.innerHTML='<kbd>Shift</kbd> rewind';$('restart').after(key);}
$('replay-level').onclick=replayLevel;
document.addEventListener('keydown',e=>{
 if($('name-welcome').open){stopMoving();return;}
 if(e.target?.closest?.('input,textarea,select,[contenteditable="true"]')){stopMoving();return;}
 if(e.ctrlKey||e.metaKey||e.altKey){stopMoving();return;}
 if(e.code==='ShiftLeft'||e.code==='ShiftRight'){
  e.preventDefault();stopMoving();if(!e.repeat&&game&&game.level==='13')request('rewind');return;
 }
 if(e.code==='KeyR'){
  e.preventDefault();stopMoving();if(!e.repeat&&game)request('restart');return;
 }
 if(!game)return;
 if(e.code in arrowActions){
  e.preventDefault();
  if(e.repeat || game.outcome || heldArrow===e.code)return;
  stopMoving();heldArrow=e.code;request('step',{action:arrowActions[heldArrow]});
  if(heldArrow)moveTimer=setInterval(heldStep,MOVE_INTERVAL_MS);
 }else if(e.code==='Space'){
  e.preventDefault();stopMoving();
  if(!e.repeat&&!game.outcome)request('step',{action:0});
 }
});
document.addEventListener('keyup',e=>{if(e.code===heldArrow){e.preventDefault();stopMoving();}});
window.addEventListener('blur',()=>{stopMoving();pendingInputs.length=0;});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopMoving();pendingInputs.length=0;}});
// Public scores contain only volunteered names and winning action counts.
let leaderboardKey=null,leaderboardRequest=0,leaderboardLoading=false;
const LEVEL_PRIZES={'12':5,'11':10,'13':20};
const profileNote=document.querySelector('.player-profile small');
if(profileNote)profileNote.textContent='Your best winning score will appear on the leaderboard. Fewest actions wins; earliest solve breaks ties. Movement and Space both count. Your name is optional.';
function formatRunTime(seconds,digits=3){
 if(seconds===null||seconds===undefined)return '—';
 const scale=10**digits,ticks=Math.round(Math.max(0,seconds)*scale),minutes=Math.floor(ticks/(60*scale));
 return minutes+':'+((ticks%(60*scale))/scale).toFixed(digits).padStart(3+digits,'0');
}
async function refreshLeaderboard(force=false){
 if(!game)return;
 const key=game.level+':'+(game.outcome||'');
 if(key===leaderboardKey&&(!force||leaderboardLoading))return;
 const changed=key!==leaderboardKey;leaderboardLoading=true;
 leaderboardKey=key;const current=++leaderboardRequest,level=game.level;
 $('leaderboard-level').textContent=game.title;
 $('leaderboard-prize').textContent=`First-place prize: $${LEVEL_PRIZES[level]}`;
 const footer=document.querySelector('.standings-footer');
 if(footer)footer.innerHTML='Lowest action count wins.<br>Equal actions? Earliest solve wins.<br>Clock shows run duration.';
 if(changed)$('leaderboard-rows').textContent='Loading…';
 try{
  const response=await fetch('/api/leaderboard');if(!response.ok)throw Error();
  const data=await response.json();if(current!==leaderboardRequest)return;
  syncContestClock(data.competition);
  const rows=data.levels[level]||[],root=$('leaderboard-rows');root.replaceChildren();
  if(!rows.length){root.textContent='No winning runs yet. Be the first!';return;}
  const table=document.createElement('table'),head=document.createElement('tr');
  for(const label of ['#','Player','Actions','Run time']){const th=document.createElement('th');th.textContent=label;head.append(th);}table.append(head);
  for(const row of rows){const tr=document.createElement('tr');if(row.rank<=3)tr.className='rank-'+row.rank;for(const value of [row.rank,row.name,row.steps,formatRunTime(row.seconds)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);}root.append(table);
 }catch(error){if(current===leaderboardRequest){$('leaderboard-rows').textContent='Leaderboard temporarily unavailable.';leaderboardKey=null;}}
 finally{if(current===leaderboardRequest)leaderboardLoading=false;}
}
// Optional name; editing this field never sends game actions.
$('player-name').onfocus=()=>{stopMoving();pendingInputs.length=0;};
$('player-name').oninput=()=>{$('player-name').dataset.dirty='1';$('name-status').textContent='';};
$('player-profile').onsubmit=async e=>{
 e.preventDefault();stopMoving();
 if(!game){$('name-status').textContent='Please wait for the game to load.';return;}
 const input=$('player-name'),name=input.value;
 $('save-name').disabled=true;$('name-status').textContent='Saving…';
 try{
  const response=await fetch('/api/player-name',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  const data=await response.json();if(!response.ok)throw Error(data.error||'Could not save name.');
  if(game){rememberNameChoice();game.display_name=data.name;game.name_choice_made=true;game.pseudonym=data.pseudonym;game.needs_prize_name=game.needs_prize_name&&!data.name;if(confirmedGame)Object.assign(confirmedGame,{display_name:data.name,name_choice_made:true,pseudonym:data.pseudonym,needs_prize_name:game.needs_prize_name});$('prize-name-reminder').hidden=!game.needs_prize_name;}
  if(input.value===name){input.value=data.name;delete input.dataset.dirty;}
  $('name-status').textContent=data.name?'Name saved':'Playing anonymously';
  leaderboardKey=null;refreshLeaderboard();
 }catch(error){$('name-status').textContent=error.message;}
 finally{$('save-name').disabled=false;}
};
// Welcome prompt: store either choice against the existing browser ID.
function showNameWelcome(){
 if(!game||game.leaderboard_excluded||welcomeDismissed||game.display_name||$('name-welcome').open)return;
 stopMoving();pendingInputs.length=0;
 $('name-welcome').showModal();$('welcome-name').focus();
}
let savingWelcome=false;
async function chooseWelcome(anonymous){
 if(savingWelcome)return;
 const name=anonymous?'':$('welcome-name').value.trim();
 if(!anonymous&&!name){$('welcome-error').textContent='Enter a name, or choose Skip for now.';return;}
 savingWelcome=true;$('welcome-error').textContent='';
 $('welcome-join').disabled=true;$('welcome-skip').disabled=true;
 try{
  const response=await fetch('/api/player-name',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  const data=await response.json();if(!response.ok)throw Error(data.error||'Could not save your choice.');
  rememberNameChoice();game.display_name=data.name;game.name_choice_made=true;
  $('player-name').value=data.name;delete $('player-name').dataset.dirty;
  $('name-status').textContent=data.name?'Name saved':'Playing anonymously';
  $('name-welcome').close();canvas.focus();leaderboardKey=null;refreshLeaderboard();
 }catch(error){$('welcome-error').textContent=error.message;}
 finally{savingWelcome=false;$('welcome-join').disabled=false;$('welcome-skip').disabled=false;}
}
$('welcome-form').onsubmit=e=>{e.preventDefault();chooseWelcome(false);};
$('welcome-skip').onclick=()=>chooseWelcome(true);
$('name-welcome').addEventListener('cancel',e=>{e.preventDefault();chooseWelcome(true);});
// Refresh other players' scores without interrupting play or flickering the table.
setInterval(()=>{if(!document.hidden)refreshLeaderboard(true);},15000);
window.addEventListener('focus',()=>refreshLeaderboard(true));
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refreshLeaderboard(true);});
// Display elapsed time locally between authoritative server snapshots.
let clockBase=0,clockAnchor=0,clockRunning=false;
function runSeconds(){return clockBase+(clockRunning?(performance.now()-clockAnchor)/1000:0);}
function drawRunClock(){$('run-time').textContent=formatRunTime(runSeconds(),1);}
function syncRunClock(confirmed){
 if(confirmed){clockBase=game.elapsed_seconds||0;clockRunning=!!game.timer_running;clockAnchor=performance.now();}
 else if(!game.moves){clockBase=0;clockRunning=false;}
 else if(game.outcome){clockBase=runSeconds();clockRunning=false;}
 else if(!clockRunning){clockAnchor=performance.now();clockRunning=true;}
 drawRunClock();
}
setInterval(drawRunClock,100);
// Prize deadline; server time corrects differences in the player's device clock.
let contestDeadline=Date.parse('2026-09-19T06:59:00Z'),contestClockOffset=0;
function syncContestClock(info){
 if(!info)return;
 contestDeadline=Date.parse(info.deadline);contestClockOffset=Date.parse(info.server_now)-Date.now();drawContestClock();
}
function drawContestClock(){
 const left=Math.max(0,Math.ceil((contestDeadline-Date.now()-contestClockOffset)/1000));
 const label=$('contest-countdown');label.classList.toggle('closed',left===0);
 if(!left){label.textContent='Competition closed · play for fun';return;}
 const days=Math.floor(left/86400),hours=Math.floor(left%86400/3600),minutes=Math.floor(left%3600/60),seconds=left%60;
 label.textContent=(days?days+'d ':'')+String(hours).padStart(2,'0')+':'+String(minutes).padStart(2,'0')+':'+String(seconds).padStart(2,'0')+' left';
}
setInterval(drawContestClock,1000);drawContestClock();
// Initial game
controls();if(token)request('resume');else request('start',{level:LEVEL_ORDER[0]});
