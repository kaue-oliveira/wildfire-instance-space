"""Recompute saved solutions, 444 cells, 42 summaries and 15 pairwise tests."""
from pathlib import Path
import csv,json,hashlib,heapq,math,statistics,collections,sys,itertools
from decimal import Decimal
P=Path(__file__).resolve().parents[1]
B=P/'results/benchmark'
D=P/'results/stn'
errors=[];out={}
def readcsv(p):return list(csv.DictReader(p.open()))
def close(a,b,loc):
 if not math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9):errors.append([loc,a,b])
def centi(t):
 v=Decimal(str(t))*100
 assert v==v.to_integral_value() and v>=0, t
 return int(v)
def loadnative(p):
 d=json.loads(p.read_text(),parse_float=Decimal);n=d['|V|'];adj=[[] for _ in range(n)]
 for u,v,w in d['arcs']:
  assert 0<=u<n and 0<=v<n and w>0
  adj[u].append((v,centi(w)))
 assert len(d['t'])==len(d['c'])==len(d['delta'])
 assert len(set(map(centi,d['delta'])))==1
 return dict(n=n,adj=adj,I=d['I'],H=centi(d['H']),delay=centi(d['delta'][0]),capacity={centi(t):c for t,c in zip(d['t'],d['c'])})
def objective(m,alloc):
 if len(alloc)!=len(set(v for v,t in alloc)):raise ValueError('vertice repetido')
 for v,t in alloc:
  if not 0<=v<m['n']:raise ValueError('indice invalido')
 for t,c in collections.Counter(t for v,t in alloc).items():
  if t not in m['capacity'] or c>m['capacity'][t]:raise ValueError('capacidade/liberacao')
 used={v for v,t in alloc};arrival=[10**40]*m['n'];q=[]
 for v in m['I']:arrival[v]=0;heapq.heappush(q,(0,v))
 while q:
  t,u=heapq.heappop(q)
  if t!=arrival[u]:continue
  extra=m['delay'] if u in used else 0
  for v,w in m['adj'][u]:
   at=t+w+extra
   if at<arrival[v]:arrival[v]=at;heapq.heappush(q,(at,v))
 if any(arrival[v]<t for v,t in alloc):raise ValueError('fogo antes da alocacao')
 return sum(t<m['H'] for t in arrival)
rows=readcsv(P/'results/tables/analysis_index_final.csv');bench=[r for r in rows if r['phase']=='benchmark']
assert len(bench)==3936 and len(rows)==4126
assert len({(r['phase'],r['instance'],r['algorithm'],r['seed']) for r in rows})==len(rows)
models={};coords={};counts=collections.Counter();deviations=[];infeasible=[];cellruns=collections.defaultdict(list);inputchecks=0
for name in sorted({r['instance'] for r in bench}):
 for form in ['native','harris']:
  p1=P/f'data/instances/{form}/{name}.json'
  assert p1.is_file(),p1;inputchecks+=1
 models[name]=loadnative(P/f'data/instances/native/{name}.json')
 h=json.loads((P/f'data/instances/harris/{name}.json').read_text())
 coords[name]={tuple(x):i for i,x in enumerate(h['Nodes'])}
 assert len(coords[name])==models[name]['n']
for k,r in enumerate(bench):
 p=P/r['source_artifact_local'];result=json.loads(p.read_text());m=models[r['instance']]
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source_artifact_sha256']
 assert result['instance']==r['instance'] and str(result['seed'])==r['seed'] and result['algorithm']==r['algorithm']
 assert hashlib.sha256(json.dumps(json.loads((P/f"data/instances/native/{r['instance']}.json").read_text()),ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()==r['instance_sha256']
 if 'solution_file' in result:
  solpath=p.parent/result['solution_file'];sol=json.loads(solpath.read_text(),parse_float=Decimal)
  assert hashlib.sha256(solpath.read_bytes()).hexdigest()==result['solution_sha256']
  alloc=[(int(v),centi(a['time'])) for a in sol['allocation'].values() for v in a['protected']]
 else:
  alloc=[(coords[r['instance']][tuple(v)],centi(t)) for v,t in result.get('solution',[])]
 try:score=objective(m,alloc);valid=True
 except ValueError as e:score=objective(m,[]);valid=False;infeasible.append([r['job_id'],r['instance'],r['algorithm'],str(e)])
 if not valid and r['successful_run']=='True':errors.append(['solucao indicada valida mas invalida',r['job_id']])
 close(r['analysis_objective'],score,'obj '+r['job_id'])
 if r['objective'] and float(r['objective'])!=score:deviations.append([r['job_id'],r['algorithm'],r['objective'],score])
 counts[r['algorithm']]+=1;cellruns[r['instance'],r['algorithm']].append((r,score))
 if (k+1)%700==0:print('Soluções avaliadas',k+1,flush=True)
out['benchmark_solutions']={'rows':len(bench),'input_files_checked':inputchecks,'counts':dict(counts),'infeasible_attempts':infeasible,'raw_objective_differences':deviations}
names=sorted(models);algs=sorted(counts);bkv={n:min(s for a in algs for r,s in cellruns[n,a]) for n in names};cells={}
for (n,a),group in cellruns.items():
 vals=[s for r,s in group];mean=statistics.mean(vals);means=[statistics.mean(s for r,s in cellruns[n,b]) for b in algs]
 cells[n,a]={'instance':n,'algorithm':a,'collection':group[0][0]['collection'],'size_class':group[0][0]['size_class'],'objective_mean':mean,'objective_median':statistics.median(vals),'objective_min':min(vals),'objective_max':max(vals),'objective_sd':statistics.stdev(vals),'runs':len(vals),'failures':sum(r['successful_run']!='True' for r,s in group),'bkv_all':bkv[n],'bkv6':bkv[n],'gap_mean_pct':100*(mean-bkv[n])/bkv[n],'rank_mean_objective':1+sum(x<mean for x in means)+(sum(x==mean for x in means)-1)/2,'best_mean_tie':mean==min(means),'best_mean_unique':mean==min(means) and means.count(mean)==1}
for c in readcsv(P/'results/tables/per_instance_algorithm_decimal.csv'):
 for key,v in cells[c['instance'],c['algorithm']].items():
  if isinstance(v,bool):assert (c[key]=='True')==v
  elif isinstance(v,(int,float)):close(c[key],v,'celula '+c['instance']+'/'+c['algorithm']+'/'+key)
scopes={'all74':names,'original55':[n for n in names if cells[n,'ibs']['collection']=='original_55'],'supplementary19':[n for n in names if cells[n,'ibs']['collection']=='supplementary_19']}
scopes.update({s:[n for n in names if cells[n,'ibs']['size_class']==s] for s in ['Small','Medium','Large','Huge']})
summaries=[]
for r in readcsv(P/'results/tables/algorithm_summary_decimal.csv'):
 cs=[cells[n,r['algorithm']] for n in scopes[r['scope']]]
 calc={'instances':len(cs),'runs':sum(c['runs'] for c in cs),'failures':sum(c['failures'] for c in cs),'mean_gap_pct_equal_instances':statistics.mean(c['gap_mean_pct'] for c in cs),'median_gap_pct_equal_instances':statistics.median(c['gap_mean_pct'] for c in cs),'mean_rank':statistics.mean(c['rank_mean_objective'] for c in cs),'best_mean_ties':sum(c['best_mean_tie'] for c in cs),'best_mean_unique':sum(c['best_mean_unique'] for c in cs)}
 for key,v in calc.items():close(r[key],v,'resumo '+r['scope']+'/'+r['algorithm']+'/'+key)
 summaries.append({'scope':r['scope'],'algorithm':r['algorithm'],**calc})
paired=[]
for a,b in itertools.combinations(algs,2):
 wins=sum(cells[n,a]['objective_mean']<cells[n,b]['objective_mean'] for n in names);losses=sum(cells[n,a]['objective_mean']>cells[n,b]['objective_mean'] for n in names);N=wins+losses
 p=min(1,2*sum(math.comb(N,k) for k in range(min(wins,losses)+1))/2**N) if N else 1
 paired.append({'algorithm_a':a,'algorithm_b':b,'wins_a':wins,'ties':74-N,'wins_b':losses,'exact_sign_test_two_sided':p,'mean_gap_difference_pct_a_minus_b':statistics.mean(cells[n,a]['gap_mean_pct']-cells[n,b]['gap_mean_pct'] for n in names)})
prev=0
for rank,r in enumerate(sorted(paired,key=lambda r:r['exact_sign_test_two_sided'])):
 prev=max(prev,min(1,(15-rank)*r['exact_sign_test_two_sided']));r['holm_adjusted_sign_p']=prev
for r in readcsv(P/'results/tables/paired_instance_comparisons_decimal.csv'):
 calc=next(p for p in paired if p['algorithm_a']==r['algorithm_a'] and p['algorithm_b']==r['algorithm_b'])
 for key,v in calc.items():
  if isinstance(v,(int,float)):close(r[key],v,'teste '+r['algorithm_a']+'/'+r['algorithm_b']+'/'+key)
out['summaries']=summaries;out['paired_comparisons']=paired;out['cells']=list(cells.values());out['errors']=errors
output=P/'experiments/verification'
output.mkdir(parents=True,exist_ok=True)
(output/'benchmark.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'benchmark':out['benchmark_solutions'],'checked_cells':len(cells),'checked_summaries':len(summaries),'checked_tests':len(paired),'errors':errors[:15],'n_errors':len(errors)},ensure_ascii=False,indent=2),flush=True)

if errors: raise SystemExit(1)
