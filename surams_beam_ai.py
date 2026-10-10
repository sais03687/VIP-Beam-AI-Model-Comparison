"""VIP Part 2: SciPy baseline vs Ollama Llama 3.2 1B and OpenAI GPT agents.

Preserves the lecture's objective/constraints and the student's initial beam test.
All physical measurements come exclusively from the supplied beam_model evaluator.
"""
import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from beam_model import (evaluate_beam, WIDTH_BOUNDS_MM, HEIGHT_BOUNDS_MM,
                        MAX_DEFLECTION_MM, MAX_VON_MISES_STRESS_MPA)

START = (50.0, 80.0)
FIELDS = ['method','model','run','iteration','width_mm','height_mm','mass_kg',
          'max_deflection_mm','max_von_mises_stress_MPa','feasible','response_status',
          'raw_response','elapsed_seconds','best_feasible_mass_kg']

# Adapted from the student's beam_evaluation_VIP_fa26.py and Lecture 6.
def objective(x):
    return evaluate_beam(float(x[0]), float(x[1]))['mass_kg']

def deflection_constraint(x):
    return 1 - evaluate_beam(float(x[0]), float(x[1]))['max_deflection_mm'] / MAX_DEFLECTION_MM

def stress_constraint(x):
    return 1 - evaluate_beam(float(x[0]), float(x[1]))['max_von_mises_stress_MPa'] / MAX_VON_MISES_STRESS_MPA


def record(method, model, run, iteration, dimensions, status, raw='', elapsed=0, best=None, outputs=None):
    w, h = dimensions if dimensions is not None else ('', '')
    return dict(zip(FIELDS, [method, model, run, iteration, w, h,
        outputs['mass_kg'] if outputs else '',
        outputs['max_deflection_mm'] if outputs else '',
        outputs['max_von_mises_stress_MPa'] if outputs else '',
        outputs['feasible'] if outputs else '', status, raw, round(elapsed, 4),
        best if best is not None else '']))


def better(best, outputs):
    if outputs['feasible'] and (best is None or outputs['mass_kg'] < best):
        return outputs['mass_kg']
    return best


def numerical_run():
    rows=[]; count=0; best=None; tic=time.perf_counter()
    baseline=evaluate_beam(*START); count+=1; best=better(best, baseline)
    rows.append(record('SciPy SLSQP', 'scipy.optimize.minimize', 1, 0, START, 'starting_design',
                       elapsed=time.perf_counter()-tic, outputs=baseline, best=best))
    def measured_objective(x):
        nonlocal count, best
        out=evaluate_beam(float(x[0]),float(x[1])); count+=1; best=better(best,out)
        rows.append(record('SciPy SLSQP','scipy.optimize.minimize',1,len(rows),
                           (float(x[0]),float(x[1])),'evaluated',
                           elapsed=time.perf_counter()-tic,outputs=out,best=best))
        return out['mass_kg']
    # Constraint evaluations also use supplied evaluator. Log each call, including repeats.
    def measured_constraint(x, which):
        nonlocal count,best
        out=evaluate_beam(float(x[0]),float(x[1])); count+=1; best=better(best,out)
        rows.append(record('SciPy SLSQP','scipy.optimize.minimize',1,len(rows),
                           (float(x[0]),float(x[1])),f'evaluated_{which}',
                           elapsed=time.perf_counter()-tic,outputs=out,best=best))
        if which=='deflection': return 1-out['max_deflection_mm']/MAX_DEFLECTION_MM
        return 1-out['max_von_mises_stress_MPa']/MAX_VON_MISES_STRESS_MPA
    result=minimize(fun=measured_objective,x0=list(START), method='SLSQP',
        bounds=[WIDTH_BOUNDS_MM,HEIGHT_BOUNDS_MM],
        constraints=[{'type':'ineq','fun':lambda x: measured_constraint(x,'deflection')},
                     {'type':'ineq','fun':lambda x: measured_constraint(x,'stress')}],
        options={'maxiter':200,'ftol':1e-9})
    # The final returned x usually was evaluated already, but evaluate explicitly for verified output.
    out=evaluate_beam(float(result.x[0]),float(result.x[1])); count+=1; best=better(best,out)
    if not out['feasible'] and (out['max_deflection_mm']-MAX_DEFLECTION_MM)<1e-8 and out['max_von_mises_stress_MPa']<=MAX_VON_MISES_STRESS_MPA:
        # Explicit tiny interior adjustment; preserve raw optimizer result in metadata.
        from math import nextafter, inf
        adjusted=(float(result.x[0]),float(result.x[1]))
        for _ in range(100):
            adjusted=(adjusted[0],nextafter(adjusted[1],inf))
            new=evaluate_beam(*adjusted);count+=1;best=better(best,new)
            rows.append(record('SciPy SLSQP','scipy.optimize.minimize',1,len(rows),adjusted,
                               'roundoff_feasibility_check',elapsed=time.perf_counter()-tic,
                               best=best,outputs=new))
            if new['feasible']:
                out=new
                break
    rows.append(record('SciPy SLSQP','scipy.optimize.minimize',1,len(rows),
        (adjusted if out['feasible'] and 'adjusted' in locals() else tuple(map(float,result.x))),
        'final_design',elapsed=time.perf_counter()-tic,best=best,outputs=out))
    return rows, {'success': bool(result.success), 'message':str(result.message),
                  'evaluations':count,'elapsed_seconds':time.perf_counter()-tic,
                  'final_design':[float(v) for v in result.x], 'final_outputs':out}


def parse_proposal(raw):
    """Strictly require one JSON object containing exactly width_mm and height_mm.
    No extraction from prose, clipping, rounding, or repairing invalid proposals.
    """
    try:
        data=json.loads(raw.strip())
    except (ValueError,TypeError): return None,'invalid_json'
    if not isinstance(data,dict) or set(data)!={'width_mm','height_mm'}:
        return None,'invalid_schema'
    vals=[data['width_mm'],data['height_mm']]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in vals):
        return None,'invalid_type'
    import math
    if not all(math.isfinite(v) for v in vals): return None,'invalid_number'
    w,h=map(float,vals)
    if not (WIDTH_BOUNDS_MM[0]<=w<=WIDTH_BOUNDS_MM[1] and
            HEIGHT_BOUNDS_MM[0]<=h<=HEIGHT_BOUNDS_MM[1]):
        return None,'out_of_bounds'
    return (w,h),'evaluated'


def prompt_for(history):
    return ("You are selecting the next steel beam cross-section. Minimize mass while "
            "satisfying BOTH limits. Width 20 to 100 mm, height 40 to 120 mm. "
            "Maximum deflection 1.0 mm and maximum stress 120 MPa. "
            "The only evaluator is the provided beam_model function. "
            "Here are prior evaluations and attempted proposals (most recent last):\n"
            +json.dumps(history,ensure_ascii=True,default=str)
            +"\nPropose the NEXT width and height in millimeters using feedback above. "
             "Reply ONLY with one JSON object with exactly two numeric keys, "
             "width_mm and height_mm. Do not include markdown or explanation.")


def ollama_query(prompt, model, endpoint, temperature, timeout):
    payload={'model':model,'prompt':prompt,'stream':False,
             'options':{'temperature':temperature,'num_predict':180}}
    req=urllib.request.Request(endpoint.rstrip('/')+'/api/generate',
        json.dumps(payload).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as res:
        return json.load(res)['response']


def openai_query(prompt, model, temperature, timeout):
    key=os.getenv('OPENAI_API_KEY')
    if not key: raise RuntimeError('OPENAI_API_KEY environment variable is not set')
    url=os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1').rstrip('/')+'/chat/completions'
    payload={'model':model,'messages':[{'role':'user','content':prompt}],
             'temperature':temperature,'max_completion_tokens':250}
    req=urllib.request.Request(url,json.dumps(payload).encode(),
        {'Content-Type':'application/json','Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=timeout) as res:
        return json.load(res)['choices'][0]['message']['content'] or ''


def ai_run(provider, model, run, proposals, temp, endpoint, timeout):
    rows=[]; history=[]; best=None; tic=time.perf_counter(); invalid=0; evals=0
    out=evaluate_beam(*START); evals+=1; best=better(best,out)
    rows.append(record(provider,model,run,0,START,'starting_design',outputs=out,best=best))
    history.append({'iteration':0,'width_mm':START[0],'height_mm':START[1],**out})
    for index in range(1,proposals+1):
        raw=''; dims=None; outputs=None
        try:
            message=prompt_for(history)
            if provider=='Ollama Llama':
                raw=ollama_query(message,model,endpoint,temp,timeout)
            else:
                raw=openai_query(message,model,temp,timeout)
            dims,status=parse_proposal(raw)
        except (urllib.error.URLError,TimeoutError,RuntimeError,KeyError,ValueError) as exc:
            status='service_error'; raw=type(exc).__name__+': '+str(exc)
        if dims is not None:
            outputs=evaluate_beam(*dims); evals+=1; best=better(best,outputs)
            history.append({'iteration':index,'width_mm':dims[0], 'height_mm':dims[1],**outputs})
        else:
            invalid+=1
            history.append({'iteration':index,'status':status,'raw_reply':raw[:400]})
        rows.append(record(provider,model,run,index,dims,status,raw[:2000],
                           time.perf_counter()-tic,best,outputs))
    return rows, {'evaluations':evals,'invalid_replies':invalid,
                  'elapsed_seconds':time.perf_counter()-tic,'best_mass_kg':best}


def write_csv(rows,path):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)


def generate_plot(rows,path):
    fig,ax=plt.subplots(figsize=(9,5))
    for (method,run) in sorted(set((r['method'],r['run']) for r in rows)):
        if method=='SciPy SLSQP': continue
        part=[r for r in rows if r['method']==method and r['run']==run]
        xs=[r['iteration'] for r in part]
        ys=[float(r['best_feasible_mass_kg']) if r['best_feasible_mass_kg']!='' else float('nan') for r in part]
        ax.step(xs,ys,where='post',marker='o',label=f'{method}, run {run}')
    ax.set(xlabel='Proposal number (0 = common starting design)',
           ylabel='Best feasible mass (kg)',title='AI search history (actual evaluated proposals)')
    if ax.lines: ax.legend(fontsize=9)
    else: ax.text(.5,.5,'No AI agent runs yet',transform=ax.transAxes,ha='center',va='center')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def summary_for(rows,meta):
    groups={}
    for r in rows: groups.setdefault((r['method'],r['model'],r['run']),[]).append(r)
    results=[]
    for (method,model,run),part in groups.items():
        feasible=[r for r in part if r['feasible'] is True]
        best=min(feasible,key=lambda r:r['mass_kg']) if feasible else None
        results.append({'method':method,'model':model,'run':run,'best':best,
            'evals':sum(isinstance(r['feasible'],bool) for r in part),
            'invalid':sum(r['response_status'] not in ('evaluated','starting_design','final_design')
                 and not r['response_status'].startswith('evaluated_') for r in part),
            'elapsed':float(part[-1]['elapsed_seconds'])})
    return results


def make_report(rows,meta,path,figure,github_url):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_CENTER
    styles=getSampleStyleSheet();body=styles['BodyText'];heading=styles['Heading2']
    story=[Paragraph('VIP Individual Project Part 2: Beam Design and AI Model Comparison',styles['Title']),Spacer(1,12)]
    story += [Paragraph('Experiment status',heading)]
    ai_count=sum(r['method']!='SciPy SLSQP' and r['iteration']>0 for r in rows)
    expected=4*meta['proposals']
    status=('Complete experimental records' if ai_count==expected and expected>0 else
            f'PARTIAL RESULTS: {ai_count}/{expected} AI proposals recorded. Live model runs still required.')
    story += [Paragraph(status,body),Spacer(1,8),Paragraph(
        'All mass, stress and deflection values come from the instructor-supplied beam_model.py. '
        'Dimensions are passed in millimeters, following the callable and assignment example.',body),Spacer(1,10)]
    story += [Paragraph('Results comparison',heading)]
    tab=[['Method / model / run','Best kg','Width mm','Height mm','Defl. margin mm','Stress margin MPa','Evals','Invalid','Time s']]
    for s in summary_for(rows,meta):
        b=s['best'];m=b['mass_kg'] if b else None
        tab.append([f"{s['method']} / {s['model']} / {s['run']}",
          f'{m:.4f}' if b else 'none',f"{b['width_mm']:.3f}" if b else '-',
          f"{b['height_mm']:.3f}" if b else '-',
          f"{MAX_DEFLECTION_MM-b['max_deflection_mm']:.4f}" if b else '-',
          f"{MAX_VON_MISES_STRESS_MPA-b['max_von_mises_stress_MPa']:.3f}" if b else '-',
          str(s['evals']),str(s['invalid']),f"{s['elapsed']:.2f}"])
    t=Table(tab,repeatRows=1,colWidths=[145,45,46,48,57,57,36,39,40]);t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF5')),
        ('GRID',(0,0),(-1,-1),.35,colors.grey),('FONTSIZE',(0,0),(-1,-1),7),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story += [t,Spacer(1,12),Paragraph('AI search histories',heading),Image(str(figure),width=510,height=283),Spacer(1,10)]
    optimizer=meta.get('optimizer',{})
    story += [Paragraph('Numerical optimizer',heading),Paragraph(
       f"SLSQP convergence success: {optimizer.get('success','not run')}. "
       f"Message: {optimizer.get('message','not available')}. "
       f"Actual evaluator calls, counting objective, constraint and final calls: {optimizer.get('evaluations','-')}.",body),Spacer(1,10)]
    story += [Paragraph('Comparison and limitations',heading),Paragraph(
        'The shared evaluator uses ideal small-deflection beam theory and excludes contact effects, self-weight and impacts. '
        'SciPy selects candidates numerically; each AI model sees prior evaluations and proposes the next design. '
        'Invalid model replies consume proposal opportunities and receive no beam evaluation. '
        'Model quality cannot be concluded without all four actual AI runs. '
        'Run-to-run variation, repeated designs and improvement from feedback must be interpreted from completed logs. '
        'This comparison is not an Ansys finite-element comparison.',body),Spacer(1,10)]
    story += [Paragraph('Reproducibility',heading),Paragraph(
        'Common starting design: 50 x 80 mm. Each agent run: '
        f"{meta['proposals']} proposals, temperature {meta['temperature']}, shared prompt and strict JSON validator. "
        f"Requested Ollama identifier: {meta['llama_model']}; requested OpenAI identifier: {meta['gpt_model']}. "
        'Python packages: scipy, matplotlib, reportlab; Ollama local server and OpenAI API key required. '
        f"GitHub repository: {github_url if github_url else '[ADD YOUR ACTUAL REPOSITORY URL BEFORE SUBMITTING]'}.",body)]
    doc=SimpleDocTemplate(str(path),pagesize=(612,792),rightMargin=26,leftMargin=26,topMargin=30,bottomMargin=30)
    doc.build(story)


def run(args):
    base=Path(args.output_dir);base.mkdir(parents=True,exist_ok=True)
    rows,optimization=numerical_run()
    meta={'optimizer':optimization,'proposals':args.proposals,'temperature':args.temperature,
          'llama_model':args.llama_model,'gpt_model':args.gpt_model}
    if not args.optimizer_only:
        for provider,model in [('Ollama Llama',args.llama_model),('OpenAI GPT',args.gpt_model)]:
            for trial in (1,2):
                added,stats=ai_run(provider,model,trial,args.proposals,args.temperature,
                                   args.ollama_url,args.timeout)
                rows.extend(added)
                print(f'{provider} run {trial}: {stats}',flush=True)
                write_csv(rows,base/'surams_beam_ai.csv')  # durable partial results
    write_csv(rows,base/'surams_beam_ai.csv')
    generate_plot(rows,base/'surams_beam_ai_search_history.png')
    make_report(rows,meta,base/'surams_beam_ai.pdf',base/'surams_beam_ai_search_history.png',args.github_url)
    (base/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Optimizer:',optimization)
    print('Files saved to:',base.resolve())
    return rows,meta


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--optimizer-only',action='store_true',help='Run SciPy baseline only; AI findings remain incomplete')
    p.add_argument('--output-dir',default='.')
    p.add_argument('--llama-model',default='llama3.2:1b')
    p.add_argument('--gpt-model',default='gpt-4.1-mini')
    p.add_argument('--ollama-url',default='http://localhost:11434')
    p.add_argument('--temperature',type=float,default=0.2)
    p.add_argument('--timeout',type=int,default=120)
    p.add_argument('--proposals',type=int,default=11)
    p.add_argument('--github-url',default='')
    return p.parse_args()

if __name__=='__main__': run(cli())
